# -*- coding: utf-8 -*-
"""
Dual-Output Model Architecture for Integrated Cognitive Assessment
==================================================================

Architecture:
- Shared representation learning (feature extraction)
- Head 1: Binary Classification (Risk Yes/No)
- Head 2: Probability Estimation (MCI Probability 0-1)

Based on literature:
- Barnes DE et al. (2009): Risk stratification cần dual output
- Rathore S et al. (2017): Calibration critical cho probability estimates

Author: Cognitive Assessment System
Version: 2.0
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    brier_score_loss, classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
import joblib

logger = logging.getLogger(__name__)

# Try importing XGBoost and LightGBM
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("XGBoost not available")

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    logger.warning("LightGBM not available")


@dataclass
class DualOutputPrediction:
    """Dual output prediction result.
    
    ⚠️ IMPORTANT: MMSE score KHÔNG được estimate từ model.
    MMSE score phải được lấy từ bài kiểm tra chatbot và pass vào như một feature.
    
    Model chỉ output:
    1. Binary classification: Risk Yes/No
    2. Probability estimation: MCI Probability [0-1]
    """
    # Binary classification
    risk_binary: bool  # True = có nguy cơ, False = không có nguy cơ
    risk_binary_probability: float  # Probability của risk_binary
    
    # Probability estimation
    mci_probability: float  # MCI probability [0-1]
    mci_probability_calibrated: float  # Calibrated probability
    
    # Class prediction (from binary classification)
    predicted_class: str  # 'Normal', 'MCI', 'AD'
    class_probabilities: Dict[str, float]  # Probabilities for each class
    
    # Confidence
    confidence: float  # Overall confidence [0-1]
    
    # Metadata
    model_name: str
    calibration_applied: bool
    
    # Note: MMSE score KHÔNG có trong output - phải lấy từ chatbot test


class DualOutputModelTrainer:
    """
    Trainer cho dual-output model architecture.
    
    Architecture:
    1. Shared base models (RF, XGBoost, etc.)
    2. Binary classification head (Risk Yes/No)
    3. Probability estimation head (MCI Probability)
    4. Calibration cho probability estimates
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        random_state: int = 42
    ):
        """
        Initialize dual-output model trainer.
        
        Args:
            config: Configuration dictionary
            random_state: Random seed
        """
        self.config = config or {}
        self.random_state = random_state
        
        # Models
        self.binary_classifier = None  # Binary: Risk Yes/No
        self.probability_estimator = None  # Regression: MCI Probability
        self.calibrator = None  # Calibration cho probability
        
        # Label encoders
        self.binary_label_encoder = LabelEncoder()
        self.class_label_encoder = LabelEncoder()
        
        # Training results
        self.training_history = {}
        self.is_fitted = False
        self.feature_names_ = None
        
        # Calibration flag
        self.calibration_applied = False
    
    def _prepare_binary_labels(self, y: pd.Series) -> np.ndarray:
        """
        Convert multi-class labels to binary (Risk Yes/No).
        
        Normal -> No Risk (0)
        MCI/AD -> Risk (1)
        """
        binary_labels = []
        for label in y:
            if label in ['Normal', 'normal', 'NORMAL', 0]:
                binary_labels.append(0)  # No Risk
            else:
                binary_labels.append(1)  # Risk (MCI or AD)
        
        return np.array(binary_labels)
    
    def _prepare_probability_targets(self, y: pd.Series) -> np.ndarray:
        """
        Convert class labels to probability targets.
        
        Normal -> 0.0 (no MCI risk)
        MCI -> 0.6 (moderate risk)
        AD -> 0.9 (high risk)
        """
        prob_targets = []
        for label in y:
            if label in ['Normal', 'normal', 'NORMAL', 0]:
                prob_targets.append(0.0)
            elif label in ['MCI', 'mci', 'Mild Cognitive Impairment']:
                prob_targets.append(0.6)
            elif label in ['AD', 'ad', 'Alzheimer', 'Dementia', 'dementia']:
                prob_targets.append(0.9)
            else:
                prob_targets.append(0.5)  # Unknown
        
        return np.array(prob_targets)
    
    def _create_base_models(self) -> Dict[str, Any]:
        """Create base models for ensemble."""
        models = {}
        
        # Random Forest
        models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Gradient Boosting
        models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=self.random_state
        )
        
        # Logistic Regression
        models['logistic_regression'] = LogisticRegression(
            C=1.0,
            penalty='l2',
            class_weight='balanced',
            random_state=self.random_state,
            max_iter=1000
        )
        
        # XGBoost (if available)
        if XGB_AVAILABLE:
            models['xgboost'] = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state,
                eval_metric='mlogloss'
            )
        
        # LightGBM (if available)
        if LGBM_AVAILABLE:
            models['lightgbm'] = LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state,
                verbose=-1
            )
        
        return models
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        use_calibration: bool = True
    ) -> Dict[str, Any]:
        """
        Train dual-output model.
        
        Args:
            X_train: Training features
            y_train: Training labels (Normal/MCI/AD)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            use_calibration: Whether to calibrate probability estimates
        
        Returns:
            Training results dictionary
        """
        logger.info("="*60)
        logger.info("TRAINING DUAL-OUTPUT MODEL")
        logger.info("="*60)
        
        self.feature_names_ = list(X_train.columns)
        
        # Prepare targets
        y_binary = self._prepare_binary_labels(y_train)
        y_probability = self._prepare_probability_targets(y_train)
        
        # Encode class labels
        y_class_encoded = self.class_label_encoder.fit_transform(y_train)
        
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Binary labels - No Risk: {np.sum(y_binary == 0)}, Risk: {np.sum(y_binary == 1)}")
        logger.info(f"Classes: {self.class_label_encoder.classes_}")
        
        # ============================================================
        # TRAIN BINARY CLASSIFIER (Head 1)
        # ============================================================
        logger.info("\n" + "-"*60)
        logger.info("Training Binary Classifier (Risk Yes/No)")
        logger.info("-"*60)
        
        base_models = self._create_base_models()
        
        # Train individual models
        binary_scores = {}
        for name, model in base_models.items():
            try:
                model.fit(X_train, y_binary)
                cv_scores = cross_val_score(
                    model, X_train, y_binary,
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
                    scoring='f1'
                )
                binary_scores[name] = {
                    'model': model,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                logger.info(f"  {name}: CV F1 = {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            except Exception as e:
                logger.warning(f"  {name} failed: {e}")
        
        # Select best model for binary classification
        if binary_scores:
            best_binary_name = max(binary_scores.keys(), key=lambda k: binary_scores[k]['cv_mean'])
            self.binary_classifier = binary_scores[best_binary_name]['model']
            logger.info(f"✅ Selected {best_binary_name} for binary classification")
        else:
            # Fallback to Random Forest
            self.binary_classifier = base_models['random_forest']
            self.binary_classifier.fit(X_train, y_binary)
            logger.warning("Using Random Forest as fallback")
        
        # Evaluate binary classifier
        y_binary_pred = self.binary_classifier.predict(X_train)
        binary_accuracy = accuracy_score(y_binary, y_binary_pred)
        binary_f1 = f1_score(y_binary, y_binary_pred)
        logger.info(f"Binary Classifier - Train Accuracy: {binary_accuracy:.4f}, F1: {binary_f1:.4f}")
        
        # ============================================================
        # TRAIN PROBABILITY ESTIMATOR (Head 2)
        # ============================================================
        logger.info("\n" + "-"*60)
        logger.info("Training Probability Estimator (MCI Probability)")
        logger.info("-"*60)
        
        # Use same base models, but train on probability targets
        prob_scores = {}
        for name, model in base_models.items():
            try:
                # For probability estimation, we use class probabilities
                # Train on multi-class, then extract MCI+AD probability
                model.fit(X_train, y_class_encoded)
                
                # Predict probabilities
                prob_pred = model.predict_proba(X_train)
                
                # Calculate MCI probability (sum of MCI and AD probabilities)
                mci_class_idx = np.where(self.class_label_encoder.classes_ == 'MCI')[0]
                ad_class_idx = np.where(self.class_label_encoder.classes_ == 'AD')[0]
                
                if len(mci_class_idx) > 0 and len(ad_class_idx) > 0:
                    mci_prob_pred = prob_pred[:, mci_class_idx[0]] + prob_pred[:, ad_class_idx[0]]
                elif len(mci_class_idx) > 0:
                    mci_prob_pred = prob_pred[:, mci_class_idx[0]]
                else:
                    # Fallback: use probability of not being Normal
                    normal_class_idx = np.where(self.class_label_encoder.classes_ == 'Normal')[0]
                    if len(normal_class_idx) > 0:
                        mci_prob_pred = 1.0 - prob_pred[:, normal_class_idx[0]]
                    else:
                        mci_prob_pred = prob_pred[:, -1]  # Last class
                
                # Calculate Brier score
                brier = brier_score_loss(y_probability, mci_prob_pred)
                prob_scores[name] = {
                    'model': model,
                    'brier_score': brier
                }
                logger.info(f"  {name}: Brier Score = {brier:.4f}")
            except Exception as e:
                logger.warning(f"  {name} failed: {e}")
        
        # Select best model for probability estimation
        if prob_scores:
            best_prob_name = min(prob_scores.keys(), key=lambda k: prob_scores[k]['brier_score'])
            self.probability_estimator = prob_scores[best_prob_name]['model']
            logger.info(f"✅ Selected {best_prob_name} for probability estimation")
        else:
            # Fallback
            self.probability_estimator = base_models['random_forest']
            self.probability_estimator.fit(X_train, y_class_encoded)
            logger.warning("Using Random Forest as fallback")
        
        # ============================================================
        # CALIBRATION (if requested)
        # ============================================================
        if use_calibration:
            logger.info("\n" + "-"*60)
            logger.info("Calibrating Probability Estimates")
            logger.info("-"*60)
            
            try:
                # Calibrate binary classifier
                self.calibrator = CalibratedClassifierCV(
                    self.binary_classifier,
                    method='isotonic',
                    cv=5
                )
                self.calibrator.fit(X_train, y_binary)
                self.calibration_applied = True
                logger.info("✅ Applied isotonic calibration")
            except Exception as e:
                logger.warning(f"Calibration failed: {e}")
                self.calibration_applied = False
        
        # ============================================================
        # VALIDATION (if provided)
        # ============================================================
        if X_val is not None and y_val is not None:
            logger.info("\n" + "-"*60)
            logger.info("Validation Results")
            logger.info("-"*60)
            
            val_results = self.evaluate(X_val, y_val)
            for metric, value in val_results.items():
                logger.info(f"  {metric}: {value:.4f}")
        
        self.is_fitted = True
        
        # Store training history
        self.training_history = {
            'binary_scores': {k: v['cv_mean'] for k, v in binary_scores.items()},
            'prob_scores': {k: v['brier_score'] for k, v in prob_scores.items()},
            'calibration_applied': self.calibration_applied
        }
        
        logger.info("\n" + "="*60)
        logger.info("✅ TRAINING COMPLETE")
        logger.info("="*60)
        
        return self.training_history
    
    def predict(self, X: pd.DataFrame) -> DualOutputPrediction:
        """
        Make dual-output prediction.
        
        Args:
            X: Features DataFrame
        
        Returns:
            DualOutputPrediction với binary classification và probability estimation
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")
        
        # Ensure feature alignment
        if self.feature_names_:
            missing_features = set(self.feature_names_) - set(X.columns)
            if missing_features:
                logger.warning(f"Missing {len(missing_features)} features, filling with 0")
                for feat in missing_features:
                    X[feat] = 0.0
            X = X[self.feature_names_]
        
        # ============================================================
        # HEAD 1: Binary Classification (Risk Yes/No)
        # ============================================================
        if self.calibrator is not None:
            binary_probs = self.calibrator.predict_proba(X)[0]
        else:
            binary_probs = self.binary_classifier.predict_proba(X)[0]
        
        risk_binary = binary_probs[1] > 0.5  # Risk if probability > 0.5
        risk_binary_probability = float(binary_probs[1])
        
        # ============================================================
        # HEAD 2: Probability Estimation (MCI Probability)
        # ============================================================
        class_probs = self.probability_estimator.predict_proba(X)[0]
        
        # Map to class names
        class_prob_dict = {}
        for i, class_name in enumerate(self.class_label_encoder.classes_):
            class_prob_dict[class_name] = float(class_probs[i])
        
        # Calculate MCI probability (MCI + AD)
        mci_class_idx = np.where(self.class_label_encoder.classes_ == 'MCI')[0]
        ad_class_idx = np.where(self.class_label_encoder.classes_ == 'AD')[0]
        
        mci_probability = 0.0
        if len(mci_class_idx) > 0:
            mci_probability += class_probs[mci_class_idx[0]]
        if len(ad_class_idx) > 0:
            mci_probability += class_probs[ad_class_idx[0]]
        
        if mci_probability == 0.0:
            # Fallback: use probability of not being Normal
            normal_class_idx = np.where(self.class_label_encoder.classes_ == 'Normal')[0]
            if len(normal_class_idx) > 0:
                mci_probability = 1.0 - class_probs[normal_class_idx[0]]
        
        mci_probability = float(mci_probability)
        mci_probability_calibrated = mci_probability  # Same for now (calibration applied to binary)
        
        # Predicted class
        predicted_class_idx = np.argmax(class_probs)
        predicted_class = self.class_label_encoder.classes_[predicted_class_idx]
        
        # Confidence
        confidence = float(max(class_probs))
        
        return DualOutputPrediction(
            risk_binary=bool(risk_binary),
            risk_binary_probability=risk_binary_probability,
            mci_probability=mci_probability,
            mci_probability_calibrated=mci_probability_calibrated,
            predicted_class=predicted_class,
            class_probabilities=class_prob_dict,
            confidence=confidence,
            model_name=f"Dual-Output ({type(self.binary_classifier).__name__} + {type(self.probability_estimator).__name__})",
            calibration_applied=self.calibration_applied
        )
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        """Evaluate model on test data."""
        predictions = self.predict(X)
        
        # Prepare true labels
        y_binary = self._prepare_binary_labels(y)
        y_class_encoded = self.class_label_encoder.transform(y)
        
        # Binary classification metrics
        binary_pred = predictions.risk_binary.astype(int) if isinstance(predictions, DualOutputPrediction) else None
        if binary_pred is not None:
            binary_accuracy = accuracy_score(y_binary, binary_pred)
            binary_f1 = f1_score(y_binary, binary_pred)
        else:
            binary_accuracy = 0.0
            binary_f1 = 0.0
        
        # Class prediction metrics
        class_pred = self.class_label_encoder.transform([predictions.predicted_class])[0] if isinstance(predictions, DualOutputPrediction) else None
        if class_pred is not None:
            class_accuracy = accuracy_score(y_class_encoded, [class_pred] * len(y_class_encoded))
        else:
            class_accuracy = 0.0
        
        # Probability estimation metrics
        y_probability = self._prepare_probability_targets(y)
        mci_prob = predictions.mci_probability if isinstance(predictions, DualOutputPrediction) else 0.5
        brier_score = brier_score_loss(y_probability, [mci_prob] * len(y_probability))
        
        return {
            'binary_accuracy': binary_accuracy,
            'binary_f1': binary_f1,
            'class_accuracy': class_accuracy,
            'brier_score': brier_score
        }
    
    def save(self, path: str):
        """Save trained model."""
        joblib.dump(self, path)
        logger.info(f"✅ Saved dual-output model to {path}")
    
    @staticmethod
    def load(path: str) -> 'DualOutputModelTrainer':
        """Load trained model."""
        model = joblib.load(path)
        logger.info(f"✅ Loaded dual-output model from {path}")
        return model

