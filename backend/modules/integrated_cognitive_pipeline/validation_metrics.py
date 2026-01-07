# -*- coding: utf-8 -*-
"""
Validation Metrics và Calibration Module
========================================

Comprehensive validation metrics cho dual-output model:
1. Binary Classification Metrics: Accuracy, Sensitivity, Specificity, F1-score, AUC-ROC
2. Probability Estimation Metrics: Brier score, Calibration plot, Expected Calibration Error (ECE)
3. Calibration Methods: Platt scaling, Isotonic regression
4. Cross-validation: 5-fold CV, Temporal validation

Dựa trên literature review:
- Rathore S et al. (2017): Cross-validation essential cho clinical models
- Barnes DE et al. (2009): Clinical utility metrics (Net Benefit Analysis)

Author: Cognitive Assessment System
Version: 2.0
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    brier_score_loss, classification_report
)
from sklearn.calibration import (
    CalibratedClassifierCV, calibration_curve
)
from sklearn.model_selection import (
    StratifiedKFold, TimeSeriesSplit, cross_val_score
)
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logger.warning("Matplotlib not available - calibration plots will be skipped")


@dataclass
class BinaryClassificationMetrics:
    """Metrics cho binary classification (Risk Yes/No)."""
    accuracy: float
    sensitivity: float  # Recall/TPR
    specificity: float  # TNR
    precision: float
    f1_score: float
    auc_roc: float
    confusion_matrix: np.ndarray
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'accuracy': self.accuracy,
            'sensitivity': self.sensitivity,
            'specificity': self.specificity,
            'precision': self.precision,
            'f1_score': self.f1_score,
            'auc_roc': self.auc_roc
        }


@dataclass
class ProbabilityEstimationMetrics:
    """Metrics cho probability estimation (MCI Probability)."""
    brier_score: float
    expected_calibration_error: float
    mean_absolute_error: float
    calibration_curve_data: Optional[Dict[str, np.ndarray]] = None
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'brier_score': self.brier_score,
            'expected_calibration_error': self.expected_calibration_error,
            'mean_absolute_error': self.mean_absolute_error
        }


class ValidationMetrics:
    """
    Comprehensive validation metrics cho dual-output model.
    """
    
    def __init__(self):
        """Initialize validation metrics calculator."""
        pass
    
    def calculate_binary_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> BinaryClassificationMetrics:
        """
        Calculate binary classification metrics.
        
        Args:
            y_true: True binary labels (0 = No Risk, 1 = Risk)
            y_pred: Predicted binary labels
            y_proba: Predicted probabilities (for AUC-ROC)
        
        Returns:
            BinaryClassificationMetrics object
        """
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Specificity = TN / (TN + FP)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # AUC-ROC
        auc_roc = 0.0
        if y_proba is not None and len(np.unique(y_true)) > 1:
            try:
                if len(y_proba.shape) > 1:
                    y_proba = y_proba[:, 1]  # Use positive class probability
                auc_roc = roc_auc_score(y_true, y_proba)
            except Exception as e:
                logger.warning(f"Could not calculate AUC-ROC: {e}")
                auc_roc = 0.0
        
        return BinaryClassificationMetrics(
            accuracy=float(accuracy),
            sensitivity=float(recall),
            specificity=float(specificity),
            precision=float(precision),
            f1_score=float(f1),
            auc_roc=float(auc_roc),
            confusion_matrix=cm
        )
    
    def calculate_probability_estimation_metrics(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        n_bins: int = 10
    ) -> ProbabilityEstimationMetrics:
        """
        Calculate probability estimation metrics.
        
        Args:
            y_true: True probability targets (0-1)
            y_pred_proba: Predicted probabilities (0-1)
            n_bins: Number of bins for calibration curve
        
        Returns:
            ProbabilityEstimationMetrics object
        """
        # Brier score
        brier_score = brier_score_loss(y_true, y_pred_proba)
        
        # Mean Absolute Error
        mae = np.mean(np.abs(y_true - y_pred_proba))
        
        # Expected Calibration Error (ECE)
        ece = self._calculate_ece(y_true, y_pred_proba, n_bins)
        
        # Calibration curve data
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_pred_proba, n_bins=n_bins, strategy='uniform'
            )
            calibration_curve_data = {
                'fraction_of_positives': fraction_of_positives,
                'mean_predicted_value': mean_predicted_value
            }
        except Exception as e:
            logger.warning(f"Could not calculate calibration curve: {e}")
            calibration_curve_data = None
        
        return ProbabilityEstimationMetrics(
            brier_score=float(brier_score),
            expected_calibration_error=float(ece),
            mean_absolute_error=float(mae),
            calibration_curve_data=calibration_curve_data
        )
    
    def _calculate_ece(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        
        ECE = Σ (|Bm| / n) * |acc(Bm) - conf(Bm)|
        
        Where:
        - Bm: bin m
        - acc(Bm): accuracy in bin m
        - conf(Bm): average confidence in bin m
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Accuracy in this bin
                accuracy_in_bin = y_true[in_bin].mean()
                # Average confidence in this bin
                avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                # Weighted ECE
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return float(ece)
    
    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_path: Optional[str] = None,
        title: str = "Calibration Curve"
    ):
        """
        Plot calibration curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save plot (optional)
            title: Plot title
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib not available - skipping calibration plot")
            return
        
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_pred_proba, n_bins=10, strategy='uniform'
            )
            
            plt.figure(figsize=(8, 6))
            plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
            plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
            plt.xlabel("Mean Predicted Probability")
            plt.ylabel("Fraction of Positives")
            plt.title(title)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"✅ Saved calibration plot to {save_path}")
            else:
                plt.show()
            
            plt.close()
        except Exception as e:
            logger.error(f"Failed to plot calibration curve: {e}")


class CrossValidationWrapper:
    """
    Cross-validation wrapper cho dual-output model.
    """
    
    def __init__(self, cv_type: str = 'stratified', n_splits: int = 5):
        """
        Initialize cross-validation wrapper.
        
        Args:
            cv_type: 'stratified' or 'temporal'
            n_splits: Number of folds
        """
        self.cv_type = cv_type
        self.n_splits = n_splits
        
        if cv_type == 'stratified':
            self.cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        elif cv_type == 'temporal':
            self.cv = TimeSeriesSplit(n_splits=n_splits)
        else:
            raise ValueError(f"Unknown CV type: {cv_type}")
    
    def cross_validate(
        self,
        model_trainer,
        X: pd.DataFrame,
        y: pd.Series,
        metrics_calculator: ValidationMetrics
    ) -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Args:
            model_trainer: DualOutputModelTrainer instance
            X: Features DataFrame
            y: Labels Series
            metrics_calculator: ValidationMetrics instance
        
        Returns:
            Dictionary with CV results
        """
        binary_metrics_list = []
        prob_metrics_list = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(self.cv.split(X, y)):
            logger.info(f"\n📊 Fold {fold_idx + 1}/{self.n_splits}")
            
            # Split data
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]
            
            # Train model
            model_trainer.train(
                X_train_fold, y_train_fold,
                X_val=X_val_fold, y_val=y_val_fold,
                use_calibration=True
            )
            
            # Predict
            predictions = model_trainer.predict(X_val_fold)
            
            # Prepare true labels
            y_binary = model_trainer._prepare_binary_labels(y_val_fold)
            y_probability = model_trainer._prepare_probability_targets(y_val_fold)
            
            # Calculate metrics
            binary_metrics = metrics_calculator.calculate_binary_classification_metrics(
                y_binary,
                predictions.risk_binary.astype(int),
                y_proba=np.array([predictions.risk_binary_probability] * len(y_binary))
            )
            binary_metrics_list.append(binary_metrics)
            
            prob_metrics = metrics_calculator.calculate_probability_estimation_metrics(
                y_probability,
                np.array([predictions.mci_probability] * len(y_probability))
            )
            prob_metrics_list.append(prob_metrics)
        
        # Aggregate results
        return {
            'binary_classification': {
                'mean_accuracy': np.mean([m.accuracy for m in binary_metrics_list]),
                'std_accuracy': np.std([m.accuracy for m in binary_metrics_list]),
                'mean_sensitivity': np.mean([m.sensitivity for m in binary_metrics_list]),
                'std_sensitivity': np.std([m.sensitivity for m in binary_metrics_list]),
                'mean_specificity': np.mean([m.specificity for m in binary_metrics_list]),
                'std_specificity': np.std([m.specificity for m in binary_metrics_list]),
                'mean_f1': np.mean([m.f1_score for m in binary_metrics_list]),
                'std_f1': np.std([m.f1_score for m in binary_metrics_list]),
                'mean_auc_roc': np.mean([m.auc_roc for m in binary_metrics_list]),
                'std_auc_roc': np.std([m.auc_roc for m in binary_metrics_list])
            },
            'probability_estimation': {
                'mean_brier_score': np.mean([m.brier_score for m in prob_metrics_list]),
                'std_brier_score': np.std([m.brier_score for m in prob_metrics_list]),
                'mean_ece': np.mean([m.expected_calibration_error for m in prob_metrics_list]),
                'std_ece': np.std([m.expected_calibration_error for m in prob_metrics_list]),
                'mean_mae': np.mean([m.mean_absolute_error for m in prob_metrics_list]),
                'std_mae': np.std([m.mean_absolute_error for m in prob_metrics_list])
            },
            'n_folds': self.n_splits,
            'cv_type': self.cv_type
        }


# Convenience functions
def calculate_dual_output_metrics(
    y_true: pd.Series,
    predictions: Any,  # DualOutputPrediction or list of predictions
    metrics_calculator: Optional[ValidationMetrics] = None
) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics cho dual-output predictions.
    
    Args:
        y_true: True labels (multi-class: Normal/MCI/AD)
        predictions: DualOutputPrediction or list of predictions
        metrics_calculator: Optional ValidationMetrics instance
    
    Returns:
        Dictionary with all metrics
    """
    if metrics_calculator is None:
        metrics_calculator = ValidationMetrics()
    
    # Convert predictions to arrays
    if hasattr(predictions, 'risk_binary'):
        # Single prediction
        y_binary_pred = np.array([predictions.risk_binary])
        y_binary_proba = np.array([predictions.risk_binary_probability])
        y_mci_proba = np.array([predictions.mci_probability])
    else:
        # List of predictions
        y_binary_pred = np.array([p.risk_binary for p in predictions])
        y_binary_proba = np.array([p.risk_binary_probability for p in predictions])
        y_mci_proba = np.array([p.mci_probability for p in predictions])
    
    # Prepare true labels
    # Convert multi-class to binary
    y_binary_true = np.array([
        0 if label in ['Normal', 'normal', 'NORMAL', 0] else 1
        for label in y_true
    ])
    
    # Prepare probability targets
    y_probability_true = np.array([
        0.0 if label in ['Normal', 'normal', 'NORMAL', 0] else
        0.6 if label in ['MCI', 'mci'] else
        0.9 if label in ['AD', 'ad', 'Alzheimer', 'Dementia'] else 0.5
        for label in y_true
    ])
    
    # Calculate metrics
    binary_metrics = metrics_calculator.calculate_binary_classification_metrics(
        y_binary_true, y_binary_pred, y_binary_proba
    )
    
    prob_metrics = metrics_calculator.calculate_probability_estimation_metrics(
        y_probability_true, y_mci_proba
    )
    
    return {
        'binary_classification': binary_metrics.to_dict(),
        'probability_estimation': prob_metrics.to_dict(),
        'summary': {
            'binary_accuracy': binary_metrics.accuracy,
            'binary_auc_roc': binary_metrics.auc_roc,
            'brier_score': prob_metrics.brier_score,
            'ece': prob_metrics.expected_calibration_error
        }
    }

