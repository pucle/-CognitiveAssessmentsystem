# -*- coding: utf-8 -*-
"""
Enhanced Feature Engineering Module for Integrated Cognitive Assessment Pipeline
===============================================================================

Features:
- MMSE normalization theo age và education
- Multi-modal feature integration (acoustic, linguistic, demographic, clinical)
- Missing data handling (MICE/KNN)
- Feature selection (correlation + RFE)
- Normalization và scaling

Based on literature review:
- Petersen RC et al. (2018): MMSE cần normalize theo age/education
- Battista P et al. (2020): Multi-modal features tăng accuracy 15-25%
- Rathore S et al. (2017): Feature selection critical để tránh overfitting

Author: Cognitive Assessment System
Version: 2.0
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

logger = logging.getLogger(__name__)

# Try importing MICE
try:
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    MICE_AVAILABLE = True
except ImportError:
    MICE_AVAILABLE = False
    logger.warning("MICE imputation not available - will use KNN instead")


class MMSENormalizer:
    """
    Normalize MMSE scores theo age và education.
    
    Based on:
    - Creavin ST et al. (2016): MMSE có bias theo age và education
    - ADNI normalization formulas
    """
    
    @staticmethod
    def normalize_mmse(
        mmse_raw: float,
        age: float,
        education_years: float,
        method: str = 'adni'
    ) -> Dict[str, float]:
        """
        Normalize MMSE score theo age và education.
        
        Args:
            mmse_raw: Raw MMSE score (0-30)
            age: Age in years
            education_years: Years of education
            method: 'adni' (ADNI formula) or 'simple' (simple adjustment)
        
        Returns:
            Dict với normalized scores và adjustments
        """
        if pd.isna(mmse_raw) or pd.isna(age) or pd.isna(education_years):
            return {
                'mmse_raw': mmse_raw,
                'mmse_adjusted': mmse_raw,
                'education_adjustment': 0.0,
                'age_adjustment': 0.0,
                'normalization_method': method
            }
        
        if method == 'adni':
            # ADNI formula: Adjusted = Raw + (Education - 12) * 0.3 - (Age - 65) * 0.1
            education_adjustment = (education_years - 12) * 0.3
            age_adjustment = -(age - 65) * 0.1
        else:
            # Simple adjustment
            # Education: 0-6 years: +1, 7-12: 0, 13+: -1
            if education_years <= 6:
                education_adjustment = 1.0
            elif education_years <= 12:
                education_adjustment = 0.0
            else:
                education_adjustment = -1.0
            
            # Age: <65: 0, 65-74: -1, 75-84: -2, 85+: -3
            if age < 65:
                age_adjustment = 0.0
            elif age < 75:
                age_adjustment = -1.0
            elif age < 85:
                age_adjustment = -2.0
            else:
                age_adjustment = -3.0
        
        mmse_adjusted = mmse_raw + education_adjustment + age_adjustment
        mmse_adjusted = np.clip(mmse_adjusted, 0, 30)  # Clip to valid range
        
        return {
            'mmse_raw': float(mmse_raw),
            'mmse_adjusted': float(mmse_adjusted),
            'education_adjustment': float(education_adjustment),
            'age_adjustment': float(age_adjustment),
            'normalization_method': method
        }


class IntegratedFeatureEngineer:
    """
    Enhanced feature engineering cho integrated cognitive assessment pipeline.
    
    Handles:
    1. MMSE normalization
    2. Multi-modal feature integration
    3. Missing data imputation
    4. Feature selection
    5. Scaling
    """
    
    def __init__(
        self,
        mmse_normalization_method: str = 'adni',
        imputation_method: str = 'knn',
        feature_selection_method: str = 'rfe',
        n_features_to_select: int = 100,
        correlation_threshold: float = 0.9,
        scaler_type: str = 'standard'
    ):
        """
        Initialize feature engineer.
        
        Args:
            mmse_normalization_method: 'adni' or 'simple'
            imputation_method: 'mice', 'knn', or 'median'
            feature_selection_method: 'rfe', 'kbest', or 'none'
            n_features_to_select: Number of features to select
            correlation_threshold: Threshold for removing correlated features
            scaler_type: 'standard' or 'robust'
        """
        self.mmse_normalizer = MMSENormalizer()
        self.mmse_normalization_method = mmse_normalization_method
        self.imputation_method = imputation_method
        self.feature_selection_method = feature_selection_method
        self.n_features_to_select = n_features_to_select
        self.correlation_threshold = correlation_threshold
        self.scaler_type = scaler_type
        
        # Fitted components
        self.scaler = None
        self.imputer = None
        self.feature_selector = None
        self.selected_features = None
        self.is_fitted = False
        
        # Feature groups
        self.numerical_features = []
        self.categorical_features = []
        self.feature_groups = {
            'acoustic': [],
            'linguistic': [],
            'demographic': [],
            'clinical': [],
            'lifestyle': []
        }
    
    def _identify_feature_groups(self, X: pd.DataFrame) -> Dict[str, List[str]]:
        """Identify feature groups từ column names."""
        groups = {
            'acoustic': [],
            'linguistic': [],
            'demographic': ['age', 'gender', 'education', 'education_years'],
            'clinical': ['mmse', 'mmse_raw', 'mmse_adjusted', 'mmsediff'],
            'lifestyle': []
        }
        
        for col in X.columns:
            col_lower = col.lower()
            
            # Acoustic features
            if any(prefix in col_lower for prefix in ['acoustic_', 'f0_', 'vq_', 'pause_', 'rate_', 'tone_', 'smile_']):
                groups['acoustic'].append(col)
            # Linguistic features
            elif any(prefix in col_lower for prefix in ['linguistic_', 'lex_', 'syn_', 'sem_', 'emo_']):
                groups['linguistic'].append(col)
            # Demographic
            elif col_lower in groups['demographic']:
                groups['demographic'].append(col)
            # Clinical
            elif col_lower in groups['clinical'] or 'mmse' in col_lower:
                groups['clinical'].append(col)
            # Lifestyle (if available)
            elif any(term in col_lower for term in ['physical', 'diet', 'sleep', 'exercise']):
                groups['lifestyle'].append(col)
        
        return groups
    
    def _normalize_mmse_scores(self, X: pd.DataFrame) -> pd.DataFrame:
        """Normalize MMSE scores trong dataset."""
        X = X.copy()
        
        # Check if we have MMSE and demographic data
        has_mmse = any('mmse' in col.lower() for col in X.columns)
        has_age = 'age' in X.columns
        has_education = any('education' in col.lower() for col in X.columns)
        
        if not (has_mmse and has_age):
            logger.warning("Missing MMSE or age data - skipping MMSE normalization")
            return X
        
        # Find MMSE column
        mmse_col = None
        for col in X.columns:
            if col.lower() in ['mmse', 'mmse_raw', 'mmse_score']:
                mmse_col = col
                break
        
        if mmse_col is None:
            logger.warning("No MMSE column found - skipping normalization")
            return X
        
        # Find education column
        education_col = None
        for col in X.columns:
            if 'education' in col.lower():
                education_col = col
                break
        
        if education_col is None:
            logger.warning("No education column found - using default (12 years)")
            education_years = 12.0
        else:
            education_years = X[education_col].fillna(12.0)
        
        # Normalize MMSE
        age = X['age'].fillna(65.0)
        mmse_raw = X[mmse_col].fillna(25.0)
        
        normalized_results = []
        for idx in X.index:
            result = self.mmse_normalizer.normalize_mmse(
                mmse_raw.loc[idx],
                age.loc[idx],
                education_years.loc[idx] if isinstance(education_years, pd.Series) else education_years,
                method=self.mmse_normalization_method
            )
            normalized_results.append(result)
        
        # Add normalized MMSE column
        X['mmse_adjusted'] = [r['mmse_adjusted'] for r in normalized_results]
        X['mmse_education_adj'] = [r['education_adjustment'] for r in normalized_results]
        X['mmse_age_adj'] = [r['age_adjustment'] for r in normalized_results]
        
        logger.info(f"✅ Normalized MMSE scores for {len(X)} samples")
        
        return X
    
    def _handle_missing_values(self, X: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Handle missing values với MICE, KNN, hoặc median imputation."""
        X = X.copy()
        
        # Identify numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            logger.warning("No numeric columns found")
            return X
        
        missing_counts = X[numeric_cols].isna().sum()
        total_missing = missing_counts.sum()
        
        if total_missing == 0:
            logger.info("No missing values found")
            return X
        
        logger.info(f"Handling {total_missing} missing values ({total_missing / (len(X) * len(numeric_cols)) * 100:.1f}%)")
        
        if is_training:
            # Fit imputer
            if self.imputation_method == 'mice' and MICE_AVAILABLE:
                self.imputer = IterativeImputer(
                    max_iter=10,
                    random_state=42,
                    n_nearest_features=min(10, len(numeric_cols))
                )
                X[numeric_cols] = self.imputer.fit_transform(X[numeric_cols])
                logger.info("✅ Fitted MICE imputer")
            elif self.imputation_method == 'knn':
                self.imputer = KNNImputer(n_neighbors=5)
                X[numeric_cols] = self.imputer.fit_transform(X[numeric_cols])
                logger.info("✅ Fitted KNN imputer")
            else:
                self.imputer = SimpleImputer(strategy='median')
                X[numeric_cols] = self.imputer.fit_transform(X[numeric_cols])
                logger.info("✅ Fitted median imputer")
        else:
            # Transform with fitted imputer
            if self.imputer is None:
                logger.warning("Imputer not fitted - using median imputation")
                X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
            else:
                X[numeric_cols] = self.imputer.transform(X[numeric_cols])
        
        return X
    
    def _remove_correlated_features(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Remove highly correlated features."""
        X = X.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return X, []
        
        # Calculate correlation matrix
        corr_matrix = X[numeric_cols].corr().abs()
        
        # Find pairs with correlation > threshold
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = []
        for col in upper_triangle.columns:
            high_corr = upper_triangle.index[upper_triangle[col] > self.correlation_threshold].tolist()
            if high_corr:
                # Keep the first one, drop others
                to_drop.extend(high_corr)
        
        to_drop = list(set(to_drop))  # Remove duplicates
        
        if to_drop:
            logger.info(f"Removing {len(to_drop)} highly correlated features (r > {self.correlation_threshold})")
            X = X.drop(columns=to_drop)
        
        return X, to_drop
    
    def _select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'rfe'
    ) -> pd.DataFrame:
        """Select features using RFE or SelectKBest."""
        if method == 'none' or len(X.columns) <= self.n_features_to_select:
            self.selected_features = list(X.columns)
            return X
        
        logger.info(f"Selecting {self.n_features_to_select} features using {method}...")
        
        if method == 'rfe':
            # Use Random Forest for RFE
            estimator = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                random_state=42,
                n_jobs=-1
            )
            self.feature_selector = RFE(
                estimator=estimator,
                n_features_to_select=self.n_features_to_select,
                step=10
            )
            self.feature_selector.fit(X, y)
            self.selected_features = X.columns[self.feature_selector.support_].tolist()
            X_selected = X[self.selected_features]
            logger.info(f"✅ Selected {len(self.selected_features)} features using RFE")
        
        elif method == 'kbest':
            self.feature_selector = SelectKBest(
                score_func=f_classif,
                k=min(self.n_features_to_select, len(X.columns))
            )
            self.feature_selector.fit(X, y)
            self.selected_features = X.columns[self.feature_selector.get_support()].tolist()
            X_selected = X[self.selected_features]
            logger.info(f"✅ Selected {len(self.selected_features)} features using SelectKBest")
        
        else:
            self.selected_features = list(X.columns)
            X_selected = X
        
        return X_selected
    
    def _fit_scaler(self, X: pd.DataFrame):
        """Fit scaler on training data."""
        if self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        
        self.scaler.fit(X)
        logger.info(f"✅ Fitted {self.scaler_type} scaler")
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        do_mmse_normalization: bool = True,
        do_feature_selection: bool = True
    ) -> pd.DataFrame:
        """
        Full preprocessing pipeline for training data.
        
        Steps:
        1. MMSE normalization
        2. Handle missing values
        3. Remove correlated features
        4. Feature selection
        5. Fit scaler
        6. Transform
        """
        logger.info("="*60)
        logger.info("INTEGRATED FEATURE ENGINEERING - TRAINING")
        logger.info("="*60)
        
        # Step 1: MMSE normalization
        if do_mmse_normalization:
            X = self._normalize_mmse_scores(X)
        
        # Step 2: Handle missing values
        X = self._handle_missing_values(X, is_training=True)
        
        # Step 3: Remove correlated features
        X, dropped = self._remove_correlated_features(X)
        
        # Step 4: Feature selection
        if do_feature_selection and self.feature_selection_method != 'none':
            X = self._select_features(X, y, method=self.feature_selection_method)
        else:
            self.selected_features = list(X.columns)
        
        # Step 5: Fit scaler
        self._fit_scaler(X)
        
        # Step 6: Transform
        X_transformed = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Identify feature groups
        self.feature_groups = self._identify_feature_groups(X_transformed)
        self.numerical_features = list(X_transformed.columns)
        
        self.is_fitted = True
        
        logger.info(f"✅ Final features: {X_transformed.shape}")
        logger.info(f"   - Acoustic: {len(self.feature_groups['acoustic'])}")
        logger.info(f"   - Linguistic: {len(self.feature_groups['linguistic'])}")
        logger.info(f"   - Demographic: {len(self.feature_groups['demographic'])}")
        logger.info(f"   - Clinical: {len(self.feature_groups['clinical'])}")
        
        return X_transformed
    
    def transform(self, X: pd.DataFrame, do_mmse_normalization: bool = True) -> pd.DataFrame:
        """Transform new data using fitted pipeline."""
        if not self.is_fitted:
            raise ValueError("Feature engineer not fitted. Call fit_transform() first.")
        
        # Step 1: MMSE normalization
        if do_mmse_normalization:
            X = self._normalize_mmse_scores(X)
        
        # Step 2: Handle missing values
        X = self._handle_missing_values(X, is_training=False)
        
        # Step 3: Remove correlated features (same as training)
        # Note: We don't re-calculate correlation, just drop same features if they exist
        
        # Step 4: Apply feature selection
        if self.selected_features:
            missing_features = set(self.selected_features) - set(X.columns)
            if missing_features:
                logger.warning(f"Missing {len(missing_features)} features in input, filling with 0")
                for feat in missing_features:
                    X[feat] = 0.0
            
            X = X[self.selected_features]
        else:
            # If no selection was done, keep all features
            pass
        
        # Step 5: Transform with scaler
        if self.scaler is None:
            raise ValueError("Scaler not fitted")
        
        X_transformed = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        
        return X_transformed
    
    def save(self, path: str):
        """Save fitted feature engineer."""
        import joblib
        joblib.dump(self, path)
        logger.info(f"✅ Saved feature engineer to {path}")
    
    @staticmethod
    def load(path: str) -> 'IntegratedFeatureEngineer':
        """Load fitted feature engineer."""
        import joblib
        engineer = joblib.load(path)
        logger.info(f"✅ Loaded feature engineer from {path}")
        return engineer

