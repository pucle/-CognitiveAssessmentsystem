# -*- coding: utf-8 -*-
"""
Integration Service for Vietnamese MCI Screening
Combines all analysis modules into a unified pipeline

Author: Cognitive Assessment System
Version: 1.0

This service provides the main entry point for:
1. Audio analysis (acoustic features)
2. Transcript analysis (linguistic features)
3. Multimodal fusion
4. MCI prediction and MMSE estimation
"""

import logging
import os
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, asdict
import json
import time

logger = logging.getLogger(__name__)

# Import modules - if import succeeds, module will be used (not None)
# If import fails, raise error (no graceful fallback - system requires these modules)
from .acoustic_analyzer import AcousticAnalyzer
from .linguistic_analyzer import VietnameseLinguisticAnalyzer
from .multimodal_fusion import MultimodalFusion, FusionConfig
from .mci_predictor import MCIPredictor, MCIPrediction

# ✅ Phase 11: Import SHAP and Comprehensive Results
try:
    from .mci_shap_explainer import MCIShapExplainer, create_shap_explainer
    from .comprehensive_results import ComprehensiveResultsGenerator, generate_comprehensive_results
    SHAP_AVAILABLE = True
except ImportError as e:
    logger.warning(f"SHAP or Comprehensive Results modules not available: {e}")
    SHAP_AVAILABLE = False

# ✅ Phase 12: Import Integrated Cognitive Pipeline (v2.0)
try:
    from .integrated_cognitive_pipeline.feature_engineer_v2 import IntegratedFeatureEngineer
    from .integrated_cognitive_pipeline.dual_output_model import DualOutputModelTrainer, DualOutputPrediction
    INTEGRATED_PIPELINE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Integrated pipeline modules not available: {e}")
    INTEGRATED_PIPELINE_AVAILABLE = False
    DualOutputModelTrainer = None
    DualOutputPrediction = None


@dataclass
class AnalysisResult:
    """Complete analysis result
    
    ⚠️ IMPORTANT: mmse_score là điểm MMSE từ bài kiểm tra chatbot,
    KHÔNG phải estimate từ model prediction.
    MMSE score được pass vào model như một feature để chẩn đoán.
    """
    success: bool
    acoustic_features: Dict[str, Any]
    linguistic_features: Dict[str, Any]
    fused_features: Dict[str, Any]
    mci_prediction: Optional[Dict[str, Any]]
    mmse_score: float  # MMSE từ chatbot test (KHÔNG phải estimate)
    severity: str
    confidence: float
    risk_factors: list
    recommendations: list
    feature_summary: Dict[str, Any]
    processing_time: float
    errors: list


class MCIScreeningService:
    """
    Main service class for MCI screening
    
    Combines:
    - Acoustic analysis (eGeMAPS + Vietnamese tone features)
    - Linguistic analysis (Vietnamese NLP)
    - Multimodal fusion
    - MCI prediction and MMSE estimation
    
    Usage:
        service = MCIScreeningService()
        result = service.analyze(audio_path="audio.wav", transcript="Xin chào...")
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 use_phobert: bool = True,
                 use_integrated_pipeline: bool = False):
        """
        Initialize MCI Screening Service
        
        Args:
            model_path: Path to pre-trained prediction model (optional)
                       If None, will auto-detect newest model
            use_phobert: Whether to use PhoBERT for semantic analysis
            use_integrated_pipeline: Whether to use new integrated pipeline (v2.0)
                                    with MMSE normalization and dual-output model
        """
        self.errors = []
        self.use_integrated_pipeline = use_integrated_pipeline and INTEGRATED_PIPELINE_AVAILABLE
        
        # Auto-detect newest model if not provided
        if model_path is None:
            model_path = self._find_newest_model()
            if model_path:
                logger.info(f"✅ Auto-detected newest model: {model_path}")
        
        # Initialize acoustic analyzer - if import succeeded, must use it
        try:
            self.acoustic_analyzer = AcousticAnalyzer()
            logger.info("✅ AcousticAnalyzer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize AcousticAnalyzer: {e}")
            self.errors.append(f"AcousticAnalyzer: {e}")
            raise  # If import succeeded but init failed, raise error
        
        # Initialize linguistic analyzer - if import succeeded, must use it
        try:
            self.linguistic_analyzer = VietnameseLinguisticAnalyzer(
                use_phobert=use_phobert
            )
            logger.info("✅ LinguisticAnalyzer initialized (underthesea + PhoBERT)")
        except Exception as e:
            logger.error(f"Failed to initialize LinguisticAnalyzer: {e}")
            self.errors.append(f"LinguisticAnalyzer: {e}")
            raise  # If import succeeded but init failed, raise error
        
        # Initialize multimodal fusion - if import succeeded, must use it
        try:
            config = FusionConfig(
                acoustic_weight=0.5,
                linguistic_weight=0.5,
                fusion_method='early',
                normalize=True
            )
            self.fusion = MultimodalFusion(config)
            logger.info("✅ MultimodalFusion initialized")
        except Exception as e:
            logger.error(f"Failed to initialize MultimodalFusion: {e}")
            self.errors.append(f"MultimodalFusion: {e}")
            raise  # If import succeeded but init failed, raise error
        
        # Initialize predictor with auto-detected newest model - if import succeeded, must use it
        try:
            # Auto-detect newest model if not provided
            if model_path is None:
                model_path = self._find_newest_model()
                if model_path:
                    logger.info(f"✅ Auto-detected newest model: {model_path}")
            
            self.predictor = MCIPredictor(model_path)
            logger.info("✅ MCIPredictor initialized")
        except Exception as e:
            logger.error(f"Failed to initialize MCIPredictor: {e}")
            self.errors.append(f"MCIPredictor: {e}")
            raise  # If import succeeded but init failed, raise error
        
        # ✅ Phase 12: Initialize Integrated Pipeline components (if enabled)
        self.integrated_feature_engineer = None
        self.dual_output_model = None
        if self.use_integrated_pipeline:
            try:
                self.integrated_feature_engineer = IntegratedFeatureEngineer(
                    mmse_normalization_method='adni',
                    imputation_method='knn',
                    feature_selection_method='rfe',
                    n_features_to_select=100,
                    correlation_threshold=0.9,
                    scaler_type='standard'
                )
                logger.info("✅ Integrated Feature Engineer initialized (v2.0)")
                
                # Try to load dual-output model if available
                # (Model sẽ được load khi có trained model file)
                self.dual_output_model = None  # Will be loaded on demand
                logger.info("✅ Dual-output model slot ready (will load on demand)")
            except Exception as e:
                logger.warning(f"Failed to initialize Integrated Feature Engineer: {e}")
                self.use_integrated_pipeline = False
        
        # ✅ Phase 11: Initialize SHAP explainer and comprehensive results generator
        self.shap_explainer: Optional[MCIShapExplainer] = None
        self.results_generator = None
        if SHAP_AVAILABLE:
            try:
                self.results_generator = ComprehensiveResultsGenerator()
                self._init_shap_explainer()
            except Exception as e:
                logger.warning(f"Failed to initialize SHAP/Comprehensive Results: {e}")
        
        logger.info(f"MCIScreeningService initialized (errors: {len(self.errors)}, integrated_pipeline: {self.use_integrated_pipeline})")
    
    def _find_newest_model(self) -> Optional[str]:
        """
        Find the newest model file or directory automatically.
        Priority order:
        1. data/ml_output/ (Latest pipeline results - 89.5% Accuracy)
        2. models/best_model.pkl (newest, 2025-12-16)
        3. model_bundle/model_new_clean/model.pkl (2025-12-11)
        4. model_bundle/model_new/model.pkl (2025-12-11)
        5. models/mci_fusion_model.pkl (if exists)
        """
        from pathlib import Path
        import os
        
        # Get project root (assuming this file is in backend/modules/)
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        
        # ✅ FIX: Priority 1: backend/data/ml_output/ (Phase 11: High-Performance ML Models - 89.5% Accuracy)
        ml_output = project_root / "backend" / "data" / "ml_output"
        models_file = ml_output / "models.joblib"
        eng_file = ml_output / "feature_engineer.joblib"
        if ml_output.exists() and models_file.exists() and eng_file.exists():
            logger.info(f"✅ Found latest high-performance pipeline artifacts: {ml_output}")
            logger.info(f"   - Models: {models_file.exists()}")
            logger.info(f"   - Feature Engineer: {eng_file.exists()}")
            return str(ml_output)
            
        # Try without backend/ just in case (e.g. if root is backend)
        ml_output_alt = project_root / "data" / "ml_output"
        models_file_alt = ml_output_alt / "models.joblib"
        eng_file_alt = ml_output_alt / "feature_engineer.joblib"
        if ml_output_alt.exists() and models_file_alt.exists() and eng_file_alt.exists():
            logger.info(f"✅ Found latest pipeline artifacts (alt): {ml_output_alt}")
            return str(ml_output_alt)
            
        # Priority 2: models/best_model.pkl (newest)
        newest_model = project_root / "models" / "best_model.pkl"
        if newest_model.exists():
            return str(newest_model)
        
        # Priority 3: model_bundle/model_new_clean/model.pkl
        clean_model = project_root / "model_bundle" / "model_new_clean" / "model.pkl"
        if clean_model.exists():
            return str(clean_model)
        
        # Priority 4: model_bundle/model_new/model.pkl
        new_model = project_root / "model_bundle" / "model_new" / "model.pkl"
        if new_model.exists():
            return str(new_model)
        
        # Priority 5: models/mci_fusion_model.pkl
        fusion_model = project_root / "models" / "mci_fusion_model.pkl"
        if fusion_model.exists():
            return str(fusion_model)
        
        logger.warning("⚠️ No model found, will use default untrained models")
        return None
    
    def _apply_integrated_feature_engineering(
        self,
        features: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply integrated feature engineering với MMSE normalization.
        
        Args:
            features: Combined features dict
            metadata: Metadata với age, education, mmse_raw
        
        Returns:
            Enhanced features dict với MMSE normalization
        """
        import pandas as pd
        
        # Ensure we have required metadata for MMSE normalization
        if 'age' not in features and 'age' in metadata:
            features['age'] = metadata['age']
        if 'education_years' not in features and 'education' in metadata:
            features['education_years'] = metadata['education']
        if 'mmse_raw' not in features and 'mmse' in metadata:
            features['mmse_raw'] = metadata['mmse']
        
        # Convert to DataFrame for processing
        df = pd.DataFrame([features])
        
        # Apply MMSE normalization (if engineer is fitted, otherwise just prepare data)
        if self.integrated_feature_engineer and self.integrated_feature_engineer.is_fitted:
            # Transform using fitted engineer
            df_processed = self.integrated_feature_engineer.transform(df, do_mmse_normalization=True)
            # Convert back to dict
            enhanced_features = df_processed.iloc[0].to_dict()
        else:
            # Just normalize MMSE if we have the data
            if 'mmse_raw' in features and 'age' in features and 'education_years' in features:
                from .integrated_cognitive_pipeline.feature_engineer_v2 import MMSENormalizer
                normalizer = MMSENormalizer()
                mmse_result = normalizer.normalize_mmse(
                    features['mmse_raw'],
                    features['age'],
                    features['education_years'],
                    method='adni'
                )
                features['mmse_adjusted'] = mmse_result['mmse_adjusted']
                features['mmse_education_adj'] = mmse_result['education_adjustment']
                features['mmse_age_adj'] = mmse_result['age_adjustment']
            enhanced_features = features
        
        return enhanced_features
    
    def _predict_with_dual_output(self, features: Dict[str, Any]) -> 'DualOutputPrediction':
        """
        Predict using dual-output model.
        
        Args:
            features: Combined features dict
        
        Returns:
            DualOutputPrediction object
        """
        import pandas as pd
        
        if self.dual_output_model is None:
            raise ValueError("Dual-output model not loaded")
        
        # Convert features to DataFrame
        df = pd.DataFrame([features])
        
        # Transform using feature engineer if available
        if self.integrated_feature_engineer and self.integrated_feature_engineer.is_fitted:
            df_processed = self.integrated_feature_engineer.transform(df, do_mmse_normalization=True)
        else:
            df_processed = df
        
        # Predict
        prediction = self.dual_output_model.predict(df_processed)
        
        return prediction
    
    def _get_mmse_from_features(self, features: Dict[str, Any], metadata: Dict[str, Any]) -> float:
        """
        Get MMSE score từ features hoặc metadata.
        
        ⚠️ IMPORTANT: MMSE score phải được lấy từ bài kiểm tra chatbot,
        KHÔNG được estimate từ model prediction.
        
        Priority:
        1. mmse_raw từ features
        2. mmse từ metadata
        3. mmse_adjusted từ features (nếu có)
        4. Default: 25.0 (assume normal)
        """
        # Priority 1: mmse_raw từ features
        if 'mmse_raw' in features:
            return float(features['mmse_raw'])
        
        # Priority 2: mmse từ metadata
        if metadata and 'mmse' in metadata:
            return float(metadata['mmse'])
        
        # Priority 3: mmse_adjusted từ features
        if 'mmse_adjusted' in features:
            return float(features['mmse_adjusted'])
        
        # Priority 4: Check features directly
        if 'mmse' in features:
            return float(features['mmse'])
        
        # Default: assume normal (will be used for severity classification only)
        logger.warning("⚠️ MMSE score not found in features or metadata, using default 25.0")
        return 25.0
    
    def _classify_severity_from_mmse(self, mmse_score: float) -> str:
        """Classify severity từ MMSE score."""
        if mmse_score >= 24:
            return 'Bình thường'
        elif mmse_score >= 18:
            return 'Suy giảm nhận thức nhẹ (MCI)'
        elif mmse_score >= 10:
            return 'Sa sút trí tuệ mức độ trung bình'
        else:
            return 'Sa sút trí tuệ mức độ nặng'
    
    def _extract_risk_factors_from_dual_output(
        self,
        prediction: 'DualOutputPrediction',
        features: Dict[str, Any]
    ) -> List[str]:
        """Extract risk factors từ dual-output prediction và features."""
        risk_factors = []
        
        # Risk level based on MCI probability
        if prediction.mci_probability >= 0.7:
            risk_factors.append("Nguy cơ suy giảm nhận thức cao (MCI probability ≥ 70%)")
        elif prediction.mci_probability >= 0.4:
            risk_factors.append("Nguy cơ suy giảm nhận thức trung bình (MCI probability ≥ 40%)")
        
        # Age risk
        if 'age' in features and features['age'] >= 75:
            risk_factors.append(f"Tuổi cao ({features['age']} tuổi) là yếu tố nguy cơ")
        
        # Education risk
        if 'education_years' in features and features['education_years'] < 6:
            risk_factors.append(f"Trình độ học vấn thấp ({features['education_years']} năm) là yếu tố nguy cơ")
        
        # MMSE risk
        if 'mmse_adjusted' in features:
            mmse = features['mmse_adjusted']
            if mmse < 24:
                risk_factors.append(f"Điểm MMSE điều chỉnh thấp ({mmse:.1f}/30) chỉ ra suy giảm nhận thức")
        
        return risk_factors
    
    def _generate_recommendations_from_dual_output(
        self,
        prediction: 'DualOutputPrediction',
        mmse_estimate: float
    ) -> List[str]:
        """Generate recommendations từ dual-output prediction."""
        recommendations = []
        
        if prediction.risk_binary:
            if prediction.mci_probability >= 0.7:
                recommendations.append("Phát hiện nguy cơ suy giảm nhận thức cao")
                recommendations.append("Cần đánh giá y tế khẩn cấp bởi bác sĩ chuyên khoa")
                recommendations.append("Khuyến nghị khám chuyên khoa thần kinh hoặc lão khoa")
            elif prediction.mci_probability >= 0.4:
                recommendations.append("Phát hiện dấu hiệu suy giảm nhận thức nhẹ")
                recommendations.append("Khuyến nghị đánh giá chuyên sâu bởi bác sĩ")
                recommendations.append("Cân nhắc chụp MRI não để loại trừ nguyên nhân khác")
                recommendations.append("Khuyến khích các hoạt động kích thích trí não")
            else:
                recommendations.append("Có dấu hiệu nguy cơ nhẹ, cần theo dõi")
                recommendations.append("Khuyến nghị tái đánh giá sau 6-12 tháng")
        else:
            recommendations.append("Kết quả trong giới hạn bình thường")
            recommendations.append("Khuyến nghị tái đánh giá sau 6-12 tháng để theo dõi")
        
        return recommendations
    
    def _init_shap_explainer(self):
        """Initialize SHAP explainer after model is loaded"""
        if not SHAP_AVAILABLE:
            return
        
        try:
            predictor = self.predictor
            if predictor.is_pipeline_model and predictor.model_trainer and predictor.feature_engineer:
                # Load background data (sample from training set)
                from pathlib import Path
                import pandas as pd
                
                # Try multiple paths
                background_paths = [
                    Path("backend/data/ml_output/background_data.csv"),
                    Path("data/ml_output/background_data.csv"),
                    Path(__file__).parent.parent / "data" / "ml_output" / "background_data.csv"
                ]
                
                background_data = None
                for bg_path in background_paths:
                    if bg_path.exists():
                        try:
                            background_data = pd.read_csv(bg_path, index_col=0)
                            logger.info(f"✅ Loaded background data from {bg_path}: {len(background_data)} samples")
                            break
                        except Exception as e:
                            logger.warning(f"Failed to load background data from {bg_path}: {e}")
                            continue
                
                if background_data is not None:
                    self.shap_explainer = create_shap_explainer(
                        predictor.model_trainer,
                        predictor.feature_engineer,
                        background_data
                    )
                    logger.info("✅ SHAP explainer initialized successfully")
                else:
                    logger.warning("⚠️ Background data not found for SHAP initialization")
                    logger.warning("   SHAP explanations will not be available. Run training pipeline to generate background_data.csv")
        except Exception as e:
            logger.error(f"❌ Failed to initialize SHAP explainer: {e}", exc_info=True)
    
    def analyze(self, 
                audio_path: Optional[str] = None,
                transcript: Optional[str] = None,
                task_type: Optional[str] = None,
                metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main analysis function: Perform complete MCI screening
        
        Args:
            audio_path: Path to audio file (WAV, 16kHz recommended)
            transcript: Text transcript of the speech
            task_type: Type of cognitive task ('verbal_fluency', 'picture_description', 
                       'spontaneous_speech', 'qa')
            metadata: Optional clinical metadata (age, gender, mmse, etc.)
        
        Returns:
            Dict: Complete analysis result
        """
        start_time = time.time()
        errors = []
        acoustic_features = {}
        linguistic_features = {}
        fused_features = {}
        mci_prediction = None
        feature_summary = {}
        
        # Merge metadata into feature set if provided
        metadata = metadata or {}
        clinical_features = {k: v for k, v in metadata.items() if v is not None}
        
        # Default values
        # ⚠️ MMSE score phải được lấy từ chatbot test, không estimate
        mmse_score_from_chatbot = self._get_mmse_from_features({}, metadata or {})
        mmse_score = mmse_score_from_chatbot  # MMSE từ chatbot test
        severity = "Không xác định"
        confidence = 0.0
        risk_factors = []
        recommendations = []
        
        # Step 1: Acoustic Analysis (if audio provided)
        if audio_path and self.acoustic_analyzer:
            logger.info(f"🎤 Analyzing audio: {audio_path}")
            try:
                acoustic_features = self.acoustic_analyzer.extract_all_features(
                    audio_path, 
                    transcript=transcript
                )
                logger.info(f"✅ Extracted {len(acoustic_features)} acoustic features")
            except Exception as e:
                logger.error(f"Acoustic analysis failed: {e}")
                errors.append(f"Acoustic: {e}")
        elif audio_path:
            logger.warning("Audio provided but AcousticAnalyzer not available")
            errors.append("AcousticAnalyzer not available")
        
        # Step 2: Linguistic Analysis (if transcript provided)
        if transcript and self.linguistic_analyzer:
            logger.info(f"📝 Analyzing transcript ({len(transcript)} chars)")
            try:
                linguistic_features = self.linguistic_analyzer.extract_all_features(
                    transcript,
                    task_type=task_type
                )
                logger.info(f"✅ Extracted {len(linguistic_features)} linguistic features")
            except Exception as e:
                logger.error(f"Linguistic analysis failed: {e}")
                errors.append(f"Linguistic: {e}")
        elif transcript:
            logger.warning("Transcript provided but LinguisticAnalyzer not available")
            errors.append("LinguisticAnalyzer not available")
        
        # Step 3: Multimodal Fusion
        if self.fusion and (acoustic_features or linguistic_features):
            logger.info("🔗 Performing multimodal fusion")
            try:
                fused_features = self.fusion.fuse_features(
                    acoustic_features or {},
                    linguistic_features or {}
                )
                
                # Add clinical metadata to fused features
                fused_features.update(clinical_features)
                
                # ✅ Phase 12: Apply Integrated Feature Engineering (if enabled)
                if self.use_integrated_pipeline and self.integrated_feature_engineer:
                    try:
                        fused_features = self._apply_integrated_feature_engineering(
                            fused_features,
                            metadata or {}
                        )
                        logger.info("✅ Applied integrated feature engineering (v2.0)")
                    except Exception as e:
                        logger.warning(f"Integrated feature engineering failed, using standard: {e}")
                
                # Create feature summary
                feature_summary = self.fusion.create_feature_summary(
                    acoustic_features or {},
                    linguistic_features or {}
                )
                
                logger.info("✅ Fusion complete")
            except Exception as e:
                logger.error(f"Fusion failed: {e}")
                errors.append(f"Fusion: {e}")
        
        # Step 4: MCI Prediction
        if self.predictor and (acoustic_features or linguistic_features or clinical_features):
            logger.info("🧠 Predicting MCI status")
            try:
                # Combine all features for prediction
                all_features = {}
                all_features.update(acoustic_features)
                all_features.update(linguistic_features)
                all_features.update(clinical_features)
                
                # ✅ Phase 12: Use dual-output model if available and enabled
                # Note: Dual-output model cần được load từ trained model file
                # Hiện tại sẽ dùng standard predictor, dual-output sẽ được enable khi có trained model
                if self.use_integrated_pipeline and self.dual_output_model is not None:
                    logger.info("📊 Using dual-output model (v2.0)")
                    try:
                        prediction = self._predict_with_dual_output(all_features)
                        # Format dual-output prediction
                        mci_prediction = {
                            'mci_probability': prediction.mci_probability,
                            'mci_probability_calibrated': prediction.mci_probability_calibrated,
                            'mci_class': prediction.predicted_class,
                            'risk_binary': prediction.risk_binary,
                            'risk_binary_probability': prediction.risk_binary_probability,
                            'class_probabilities': prediction.class_probabilities,
                            'confidence': prediction.confidence,
                            'model_name': prediction.model_name,
                            'calibration_applied': prediction.calibration_applied
                        }
                        # Get MMSE từ features (từ chatbot test, KHÔNG estimate)
                        mmse_score = self._get_mmse_from_features(all_features, metadata or {})
                        severity = self._classify_severity_from_mmse(mmse_score)
                        risk_factors = self._extract_risk_factors_from_dual_output(prediction, all_features)
                        recommendations = self._generate_recommendations_from_dual_output(prediction, mmse_score)
                        confidence = prediction.confidence
                        mmse_estimate = mmse_score  # For backward compatibility in result
                    except Exception as e:
                        logger.warning(f"Dual-output prediction failed, falling back to standard: {e}")
                        prediction = self.predictor.predict(all_features)
                        # Get MMSE từ features (từ chatbot test)
                        mmse_score = self._get_mmse_from_features(all_features, metadata or {})
                        mci_prediction = {
                            'mci_probability': prediction.mci_probability,
                            'mci_class': prediction.mci_class,
                            'mmse_score': mmse_score,  # Từ chatbot test, KHÔNG phải estimate
                            'confidence': prediction.confidence,
                            'severity': prediction.severity
                        }
                        mmse_estimate = mmse_score  # For backward compatibility
                        severity = prediction.severity
                        confidence = prediction.confidence
                        risk_factors = prediction.risk_factors
                        recommendations = prediction.recommendations
                else:
                    # Use standard predictor
                    prediction = self.predictor.predict(all_features)
                    # Get MMSE từ features (từ chatbot test)
                    mmse_score = self._get_mmse_from_features(all_features, metadata or {})
                    
                    mci_prediction = {
                        'mci_probability': prediction.mci_probability,
                        'mci_class': prediction.mci_class,
                        'mmse_score': mmse_score,  # Từ chatbot test, KHÔNG phải estimate
                        'confidence': prediction.confidence,
                        'severity': self._classify_severity_from_mmse(mmse_score)
                    }
                    mmse_estimate = mmse_score  # Use MMSE từ chatbot
                    severity = mci_prediction['severity']
                    confidence = prediction.confidence
                    risk_factors = prediction.risk_factors if hasattr(prediction, 'risk_factors') else []
                    recommendations = prediction.recommendations if hasattr(prediction, 'recommendations') else []
                
                logger.info(f"✅ Prediction: {mci_prediction.get('mci_class', 'Unknown')}, MMSE (from chatbot) = {mmse_score:.1f}")
                
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                errors.append(f"Prediction: {e}")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"⏱️ Total processing time: {processing_time:.2f}s")
        
        # Determine success
        success = len(errors) == 0 and (acoustic_features or linguistic_features or clinical_features)
        
        from dataclasses import asdict
        result = AnalysisResult(
            success=success,
            acoustic_features=acoustic_features,
            linguistic_features=linguistic_features,
            fused_features=fused_features,
            mci_prediction=mci_prediction,
            mmse_score=mmse_score,  # MMSE từ chatbot test
            severity=severity,
            confidence=confidence,
            risk_factors=risk_factors,
            recommendations=recommendations,
            feature_summary=feature_summary,
            processing_time=processing_time,
            errors=errors
        )
        return asdict(result)
    
    def run_comprehensive_assessment(
        self,
        audio_path: str,
        transcript: str,
        session_id: str,
        age: int,
        education: int,
        mmse_raw_score: Optional[float] = None,
        task_type: Optional[str] = None,
        metadata: Optional[Dict] = None,
        mmse_responses: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive assessment with SHAP explanations.
        
        ✅ Phase 11: Production-grade assessment with:
        - MMSE scoring
        - ML classification
        - SHAP feature importance
        - Clinical recommendations
        
        Args:
            audio_path: Path to audio file
            transcript: Text transcript
            session_id: Unique session identifier
            age: Participant age
            education: Years of education
            mmse_raw_score: Raw MMSE score (if available)
            task_type: Type of cognitive task
            metadata: Additional metadata
            
        Returns:
            Dictionary with comprehensive assessment results
        """
        if not SHAP_AVAILABLE or not self.results_generator:
            logger.warning("⚠️ Comprehensive assessment not available - falling back to standard analyze()")
            return self.analyze(audio_path, transcript, task_type, metadata)
        
        logger.info(f"📊 Running comprehensive assessment for session {session_id}")
        
        # 1. Extract all features
        all_features = {}
        if audio_path and self.acoustic_analyzer:
            try:
                acoustic_features = self.acoustic_analyzer.extract_all_features(
                    audio_path, transcript=transcript
                )
                all_features.update(acoustic_features)
            except Exception as e:
                logger.error(f"Acoustic extraction failed: {e}")
        
        if transcript and self.linguistic_analyzer:
            try:
                linguistic_features = self.linguistic_analyzer.extract_all_features(
                    transcript, task_type=task_type
                )
                all_features.update(linguistic_features)
            except Exception as e:
                logger.error(f"Linguistic extraction failed: {e}")
        
        if metadata:
            all_features.update({k: v for k, v in metadata.items() if v is not None})
        
        # 2. ML Prediction
        ml_prediction_dict = {}
        if self.predictor:
            try:
                prediction = self.predictor.predict(all_features)
                ml_prediction_dict = {
                    'mci_class': prediction.mci_class,
                    'confidence': prediction.confidence,
                    'mci_probability': prediction.mci_probability,
                    'mmse_estimate': prediction.mmse_estimate,
                    'model_name': 'Ensemble (RF+XGBoost+SVM)',
                    'class_probabilities': {
                        'Normal': 1.0 - prediction.mci_probability if prediction.mci_class == 'Normal' else 0.0,
                        'MCI': prediction.mci_probability if 'MCI' in prediction.mci_class else 0.0,
                        'AD': prediction.mci_probability if 'AD' in prediction.mci_class else 0.0
                    }
                }
            except Exception as e:
                logger.error(f"ML prediction failed: {e}")
        
        # 3. SHAP Explanation
        shap_contributions = []
        if self.shap_explainer:
            try:
                target_class = ml_prediction_dict.get('mci_class', 'MCI')
                shap_contributions = self.shap_explainer.explain_prediction(
                    all_features,
                    target_class=target_class,
                    top_k=10
                )
                logger.info(f"✅ Generated {len(shap_contributions)} SHAP contributions")
            except Exception as e:
                logger.error(f"SHAP explanation failed: {e}", exc_info=True)
        
        # 4. Get MMSE score từ chatbot test (KHÔNG estimate)
        # MMSE score phải được pass vào từ chatbot test
        if mmse_raw_score is None:
            # Try to get from metadata or features
            mmse_raw_score = self._get_mmse_from_features(all_features, metadata or {})
            logger.info(f"📊 Using MMSE score from chatbot test: {mmse_raw_score:.1f}")
        
        # 5. Generate Comprehensive Results
        comprehensive_results = self.results_generator.generate(
            session_id=session_id,
            mmse_raw_score=mmse_raw_score,
            ml_prediction=ml_prediction_dict,
            shap_contributions=shap_contributions,
            features=all_features,
            participant_age=age,
            participant_education=education,
            mmse_responses=mmse_responses,
            audio_path=audio_path,
            transcript=transcript
        )
        
        # 6. Format for output (pass all_features for feature counting)
        return self._format_comprehensive_output(comprehensive_results, all_features)
    
    def _format_comprehensive_output(self, results, all_features: Dict[str, Any] = None) -> Dict[str, Any]:
        """Format ComprehensiveResult for API/PDF output"""
        
        return {
            'session_info': {
                'session_id': results.session_id,
                'date': results.assessment_date.isoformat(),
                'age': results.participant_age,
                'education': results.participant_education
            },
            'mmse_assessment': {
                'raw_score': results.mmse_score.raw_score,
                'standardized_score': results.mmse_score.standardized_score,
                'adjusted_score': results.mmse_score.adjusted_score,
                'interpretation': results.mmse_score.interpretation
            },
            'ml_classification': {
                'predicted_class': results.ml_classification.predicted_class,
                'confidence': results.ml_classification.confidence,
                'probabilities': results.ml_classification.class_probabilities,
                'model_name': results.ml_classification.model_name,
                'model_accuracy': results.ml_classification.model_accuracy,
                'note': '⚠️ Model chỉ phân loại (Normal/MCI/AD), KHÔNG dự đoán MMSE trực tiếp'
            },
            'clinical_estimate': {
                'estimated_mmse': results.clinical_estimate.estimated_mmse,
                'confidence_interval': results.clinical_estimate.confidence_interval,
                'severity': results.clinical_estimate.severity,
                'method': results.clinical_estimate.estimation_method,
                'note': '⚠️ Điểm MMSE ước tính từ ánh xạ lâm sàng, KHÔNG phải dự đoán trực tiếp từ mô hình'
            },
            'risk_assessment': {
                'overall_risk': results.risk_assessment.overall_risk_level,
                'risk_factors': results.risk_assessment.risk_factors,
                'protective_factors': results.risk_assessment.protective_factors
            },
            'feature_analysis': {
                'total_features': results.feature_analysis.total_features_analyzed,
                'acoustic_features': results.feature_analysis.acoustic_feature_count,
                'linguistic_features': results.feature_analysis.linguistic_feature_count,
                'top_features': results.feature_analysis.top_contributing_features,
                'abnormal_features': results.feature_analysis.abnormal_features
            },
            'features': all_features,  # ✅ Include actual features dict for counting
            'shap_explanation': {
                'available': len(results.feature_analysis.top_contributing_features) > 0,
                'top_contributors': results.feature_analysis.top_contributing_features,
                'citation': 'Lundberg & Lee (2017) - A Unified Approach to Interpreting Model Predictions'
            },
            'recommendations': results.recommendations,
            'follow_up': results.follow_up_timeline,
            'model_performance': results.model_performance
        }
    
    def analyze_audio_only(self, audio_path: str) -> Dict[str, Any]:
        """
        Analyze audio file only
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            dict: Acoustic features
        """
        if not self.acoustic_analyzer:
            return {'error': 'AcousticAnalyzer not available'}
        
        try:
            return self.acoustic_analyzer.extract_all_features(audio_path)
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_transcript_only(self, transcript: str, 
                                 task_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze transcript only
        
        Args:
            transcript: Text transcript
            task_type: Optional task type
        
        Returns:
            dict: Linguistic features
        """
        if not self.linguistic_analyzer:
            return {'error': 'LinguisticAnalyzer not available'}
        
        try:
            return self.linguistic_analyzer.extract_all_features(transcript, task_type)
        except Exception as e:
            return {'error': str(e)}
    
    def get_prediction_only(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Get MCI prediction from pre-extracted features
        
        Args:
            features: Pre-extracted features
        
        Returns:
            dict: Prediction result
        """
        if not self.predictor:
            return {'error': 'MCIPredictor not available'}
        
        try:
            prediction = self.predictor.predict(features)
            return asdict(prediction) if hasattr(prediction, '__dict__') else {
                'mci_probability': prediction.mci_probability,
                'mci_class': prediction.mci_class,
                'mmse_score': self._get_mmse_from_features(features, {}),  # Từ chatbot test
                'confidence': prediction.confidence,
                'severity': prediction.severity,
                'risk_factors': prediction.risk_factors,
                'recommendations': prediction.recommendations
            }
        except Exception as e:
            return {'error': str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get service status
        
        Returns:
            dict: Status of each component
        """
        return {
            'acoustic_analyzer': self.acoustic_analyzer is not None,
            'linguistic_analyzer': self.linguistic_analyzer is not None,
            'multimodal_fusion': self.fusion is not None,
            'mci_predictor': self.predictor is not None,
            'initialization_errors': self.errors,
            'is_ready': (self.acoustic_analyzer is not None or 
                        self.linguistic_analyzer is not None)
        }


# Singleton instance for easy access
_service_instance: Optional[MCIScreeningService] = None


def get_mci_service() -> MCIScreeningService:
    """
    Get or create singleton MCIScreeningService instance
    
    Returns:
        MCIScreeningService: The service instance
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = MCIScreeningService()
    return _service_instance


def analyze_for_mci(audio_path: Optional[str] = None,
                    transcript: Optional[str] = None,
                    task_type: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function for MCI analysis
    
    Args:
        audio_path: Path to audio file
        transcript: Text transcript
        task_type: Type of cognitive task
        metadata: Additional metadata (age, gender, mmse, etc.)
    
    Returns:
        dict: Analysis result with convenient access to key metrics
    """
    service = get_mci_service()
    result = service.analyze(audio_path, transcript, task_type, metadata)
    
    # Extract MCI probability from prediction
    mci_prob = 0.0
    mci_class = "Unknown"
    if result.mci_prediction:
        mci_prob = result.mci_prediction.get('mci_probability', 0.0)
        mci_class = result.mci_prediction.get('mci_class', 'Unknown')
    
    # Convert to dict with convenient access fields
    return {
        'success': result.success,
        # Convenient access to key metrics
        'mci_probability': mci_prob,
        'mci_class': mci_class,
        'mmse_score': result.mmse_score,  # MMSE từ chatbot test
        'severity': result.severity,
        'confidence': result.confidence,
        # Feature counts
        'acoustic_feature_count': len(result.acoustic_features) if result.acoustic_features else 0,
        'linguistic_feature_count': len(result.linguistic_features) if result.linguistic_features else 0,
        # Full data
        'acoustic_features': result.acoustic_features,
        'linguistic_features': result.linguistic_features,
        'fused_features': result.fused_features,
        'mci_prediction': result.mci_prediction,
        'risk_factors': result.risk_factors,
        'recommendations': result.recommendations,
        'feature_summary': result.feature_summary,
        'processing_time': result.processing_time,
        'errors': result.errors
    }

