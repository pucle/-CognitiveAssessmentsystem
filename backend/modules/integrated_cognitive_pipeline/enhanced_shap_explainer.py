# -*- coding: utf-8 -*-
"""
Enhanced SHAP Explainer với Scientific Interpretations
======================================================

Dựa trên literature review:
- Lundberg & Lee (2017): SHAP values provide theoretically grounded feature importance
- Recent studies (2020-2024): SHAP applications trong cognitive assessment
- Clinical interpretations dựa trên research findings

Features:
- Scientific interpretations cho mỗi feature
- Normal ranges từ literature
- Clinical significance levels
- Feature grouping và domain analysis

Author: Cognitive Assessment System
Version: 2.0
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available - install with: pip install shap")


@dataclass
class ScientificFeatureContribution:
    """Enhanced feature contribution với scientific interpretations."""
    feature_name: str
    feature_name_display: str
    shap_value: float
    feature_value: float
    baseline_value: float
    
    # Impact analysis
    impact_direction: str  # 'increases_risk' or 'decreases_risk'
    impact_magnitude: str  # 'strong', 'moderate', 'weak'
    
    # Scientific interpretation
    clinical_interpretation: str
    scientific_evidence: str  # Citation/reference
    
    # Domain classification
    feature_domain: str  # 'acoustic', 'linguistic', 'demographic', 'clinical', 'lifestyle'
    
    # Clinical significance
    clinical_significance: str  # 'high', 'moderate', 'low'
    
    # Optional fields with defaults
    normal_range: Optional[Tuple[float, float]] = None
    abnormal_threshold: Optional[float] = None


class EnhancedShapExplainer:
    """
    Enhanced SHAP explainer với scientific interpretations dựa trên literature review.
    """
    
    # Scientific feature interpretations từ literature
    FEATURE_SCIENTIFIC_INTERPRETATIONS = {
        # ============================================================
        # ACOUSTIC FEATURES
        # ============================================================
        'acoustic_f0_mean': {
            'display_name': 'Tần số cơ bản trung bình (F0)',
            'normal_range': (100, 250),  # Hz for adults
            'domain': 'acoustic',
            'interpretations': {
                'high': 'F0 cao có thể chỉ ra căng thẳng hoặc nỗ lực trong giao tiếp. Nghiên cứu cho thấy F0 variation giảm ở người MCI (Battista et al., 2020).',
                'low': 'F0 thấp có thể liên quan đến suy giảm kiểm soát vận động. F0 variation thấp là biomarker tiềm năng cho MCI.',
                'evidence': 'Battista P et al. (2020): Acoustic features (prosody, pause patterns) là biomarkers mạnh cho MCI prediction.'
            }
        },
        'acoustic_f0_cv': {
            'display_name': 'Biến thiên tần số cơ bản (F0 CV)',
            'normal_range': (0.10, 0.25),
            'domain': 'acoustic',
            'interpretations': {
                'high': 'F0 variation cao cho thấy prosody tốt và kiểm soát vận động tốt.',
                'low': 'F0 variation thấp (<0.15) là dấu hiệu của prosody phẳng, liên quan đến MCI. Đây là biomarker mạnh (Battista et al., 2020).',
                'evidence': 'Battista P et al. (2020): F0 variation giảm ở người MCI do suy giảm kiểm soát vận động.'
            }
        },
        'acoustic_pause_rate': {
            'display_name': 'Tỷ lệ ngừng nghỉ',
            'normal_range': (0.10, 0.30),
            'domain': 'acoustic',
            'interpretations': {
                'high': 'Tần suất ngừng nghỉ cao (>0.4) có thể chỉ ra khó khăn trong việc tìm từ hoặc lập kế hoạch câu. Đây là biomarker quan trọng cho MCI (Battista et al., 2020).',
                'low': 'Tần suất ngừng nghỉ bình thường cho thấy xử lý ngôn ngữ trôi chảy.',
                'evidence': 'Battista P et al. (2020): Pause patterns là acoustic biomarkers mạnh cho MCI prediction.'
            }
        },
        'acoustic_rate_words_per_minute': {
            'display_name': 'Tốc độ nói (từ/phút)',
            'normal_range': (120, 200),
            'domain': 'acoustic',
            'interpretations': {
                'high': 'Tốc độ nói bình thường cho thấy xử lý ngôn ngữ hiệu quả.',
                'low': 'Tốc độ nói chậm (<80 từ/phút) có thể chỉ ra suy giảm xử lý ngôn ngữ hoặc khó khăn trong lexical access.',
                'evidence': 'Battista P et al. (2020): Speaking rate giảm ở người MCI do suy giảm cognitive processing speed.'
            }
        },
        'acoustic_tone_flattening_score': {
            'display_name': 'Mức độ phẳng thanh điệu (Vietnamese-specific)',
            'normal_range': (0.0, 0.3),
            'domain': 'acoustic',
            'interpretations': {
                'high': 'Phẳng thanh điệu cao (>0.5) là biomarker đặc trưng cho MCI ở người Việt. Đây là phát hiện mới trong nghiên cứu tiếng Việt.',
                'low': 'Duy trì thanh điệu tốt cho thấy kiểm soát vận động tốt và không có dấu hiệu suy giảm.',
                'evidence': 'Vietnamese-specific biomarker: Tone flattening là đặc trưng quan trọng cho MCI prediction trong tiếng Việt.'
            }
        },
        'acoustic_vq_jitter_local': {
            'display_name': 'Độ bất ổn định giọng nói (Jitter)',
            'normal_range': (0.005, 0.020),
            'domain': 'acoustic',
            'interpretations': {
                'high': 'Jitter cao (>0.025) chỉ ra bất ổn định trong voice quality, có thể liên quan đến suy giảm motor control.',
                'low': 'Jitter bình thường cho thấy voice quality ổn định.',
                'evidence': 'Battista P et al. (2020): Voice quality features (jitter, shimmer) là acoustic biomarkers cho MCI.'
            }
        },
        
        # ============================================================
        # LINGUISTIC FEATURES
        # ============================================================
        'linguistic_sem_idea_density': {
            'display_name': 'Mật độ ý tưởng (Idea Density)',
            'normal_range': (4.0, 6.5),
            'domain': 'linguistic',
            'interpretations': {
                'high': 'Mật độ ý tưởng cao cho thấy tư duy phức tạp và tổ chức tốt. Đây là dấu hiệu tích cực.',
                'low': 'Mật độ ý tưởng thấp (<3.5) là biomarker MẠNH NHẤT cho suy giảm nhận thức. Nghiên cứu Nun Study cho thấy idea density thấp ở tuổi trẻ dự đoán dementia sau này (Snowdon et al., 1996).',
                'evidence': 'Nun Study (Snowdon et al., 1996): Idea density là predictor mạnh nhất cho dementia. Battista et al. (2020): Idea density có importance score cao nhất trong linguistic features.'
            }
        },
        'linguistic_lex_ttr': {
            'display_name': 'Đa dạng từ vựng (Type-Token Ratio, TTR)',
            'normal_range': (0.45, 0.75),
            'domain': 'linguistic',
            'interpretations': {
                'high': 'Đa dạng từ vựng tốt là dấu hiệu tích cực của chức năng ngôn ngữ và lexical access.',
                'low': 'Đa dạng từ vựng hạn chế (<0.35) có thể chỉ ra suy giảm lexical access, một dấu hiệu của MCI.',
                'evidence': 'Battista P et al. (2020): Lexical diversity (TTR) là linguistic biomarker quan trọng cho MCI prediction.'
            }
        },
        'linguistic_lex_pronoun_ratio': {
            'display_name': 'Tỷ lệ đại từ',
            'normal_range': (0.10, 0.20),
            'domain': 'linguistic',
            'interpretations': {
                'high': 'Sử dụng đại từ cao (>0.25) có thể chỉ ra khó khăn trong việc tìm từ cụ thể, một dấu hiệu của lexical access deficit.',
                'low': 'Tỷ lệ đại từ bình thường cho thấy lexical access tốt.',
                'evidence': 'Battista P et al. (2020): Pronoun ratio tăng ở người MCI do lexical access difficulties.'
            }
        },
        'linguistic_syn_mlu_words': {
            'display_name': 'Độ dài câu trung bình (MLU)',
            'normal_range': (8, 15),
            'domain': 'linguistic',
            'interpretations': {
                'high': 'Độ dài câu tốt cho thấy xử lý cú pháp và working memory tốt.',
                'low': 'Câu rất ngắn (<6 từ) có thể chỉ ra suy giảm xử lý ngôn ngữ hoặc working memory limitations.',
                'evidence': 'Battista P et al. (2020): Mean length of utterance (MLU) giảm ở người MCI do working memory và syntactic processing deficits.'
            }
        },
        'linguistic_sem_semantic_coherence': {
            'display_name': 'Mạch lạc ngữ nghĩa',
            'normal_range': (0.6, 1.0),
            'domain': 'linguistic',
            'interpretations': {
                'high': 'Mạch lạc ngữ nghĩa tốt cho thấy tổ chức tư duy và discourse planning tốt.',
                'low': 'Mạch lạc ngữ nghĩa thấp có thể chỉ ra suy giảm trong discourse organization, một dấu hiệu của executive function deficit.',
                'evidence': 'Battista P et al. (2020): Semantic coherence là linguistic biomarker cho executive function và MCI.'
            }
        },
        
        # ============================================================
        # DEMOGRAPHIC FEATURES
        # ============================================================
        'age': {
            'display_name': 'Tuổi',
            'normal_range': (50, 80),
            'domain': 'demographic',
            'interpretations': {
                'high': 'Tuổi cao (>75) là yếu tố nguy cơ chính cho suy giảm nhận thức. Risk tăng đáng kể sau 75 tuổi (Barnes et al., 2009).',
                'low': 'Tuổi trẻ hơn là yếu tố bảo vệ.',
                'evidence': 'Barnes DE et al. (2009): Age là top predictor trong late-life dementia risk index. Risk tăng gấp đôi mỗi 5 năm sau 65 tuổi.'
            }
        },
        'education_years': {
            'display_name': 'Số năm học',
            'normal_range': (6, 16),
            'domain': 'demographic',
            'interpretations': {
                'high': 'Education cao (>12 năm) là yếu tố bảo vệ (cognitive reserve). Giảm nguy cơ dementia 30-40% (Anstey et al., 2013).',
                'low': 'Education thấp (<6 năm) là yếu tố nguy cơ. MMSE scores cần được adjust theo education (Creavin et al., 2016).',
                'evidence': 'Anstey KJ et al. (2013): Education là protective factor. Creavin ST et al. (2016): MMSE có education bias, cần normalize.'
            }
        },
        'gender': {
            'display_name': 'Giới tính',
            'normal_range': None,
            'domain': 'demographic',
            'interpretations': {
                'high': 'Nữ giới có risk cao hơn một chút (do sống lâu hơn), nhưng không phải yếu tố nguy cơ chính.',
                'low': 'Nam giới có risk tương đương sau khi adjust cho age.',
                'evidence': 'Barnes DE et al. (2009): Gender không phải là predictor mạnh trong dementia risk index.'
            }
        },
        
        # ============================================================
        # CLINICAL FEATURES - MMSE (INPUT FEATURE từ chatbot test)
        # ============================================================
        'mmse_adjusted': {
            'display_name': 'Điểm MMSE đã điều chỉnh (theo age và education)',
            'normal_range': (24, 30),
            'domain': 'clinical',
            'interpretations': {
                'high': 'MMSE cao (≥24) sau khi adjust cho age/education cho thấy nhận thức bình thường. Đây là input feature từ chatbot test, không phải estimate từ model.',
                'low': 'MMSE thấp (<24) sau khi adjust là dấu hiệu suy giảm nhận thức. Tuy nhiên, MMSE đơn lẻ chỉ có sensitivity 79-85%, cần kết hợp với acoustic và linguistic features để tăng accuracy lên 92-95% (Petersen et al., 2018).',
                'evidence': 'Petersen RC et al. (2018): MMSE chỉ là công cụ sàng lọc, không đủ để chẩn đoán MCI. Khi kết hợp với multi-modal features, sensitivity tăng từ 79-85% lên 92-95%. Creavin ST et al. (2016): MMSE cần normalize theo age và education.'
            }
        },
        'mmse_raw': {
            'display_name': 'Điểm MMSE thô (từ chatbot test)',
            'normal_range': (24, 30),
            'domain': 'clinical',
            'interpretations': {
                'high': 'MMSE raw score cao từ chatbot test. Đây là input feature quan trọng, nhưng cần adjust theo age và education để chính xác. Model sử dụng MMSE cùng với acoustic và linguistic features để dự đoán risk.',
                'low': 'MMSE raw score thấp từ chatbot test có thể do age/education bias hoặc suy giảm nhận thức thực sự. Model sẽ kết hợp với các features khác để đánh giá chính xác hơn.',
                'evidence': '⚠️ IMPORTANT: MMSE score là INPUT FEATURE từ chatbot test, KHÔNG phải estimate từ model. Creavin ST et al. (2016): MMSE có age và education bias, cần adjustment. Battista P et al. (2020): Multi-modal features (MMSE + acoustic + linguistic) tăng accuracy 15-25% so với MMSE đơn lẻ.'
            }
        },
        'mmse': {
            'display_name': 'Điểm MMSE (từ chatbot test)',
            'normal_range': (24, 30),
            'domain': 'clinical',
            'interpretations': {
                'high': 'MMSE score cao từ chatbot test. Model sử dụng MMSE như một trong nhiều features để dự đoán risk. MMSE đơn lẻ có sensitivity 79-85%, nhưng khi kết hợp với acoustic và linguistic features, accuracy tăng lên 92-95% (Petersen et al., 2018).',
                'low': 'MMSE score thấp từ chatbot test. Đây là input feature quan trọng, nhưng model sẽ kết hợp với acoustic features (prosody, pause patterns) và linguistic features (idea density, lexical diversity) để đánh giá toàn diện hơn.',
                'evidence': '⚠️ MMSE là INPUT FEATURE từ chatbot test, KHÔNG phải output của model. Petersen RC et al. (2018): MMSE đơn lẻ chỉ có sensitivity 79-85%, nhưng khi kết hợp với multi-modal features, sensitivity tăng lên 92-95%. Battista P et al. (2020): Multi-modal assessment tăng accuracy 15-25% so với single-modal.'
            }
        },
    }
    
    def __init__(self, model_trainer, feature_engineer):
        """
        Initialize enhanced SHAP explainer.
        
        ⚠️ IMPORTANT: Model trainer có thể là:
        - DualOutputModelTrainer: Output binary risk + MCI probability
        - Standard model trainer: Output class probabilities
        
        MMSE score là INPUT FEATURE từ chatbot test, không phải output.
        
        Args:
            model_trainer: Trained model trainer (dual-output or standard)
            feature_engineer: Fitted feature engineer (IntegratedFeatureEngineer)
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not available. Install with: pip install shap")
        
        self.model_trainer = model_trainer
        self.feature_engineer = feature_engineer
        self.explainer = None
        self.background_data = None
        self.feature_names = None
        self.is_dual_output = hasattr(model_trainer, 'binary_classifier') and hasattr(model_trainer, 'probability_estimator')
        
        logger.info(f"EnhancedShapExplainer initialized (dual-output: {self.is_dual_output})")
    
    def initialize_explainer(self, X_background: pd.DataFrame, max_background: int = 100):
        """Initialize SHAP explainer với background data."""
        if len(X_background) > max_background:
            X_background = X_background.sample(n=max_background, random_state=42)
        
        self.background_data = X_background
        self.feature_names = list(X_background.columns)
        
        # Get best model from trainer
        # ⚠️ For dual-output model, we explain the binary classifier (risk Yes/No)
        # The MCI probability is derived from class probabilities
        if hasattr(self.model_trainer, 'binary_classifier'):
            # Dual-output model: use binary classifier for SHAP (explains risk Yes/No)
            best_model = self.model_trainer.binary_classifier
            logger.info("✅ Using binary classifier from dual-output model for SHAP explanation")
        elif hasattr(self.model_trainer, 'probability_estimator'):
            # Alternative: explain probability estimator (MCI probability)
            best_model = self.model_trainer.probability_estimator
            logger.info("✅ Using probability estimator from dual-output model for SHAP explanation")
        elif hasattr(self.model_trainer, 'best_model'):
            best_model = self.model_trainer.best_model
        else:
            raise ValueError("No model found in model_trainer")
        
        # Initialize TreeExplainer for tree-based models
        try:
            if hasattr(best_model, 'estimators_') or hasattr(best_model, 'get_booster'):
                self.explainer = shap.TreeExplainer(
                    best_model,
                    data=X_background,
                    feature_perturbation='interventional'
                )
                logger.info("✅ Initialized TreeExplainer")
            else:
                # Use KernelExplainer for non-tree models
                background_sample = X_background.sample(min(50, len(X_background)), random_state=42)
                self.explainer = shap.KernelExplainer(
                    best_model.predict_proba,
                    background_sample
                )
                logger.info("✅ Initialized KernelExplainer")
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            raise
    
    def explain_prediction(
        self,
        features: Dict[str, float],
        target_class: str = 'MCI',
        top_k: int = 15,
        explain_output: str = 'risk_binary'
    ) -> List[ScientificFeatureContribution]:
        """
        Calculate SHAP values với scientific interpretations.
        
        ⚠️ IMPORTANT:
        - MMSE score là INPUT FEATURE từ chatbot test, không phải output
        - For dual-output model: explain_output có thể là 'risk_binary' hoặc 'mci_probability'
        - SHAP sẽ giải thích đóng góp của MMSE vs other features (acoustic, linguistic, demographic)
        
        Args:
            features: Dictionary of extracted features (bao gồm MMSE từ chatbot test)
            target_class: Target class to explain ('Normal', 'MCI', 'AD')
            top_k: Number of top features to return
            explain_output: For dual-output model, 'risk_binary' (default) or 'mci_probability'
        
        Returns:
            List of ScientificFeatureContribution objects với scientific interpretations
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer() first.")
        
        # Transform features
        numeric_features = {}
        for k, v in features.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                numeric_features[k] = float(v)
            elif isinstance(v, bool):
                numeric_features[k] = float(v)
        
        X = self.feature_engineer.transform(pd.DataFrame([numeric_features]))
        
        # Align columns
        missing_cols = set(self.feature_names) - set(X.columns)
        if missing_cols:
            for col in missing_cols:
                X[col] = 0.0
        X = X[self.feature_names]
        
        # Calculate SHAP values
        try:
            shap_values = self.explainer.shap_values(X)
            
            # Handle dual-output vs standard model
            if self.is_dual_output:
                # Dual-output model: binary classifier outputs [No Risk, Risk]
                # SHAP values explain contribution to Risk (class 1)
                if isinstance(shap_values, list):
                    # Binary classification: shap_values = [class0_shap, class1_shap]
                    shap_values_class = shap_values[1][0]  # Risk class (class 1)
                elif len(shap_values.shape) == 2:
                    # Binary: shape (n_samples, n_features)
                    shap_values_class = shap_values[0]
                else:
                    shap_values_class = shap_values[0]
                logger.info("✅ Calculated SHAP values for dual-output binary classifier (Risk Yes/No)")
            else:
                # Standard multi-class model
                if isinstance(shap_values, list):
                    # Get class index
                    if hasattr(self.model_trainer, 'class_label_encoder'):
                        class_idx = self.model_trainer.class_label_encoder.transform([target_class])[0]
                    elif hasattr(self.model_trainer, 'label_encoder'):
                        class_idx = self.model_trainer.label_encoder.transform([target_class])[0]
                    else:
                        class_idx = 1  # Default to MCI class
                    shap_values_class = shap_values[class_idx][0]
                elif len(shap_values.shape) == 3:
                    class_idx = self.model_trainer.class_label_encoder.transform([target_class])[0]
                    shap_values_class = shap_values[0, :, class_idx]
                else:
                    shap_values_class = shap_values[0]
            
            # Get baseline (expected value)
            if hasattr(self.explainer, 'expected_value'):
                if isinstance(self.explainer.expected_value, (list, np.ndarray)):
                    if self.is_dual_output:
                        baseline = self.explainer.expected_value[1]  # Risk class baseline
                    else:
                        class_idx = self.model_trainer.class_label_encoder.transform([target_class])[0]
                        baseline = self.explainer.expected_value[class_idx]
                else:
                    baseline = self.explainer.expected_value
            else:
                baseline = 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate SHAP values: {e}")
            return []
        
        # Create scientific contributions
        contributions = []
        
        for i, feature_name in enumerate(self.feature_names):
            shap_val_raw = shap_values_class[i]
            if isinstance(shap_val_raw, np.ndarray):
                shap_val = float(shap_val_raw.item() if shap_val_raw.size == 1 else shap_val_raw[0])
            else:
                shap_val = float(shap_val_raw)
            feat_val = float(X.iloc[0, i])
            
            # Skip near-zero contributions
            if abs(shap_val) < 1e-6:
                continue
            
            # Get scientific interpretation
            interpretation_data = self._get_scientific_interpretation(feature_name, feat_val, shap_val)
            
            contribution = ScientificFeatureContribution(
                feature_name=feature_name,
                feature_name_display=interpretation_data['display_name'],
                shap_value=shap_val,
                feature_value=feat_val,
                baseline_value=baseline,
                impact_direction='increases_risk' if shap_val > 0 else 'decreases_risk',
                impact_magnitude=self._get_impact_magnitude(abs(shap_val)),
                clinical_interpretation=interpretation_data['interpretation'],
                scientific_evidence=interpretation_data['evidence'],
                normal_range=interpretation_data['normal_range'],
                abnormal_threshold=interpretation_data['abnormal_threshold'],
                feature_domain=interpretation_data['domain'],
                clinical_significance=self._get_clinical_significance(abs(shap_val), interpretation_data['domain'])
            )
            
            contributions.append(contribution)
        
        # Sort by absolute SHAP value
        contributions.sort(key=lambda x: abs(x.shap_value), reverse=True)
        
        logger.info(f"✅ Generated {len(contributions)} scientific feature contributions")
        
        return contributions[:top_k]
    
    def _get_scientific_interpretation(
        self,
        feature_name: str,
        feature_value: float,
        shap_value: float
    ) -> Dict[str, Any]:
        """Get scientific interpretation cho feature."""
        
        # Check if we have predefined interpretation
        if feature_name in self.FEATURE_SCIENTIFIC_INTERPRETATIONS:
            data = self.FEATURE_SCIENTIFIC_INTERPRETATIONS[feature_name]
            normal_range = data.get('normal_range')
            
            # Determine if value is high or low
            if normal_range:
                if feature_value > normal_range[1]:
                    interpretation = data['interpretations'].get('high', '')
                elif feature_value < normal_range[0]:
                    interpretation = data['interpretations'].get('low', '')
                else:
                    interpretation = 'Giá trị trong phạm vi bình thường.'
            else:
                # Use SHAP direction
                if shap_value > 0:
                    interpretation = data['interpretations'].get('high', '')
                else:
                    interpretation = data['interpretations'].get('low', '')
            
            return {
                'display_name': data.get('display_name', feature_name),
                'interpretation': interpretation,
                'evidence': data['interpretations'].get('evidence', ''),
                'normal_range': normal_range,
                'abnormal_threshold': normal_range[0] if normal_range else None,
                'domain': data.get('domain', 'unknown')
            }
        
        # Generic interpretation
        domain = 'unknown'
        if 'acoustic' in feature_name.lower() or 'f0' in feature_name.lower() or 'pause' in feature_name.lower() or 'jitter' in feature_name.lower() or 'shimmer' in feature_name.lower():
            domain = 'acoustic'
        elif 'linguistic' in feature_name.lower() or 'lex' in feature_name.lower() or 'sem' in feature_name.lower() or 'ttr' in feature_name.lower() or 'mlu' in feature_name.lower():
            domain = 'linguistic'
        elif feature_name.lower() in ['age', 'gender', 'education', 'education_years']:
            domain = 'demographic'
        elif 'mmse' in feature_name.lower():
            domain = 'clinical'
            # Special note for MMSE
            interpretation = f"⚠️ MMSE là INPUT FEATURE từ chatbot test, không phải estimate từ model. " + interpretation
        
        if shap_value > 0:
            interpretation = f"Đặc trưng này tăng nguy cơ suy giảm nhận thức."
        else:
            interpretation = f"Đặc trưng này giảm nguy cơ suy giảm nhận thức."
        
        return {
            'display_name': feature_name.replace('_', ' ').title(),
            'interpretation': interpretation,
            'evidence': 'General feature contribution.',
            'normal_range': None,
            'abnormal_threshold': None,
            'domain': domain
        }
    
    def _get_impact_magnitude(self, abs_shap: float) -> str:
        """Determine impact magnitude từ SHAP value."""
        if abs_shap > 0.15:
            return 'strong'
        elif abs_shap > 0.05:
            return 'moderate'
        else:
            return 'weak'
    
    def _get_clinical_significance(self, abs_shap: float, domain: str) -> str:
        """Determine clinical significance."""
        # High significance: strong impact + known biomarker domain
        if abs_shap > 0.15 and domain in ['acoustic', 'linguistic']:
            return 'high'
        elif abs_shap > 0.10:
            return 'moderate'
        else:
            return 'low'
    
    def generate_scientific_summary(
        self,
        contributions: List[ScientificFeatureContribution]
    ) -> Dict[str, Any]:
        """Generate scientific summary của SHAP contributions."""
        
        # Group by domain
        domain_groups = {}
        for contrib in contributions:
            domain = contrib.feature_domain
            if domain not in domain_groups:
                domain_groups[domain] = []
            domain_groups[domain].append(contrib)
        
        # Top contributors by domain
        top_by_domain = {}
        for domain, contribs in domain_groups.items():
            top_by_domain[domain] = {
                'count': len(contribs),
                'top_features': [
                    {
                        'name': c.feature_name_display,
                        'shap_value': c.shap_value,
                        'impact': c.impact_direction,
                        'magnitude': c.impact_magnitude
                    }
                    for c in contribs[:5]
                ]
            }
        
        # Overall summary
        total_positive = sum(1 for c in contributions if c.shap_value > 0)
        total_negative = sum(1 for c in contributions if c.shap_value < 0)
        
        return {
            'total_features': len(contributions),
            'risk_increasing_features': total_positive,
            'risk_decreasing_features': total_negative,
            'top_contributors': [
                {
                    'name': c.feature_name_display,
                    'shap_value': c.shap_value,
                    'domain': c.feature_domain,
                    'clinical_significance': c.clinical_significance,
                    'evidence': c.scientific_evidence
                }
                for c in contributions[:10]
            ],
            'by_domain': top_by_domain,
            'scientific_basis': 'Lundberg & Lee (2017): SHAP values provide theoretically grounded feature importance. Recent studies (2020-2024) show SHAP applications in cognitive assessment models.'
        }


# Convenience function
def create_enhanced_shap_explainer(
    model_trainer,
    feature_engineer,
    X_train: pd.DataFrame
) -> EnhancedShapExplainer:
    """Factory function to create enhanced SHAP explainer."""
    explainer = EnhancedShapExplainer(model_trainer, feature_engineer)
    explainer.initialize_explainer(X_train)
    return explainer

