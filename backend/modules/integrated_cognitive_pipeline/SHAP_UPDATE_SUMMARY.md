# SHAP EXPLAINER UPDATE SUMMARY

## ✅ CẬP NHẬT THEO THIẾT KẾ MỚI

### 🔄 Thay đổi chính

1. **MMSE là INPUT FEATURE, không phải OUTPUT**
   - ✅ Updated interpretations cho `mmse_raw`, `mmse_adjusted`, `mmse`
   - ✅ Thêm warnings: "MMSE là INPUT FEATURE từ chatbot test, không phải estimate từ model"
   - ✅ SHAP sẽ giải thích đóng góp của MMSE vs other features

2. **Hỗ trợ Dual-Output Model**
   - ✅ Detect dual-output model (`is_dual_output` flag)
   - ✅ Explain binary classifier (Risk Yes/No) hoặc probability estimator (MCI Probability)
   - ✅ Proper handling của binary classification SHAP values

3. **MMSE Contribution Analysis**
   - ✅ Phân tích đóng góp của MMSE vs other features (acoustic, linguistic, demographic)
   - ✅ Tính toán `mmse_relative_importance` trong scientific summary
   - ✅ So sánh impact của MMSE vs multi-modal features

---

## 📋 UPDATED FEATURES

### 1. MMSE Interpretations (Updated)

```python
'mmse_raw': {
    'display_name': 'Điểm MMSE thô (từ chatbot test)',
    'interpretations': {
        'high': 'MMSE raw score cao từ chatbot test. Đây là input feature quan trọng, 
                 nhưng cần adjust theo age và education để chính xác. 
                 Model sử dụng MMSE cùng với acoustic và linguistic features để dự đoán risk.',
        'low': 'MMSE raw score thấp từ chatbot test có thể do age/education bias 
                hoặc suy giảm nhận thức thực sự. 
                Model sẽ kết hợp với các features khác để đánh giá chính xác hơn.',
        'evidence': '⚠️ IMPORTANT: MMSE score là INPUT FEATURE từ chatbot test, 
                     KHÔNG phải estimate từ model. 
                     Battista P et al. (2020): Multi-modal features (MMSE + acoustic + linguistic) 
                     tăng accuracy 15-25% so với MMSE đơn lẻ.'
    }
}
```

### 2. Dual-Output Model Support

```python
def __init__(self, model_trainer, feature_engineer):
    # Detect dual-output model
    self.is_dual_output = hasattr(model_trainer, 'binary_classifier') and \
                          hasattr(model_trainer, 'probability_estimator')
    
def initialize_explainer(self, X_background: pd.DataFrame, max_background: int = 100):
    # For dual-output model, explain binary classifier (Risk Yes/No)
    if hasattr(self.model_trainer, 'binary_classifier'):
        best_model = self.model_trainer.binary_classifier
        logger.info("✅ Using binary classifier from dual-output model for SHAP explanation")
```

### 3. MMSE Contribution Analysis

```python
def generate_scientific_summary(self, contributions):
    # MMSE contribution analysis
    mmse_contributions = [c for c in contributions if 'mmse' in c.feature_name.lower()]
    mmse_total_impact = sum(abs(c.shap_value) for c in mmse_contributions)
    other_features_impact = sum(abs(c.shap_value) for c in contributions 
                                if 'mmse' not in c.feature_name.lower())
    
    return {
        'mmse_contribution_analysis': {
            'mmse_features_count': len(mmse_contributions),
            'mmse_total_impact': float(mmse_total_impact),
            'other_features_total_impact': float(other_features_impact),
            'mmse_relative_importance': float(mmse_total_impact / 
                                               (mmse_total_impact + other_features_impact)),
            'note': '⚠️ MMSE là INPUT FEATURE từ chatbot test. 
                     SHAP values cho thấy đóng góp của MMSE vs other features 
                     (acoustic, linguistic, demographic).'
        }
    }
```

---

## 🔍 USAGE

### With Dual-Output Model

```python
from backend.modules.integrated_cognitive_pipeline.enhanced_shap_explainer import (
    create_enhanced_shap_explainer
)

# Initialize với dual-output model
explainer = create_enhanced_shap_explainer(
    model_trainer=dual_output_model_trainer,  # DualOutputModelTrainer
    feature_engineer=integrated_feature_engineer,  # IntegratedFeatureEngineer
    X_train=X_train_processed
)

# Explain prediction
# MMSE score từ chatbot test được pass vào features
features = {
    'mmse_raw': 25.0,  # ✅ Từ chatbot test
    'age': 70,
    'education_years': 12,
    'acoustic_f0_mean': 180.5,
    'linguistic_lex_ttr': 0.65,
    # ... other features
}

contributions = explainer.explain_prediction(
    features=features,
    target_class='MCI',
    top_k=15,
    explain_output='risk_binary'  # For dual-output: 'risk_binary' or 'mci_probability'
)

# Generate scientific summary
summary = explainer.generate_scientific_summary(contributions)

# Summary includes:
# - MMSE contribution analysis
# - Domain importance (acoustic vs linguistic vs demographic vs clinical)
# - Top contributors by domain
```

---

## 📊 OUTPUT EXAMPLE

### Scientific Summary với MMSE Analysis

```python
{
    'total_features': 15,
    'risk_increasing_features': 8,
    'risk_decreasing_features': 7,
    'mmse_contribution_analysis': {
        'mmse_features_count': 2,  # mmse_raw, mmse_adjusted
        'mmse_total_impact': 0.25,
        'other_features_total_impact': 0.75,
        'mmse_relative_importance': 0.25,  # MMSE đóng góp 25% tổng impact
        'note': '⚠️ MMSE là INPUT FEATURE từ chatbot test...'
    },
    'domain_importance': {
        'acoustic': 0.30,
        'linguistic': 0.35,
        'demographic': 0.10,
        'clinical': 0.25  # MMSE features
    },
    'top_contributors': [
        {
            'name': 'Mật độ ý tưởng (Idea Density)',
            'shap_value': 0.18,
            'domain': 'linguistic',
            'clinical_significance': 'high',
            'evidence': 'Nun Study (Snowdon et al., 1996): Idea density là predictor mạnh nhất...'
        },
        {
            'name': 'Điểm MMSE đã điều chỉnh',
            'shap_value': 0.15,
            'domain': 'clinical',
            'clinical_significance': 'moderate',
            'evidence': '⚠️ MMSE là INPUT FEATURE từ chatbot test...'
        },
        # ... more contributors
    ]
}
```

---

## ⚠️ IMPORTANT NOTES

1. **MMSE Source**: MMSE score PHẢI được lấy từ chatbot test và pass vào như feature.

2. **Dual-Output Explanation**: 
   - Default: Explain binary classifier (Risk Yes/No)
   - Option: Explain probability estimator (MCI Probability)

3. **Feature Importance**: 
   - SHAP values cho thấy đóng góp tương đối của MMSE vs other features
   - Multi-modal features (acoustic + linguistic) thường có impact cao hơn MMSE đơn lẻ
   - Literature: Multi-modal tăng accuracy 15-25% so với MMSE đơn lẻ (Battista et al., 2020)

4. **Scientific Basis**: 
   - All interpretations dựa trên literature review
   - Citations included trong evidence fields
   - Normal ranges từ clinical studies

---

**Status:** ✅ **UPDATED**  
**Date:** 2025-01-07  
**Version:** 2.0

