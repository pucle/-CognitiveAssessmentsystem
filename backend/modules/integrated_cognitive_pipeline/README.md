# INTEGRATED COGNITIVE ASSESSMENT PIPELINE v2.0

## 📋 TỔNG QUAN

Pipeline đánh giá nhận thức tích hợp mới với:
- **MMSE là một feature** trong multi-modal input (không phải output riêng biệt)
- **Dual output**: Binary Classification (Risk Yes/No) + Probability Estimation (MCI Probability)
- **Scientific foundation**: Dựa trên 10+ nghiên cứu khoa học
- **Enhanced SHAP**: Với scientific interpretations từ literature review

---

## 🏗️ KIẾN TRÚC

```
INPUT FEATURES (Multi-modal)
├── MMSE-based scores (normalized theo age/education)
├── Demographic: age, gender, education
├── Medical history: vascular disease, diabetes, hypertension
├── Lifestyle: physical activity, diet, sleep (nếu có)
├── Acoustic features: F0, pause patterns, voice quality, tone flattening
└── Linguistic features: idea density, TTR, MLU, semantic coherence
    ↓
FEATURE ENGINEERING
├── MMSE normalization (age + education adjustment)
├── Missing data imputation (MICE/KNN)
├── Feature selection (correlation + RFE)
└── Scaling (StandardScaler/RobustScaler)
    ↓
DUAL-OUTPUT MODEL
├── Head 1: Binary Classifier (Risk Yes/No)
└── Head 2: Probability Estimator (MCI Probability 0-1)
    ↓
POST-PROCESSING
├── Calibration (Platt scaling/Isotonic regression)
├── Risk stratification (Low/Medium/High/Very High)
└── SHAP explanations với scientific interpretations
```

---

## 📚 CƠ SỞ KHOA HỌC

Xem file `LITERATURE_REVIEW.md` để biết chi tiết về:
- Multi-domain cognitive assessment (Petersen et al., 2018)
- Machine learning for MCI prediction (Battista et al., 2020)
- Risk stratification (Barnes et al., 2009)
- MMSE limitations (Creavin et al., 2016)
- SHAP explainability (Lundberg & Lee, 2017)

---

## 🚀 SỬ DỤNG

### 1. Feature Engineering

```python
from backend.modules.integrated_cognitive_pipeline.feature_engineer_v2 import IntegratedFeatureEngineer

# Initialize
engineer = IntegratedFeatureEngineer(
    mmse_normalization_method='adni',  # or 'simple'
    imputation_method='knn',  # or 'mice', 'median'
    feature_selection_method='rfe',  # or 'kbest', 'none'
    n_features_to_select=100,
    correlation_threshold=0.9,
    scaler_type='standard'  # or 'robust'
)

# Fit on training data
X_train_processed = engineer.fit_transform(
    X_train,
    y_train,
    do_mmse_normalization=True,
    do_feature_selection=True
)

# Transform new data
X_test_processed = engineer.transform(X_test, do_mmse_normalization=True)
```

### 2. Dual-Output Model Training

```python
from backend.modules.integrated_cognitive_pipeline.dual_output_model import DualOutputModelTrainer

# Initialize
trainer = DualOutputModelTrainer(
    config=None,  # Optional config dict
    random_state=42
)

# Train
training_results = trainer.train(
    X_train=X_train_processed,
    y_train=y_train,  # Labels: 'Normal', 'MCI', 'AD'
    X_val=X_val_processed,  # Optional
    y_val=y_val,  # Optional
    use_calibration=True
)

# Predict
prediction = trainer.predict(X_test_processed)
print(f"Risk Binary: {prediction.risk_binary}")
print(f"MCI Probability: {prediction.mci_probability:.3f}")
print(f"Predicted Class: {prediction.predicted_class}")
```

### 3. Enhanced SHAP Explanations

```python
from backend.modules.integrated_cognitive_pipeline.enhanced_shap_explainer import create_enhanced_shap_explainer

# Create explainer
shap_explainer = create_enhanced_shap_explainer(
    model_trainer=trainer,
    feature_engineer=engineer,
    X_train=X_train_processed
)

# Explain prediction
contributions = shap_explainer.explain_prediction(
    features=test_features_dict,
    target_class='MCI',
    top_k=15
)

# Print scientific interpretations
for contrib in contributions:
    print(f"\n{contrib.feature_name_display}:")
    print(f"  SHAP Value: {contrib.shap_value:.4f}")
    print(f"  Impact: {contrib.impact_direction} ({contrib.impact_magnitude})")
    print(f"  Clinical: {contrib.clinical_interpretation}")
    print(f"  Evidence: {contrib.scientific_evidence}")

# Generate scientific summary
summary = shap_explainer.generate_scientific_summary(contributions)
print(f"\nTop Contributors by Domain:")
for domain, data in summary['by_domain'].items():
    print(f"  {domain}: {data['count']} features")
```

---

## 📊 OUTPUT FORMAT

### Dual-Output Prediction

```python
@dataclass
class DualOutputPrediction:
    # Binary classification
    risk_binary: bool  # True = có nguy cơ
    risk_binary_probability: float  # Probability của risk
    
    # Probability estimation
    mci_probability: float  # MCI probability [0-1]
    mci_probability_calibrated: float  # Calibrated probability
    
    # Class prediction
    predicted_class: str  # 'Normal', 'MCI', 'AD'
    class_probabilities: Dict[str, float]
    
    # Confidence
    confidence: float
    
    # Metadata
    model_name: str
    calibration_applied: bool
```

### Scientific Feature Contribution

```python
@dataclass
class ScientificFeatureContribution:
    feature_name: str
    feature_name_display: str
    shap_value: float
    feature_value: float
    
    # Impact analysis
    impact_direction: str  # 'increases_risk' or 'decreases_risk'
    impact_magnitude: str  # 'strong', 'moderate', 'weak'
    
    # Scientific interpretation
    clinical_interpretation: str
    scientific_evidence: str  # Citation
    normal_range: Optional[Tuple[float, float]]
    
    # Domain
    feature_domain: str  # 'acoustic', 'linguistic', 'demographic', 'clinical'
    clinical_significance: str  # 'high', 'moderate', 'low'
```

---

## 🔬 SCIENTIFIC INTERPRETATIONS

Enhanced SHAP explainer cung cấp interpretations dựa trên literature:

### Acoustic Features
- **F0 variation**: Giảm ở người MCI (Battista et al., 2020)
- **Pause rate**: Tăng ở người MCI do lexical access difficulties
- **Tone flattening**: Biomarker đặc trưng cho MCI ở người Việt

### Linguistic Features
- **Idea density**: Biomarker MẠNH NHẤT (Nun Study, Snowdon et al., 1996)
- **TTR**: Giảm ở người MCI do lexical access deficit
- **Pronoun ratio**: Tăng ở người MCI do word-finding difficulties

### Demographic Features
- **Age**: Top predictor (Barnes et al., 2009)
- **Education**: Protective factor, cần normalize MMSE (Creavin et al., 2016)

### Clinical Features
- **MMSE adjusted**: Normalize theo age và education (Petersen et al., 2018)

---

## 📈 VALIDATION METRICS

### Classification Metrics
- Accuracy, Sensitivity, Specificity, F1-score
- AUC-ROC
- Confusion matrix

### Probability Estimation Metrics
- Brier score (calibration)
- Expected Calibration Error (ECE)
- Calibration plot

### Clinical Utility
- Net Benefit Analysis (Decision Curve Analysis)
- Risk stratification: Low (<0.3), Moderate (0.3-0.6), High (0.6-0.8), Very High (≥0.8)

---

## ⚙️ CONFIGURATION

### Feature Engineering Config

```python
config = {
    'mmse_normalization_method': 'adni',  # 'adni' or 'simple'
    'imputation_method': 'knn',  # 'mice', 'knn', 'median'
    'feature_selection_method': 'rfe',  # 'rfe', 'kbest', 'none'
    'n_features_to_select': 100,
    'correlation_threshold': 0.9,
    'scaler_type': 'standard'  # 'standard' or 'robust'
}
```

### Model Training Config

```python
config = {
    'random_state': 42,
    'use_calibration': True,
    'cv_folds': 5
}
```

---

## 📝 NOTES

1. **MMSE Integration**: MMSE là một feature, không phải output chính. Phải normalize theo age và education.

2. **Dual Output**: 
   - Binary classification: Risk Yes/No (threshold 0.5)
   - Probability estimation: MCI probability [0-1] (calibrated)

3. **SHAP Explanations**: 
   - Scientific interpretations từ literature
   - Normal ranges để compare
   - Clinical significance levels

4. **Calibration**: Probability estimates phải calibrated để accurate, không chỉ ranking.

---

## 📚 REFERENCES

Xem `LITERATURE_REVIEW.md` để biết đầy đủ references.

Key papers:
- Petersen RC et al. (2018): Practice guideline update summary: Mild cognitive impairment
- Battista P et al. (2020): Artificial intelligence and neuropsychological measures
- Barnes DE et al. (2009): Predicting risk of dementia in older adults
- Creavin ST et al. (2016): MMSE for the detection of dementia
- Lundberg & Lee (2017): A Unified Approach to Interpreting Model Predictions

---

## 🔄 MIGRATION FROM OLD PIPELINE

### Old Pipeline
```python
# MMSE là output riêng biệt
mmse_score = predictor.predict(features).mmse_estimate
mci_prob = predictor.predict(features).mci_probability
```

### New Pipeline
```python
# MMSE là feature trong input
features['mmse_raw'] = mmse_score
features['age'] = age
features['education_years'] = education

# Dual output
prediction = trainer.predict(X_processed)
risk_binary = prediction.risk_binary
mci_prob = prediction.mci_probability
```

---

**Version:** 2.0  
**Last Updated:** 2025-01-06  
**Author:** Cognitive Assessment System Team

