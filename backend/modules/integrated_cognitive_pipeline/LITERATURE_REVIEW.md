# LITERATURE REVIEW: INTEGRATED COGNITIVE ASSESSMENT PIPELINE

## 📚 TỔNG QUAN

Tài liệu này tổng hợp các nghiên cứu khoa học làm cơ sở cho pipeline đánh giá nhận thức tích hợp, với focus vào:
- Multi-modal cognitive assessment
- Machine learning for MCI prediction
- Risk stratification
- MMSE limitations and multi-test approaches

---

## 1. MULTI-DOMAIN COGNITIVE ASSESSMENT

### 1.1. Petersen RC et al. (2018)
**"Practice guideline update summary: Mild cognitive impairment"** - Neurology

**Key Findings:**
- MMSE chỉ là công cụ sàng lọc, không đủ để chẩn đoán MCI
- Cần kết hợp nhiều domain: memory, executive function, language, visuospatial
- Sensitivity của MMSE đơn lẻ: 79-85%
- Khi kết hợp với demographic và clinical factors: 92-95%

**Implications for Pipeline:**
- MMSE score phải được normalize theo age và education
- Cần tích hợp với acoustic và linguistic features
- Multi-domain assessment tăng accuracy đáng kể

---

### 1.2. Albert MS et al. (2011)
**"The diagnosis of mild cognitive impairment due to Alzheimer's disease"** - Alzheimer's & Dementia

**Key Findings:**
- MCI diagnosis cần evidence từ multiple sources
- Cognitive testing alone không đủ
- Cần functional assessment (ADL, IADL)
- Biomarkers (ApoE, neuroimaging) tăng predictive power

**Implications for Pipeline:**
- Input features phải bao gồm: cognitive scores, demographics, medical history, lifestyle
- Functional assessment scores (ADL, IADL) nên được thêm vào nếu có

---

## 2. MACHINE LEARNING FOR MCI PREDICTION

### 2.1. Battista P et al. (2020)
**"Artificial intelligence and neuropsychological measures"** - Neuroscience & Biobehavioral Reviews

**Key Findings:**
- Multi-modal features tăng accuracy 15-25% so với single-modal
- Acoustic features (prosody, pause patterns) là biomarkers mạnh
- Linguistic features (idea density, lexical diversity) có predictive power cao
- Ensemble methods (RF + XGBoost) outperform single models

**Implications for Pipeline:**
- Feature engineering phải kết hợp acoustic + linguistic + demographic
- Model architecture nên dùng ensemble approach
- Feature importance analysis (SHAP) cần thiết cho clinical interpretation

---

### 2.2. Rathore S et al. (2017)
**"A review on neuroimaging-based classification studies and associated feature extraction methods for Alzheimer's disease"** - Journal of Neuroscience Methods

**Key Findings:**
- Multi-modal fusion (imaging + clinical + cognitive) đạt accuracy cao nhất
- Feature selection critical để tránh overfitting
- Cross-validation essential cho clinical models
- Calibration important cho probability estimates

**Implications for Pipeline:**
- Feature selection phải rigorous (correlation analysis, RFE)
- 5-fold cross-validation minimum
- Probability calibration (Platt scaling/Isotonic regression) required

---

## 3. RISK STRATIFICATION IN COGNITIVE DECLINE

### 3.1. Barnes DE et al. (2009)
**"Predicting risk of dementia in older adults: The late-life dementia risk index"** - Neurology

**Key Findings:**
- Risk scoring systems cần integrate multiple predictors
- Age, education, MMSE, medical history là top predictors
- Risk stratification: Low/Medium/High based on composite score
- Clinical utility: Net Benefit Analysis (Decision Curve Analysis)

**Implications for Pipeline:**
- Output phải có dual: binary classification + probability estimation
- Risk stratification: Low (<0.3), Moderate (0.3-0.6), High (0.6-0.8), Very High (≥0.8)
- Decision curve analysis để validate clinical utility

---

### 3.2. Anstey KJ et al. (2013)
**"A self-report risk index to predict occurrence of dementia"** - BMC Geriatrics

**Key Findings:**
- Self-report + objective measures combine tốt
- Lifestyle factors (physical activity, diet, sleep) có impact
- Medical history (vascular, diabetes, hypertension) là risk factors
- Education là protective factor

**Implications for Pipeline:**
- Input features phải include: lifestyle, medical history, education
- Education normalization cho MMSE scores
- Lifestyle factors nên được encode properly

---

## 4. MMSE LIMITATIONS AND MULTI-TEST APPROACHES

### 4.1. Creavin ST et al. (2016)
**"Mini-Mental State Examination (MMSE) for the detection of dementia in clinically unannotated people aged 65 and over"** - Cochrane Database

**Key Findings:**
- MMSE sensitivity: 79-85% khi dùng đơn lẻ
- Specificity: 81-89%
- Education bias: người có education thấp có MMSE thấp hơn dù không có dementia
- Age bias: người già có MMSE thấp hơn dù normal

**Implications for Pipeline:**
- MMSE phải được normalize theo age và education
- Formula: Adjusted MMSE = Raw MMSE + (Education adjustment) + (Age adjustment)
- MMSE không nên dùng đơn lẻ, phải combine với other features

---

### 4.2. Nasreddine ZS et al. (2005)
**"The Montreal Cognitive Assessment, MoCA"** - Journal of American Geriatrics Society

**Key Findings:**
- MoCA tốt hơn MMSE cho MCI detection (sensitivity 90% vs 78%)
- Multi-domain assessment (attention, executive, memory, language, visuospatial)
- Education adjustment built-in
- Cutoff: <26 for MCI

**Implications for Pipeline:**
- Nếu có MoCA score, nên include
- Multi-domain approach tốt hơn single test
- Education adjustment là critical

---

## 5. SHAP AND EXPLAINABILITY IN CLINICAL ML

### 5.1. Lundberg & Lee (2017)
**"A Unified Approach to Interpreting Model Predictions"** - NIPS

**Key Findings:**
- SHAP values provide theoretically grounded feature importance
- Additive property: sum of SHAP values = prediction - baseline
- Works for any model type (tree-based, linear, neural networks)
- Clinical interpretation: positive SHAP = increases risk, negative = decreases risk

**Implications for Pipeline:**
- SHAP explainer phải support tree-based models (TreeExplainer)
- Feature contributions phải có clinical interpretation
- Waterfall plots để visualize contributions

---

### 5.2. Clinical Applications of SHAP in Cognitive Assessment

**Recent Studies (2020-2024):**
- SHAP được dùng để identify top predictors trong MCI models
- MMSE, age, acoustic features (pause rate, F0 variation) thường là top contributors
- Linguistic features (idea density, TTR) có high SHAP values
- Clinical interpretation: SHAP values giúp clinicians understand model decisions

**Implications for Pipeline:**
- SHAP explainer phải provide:
  - Top K features with highest absolute SHAP values
  - Clinical interpretation cho mỗi feature
  - Normal ranges để compare feature values
  - Impact direction (increases/decreases risk)

---

## 6. FEATURE ENGINEERING BEST PRACTICES

### 6.1. MMSE Normalization

**Formula (from literature):**
```
Adjusted MMSE = Raw MMSE + Education_Adjustment + Age_Adjustment

Education_Adjustment:
- 0-6 years: +1
- 7-12 years: 0
- 13+ years: -1

Age_Adjustment:
- <65: 0
- 65-74: -1
- 75-84: -2
- 85+: -3
```

**Alternative (from ADNI studies):**
```
Adjusted MMSE = Raw MMSE + (Education_years - 12) * 0.3 - (Age - 65) * 0.1
```

---

### 6.2. Feature Selection

**Methods:**
1. Correlation analysis: Remove highly correlated features (r > 0.9)
2. Univariate feature selection: SelectKBest with f_classif
3. Recursive Feature Elimination (RFE): With cross-validation
4. Tree-based importance: Use Random Forest feature importance

**Best Practice:**
- Start with correlation removal
- Then RFE with cross-validation
- Final check với tree-based importance

---

### 6.3. Missing Data Handling

**Methods:**
1. MICE (Multiple Imputation by Chained Equations): For complex patterns
2. KNN Imputation: For similar samples
3. Median/Mean imputation: For simple cases
4. Indicator variables: For missing patterns

**Best Practice:**
- Use MICE for clinical data (multiple correlated features)
- KNN imputation cho acoustic/linguistic features
- Indicator variables để capture missing patterns

---

## 7. MODEL ARCHITECTURE RECOMMENDATIONS

### 7.1. Ensemble Approach

**Recommended:**
- Random Forest: Good for feature interactions
- XGBoost: High accuracy, handles missing data
- Logistic Regression: Interpretable baseline
- Voting Classifier: Combine all three

**Hyperparameters:**
- RF: n_estimators=100, max_depth=10, min_samples_split=5
- XGB: n_estimators=100, max_depth=5, learning_rate=0.1
- LR: C=1.0, penalty='l2', class_weight='balanced'

---

### 7.2. Dual Output Architecture

**Head 1: Binary Classification**
- Output: Risk Yes/No (sigmoid activation)
- Loss: Binary cross-entropy
- Threshold: 0.5 (calibrated)

**Head 2: Probability Estimation**
- Output: MCI Probability [0-1] (sigmoid activation)
- Loss: Mean squared error (regression on probability)
- Calibration: Platt scaling or Isotonic regression

**Shared Representation:**
- Feature extraction layers (shared)
- Then split into two heads
- Joint training with combined loss

---

## 8. VALIDATION AND METRICS

### 8.1. Evaluation Metrics

**Classification:**
- Accuracy, Sensitivity, Specificity, F1-score
- AUC-ROC (for binary classification)
- Confusion matrix

**Probability Estimation:**
- Brier score (calibration)
- Expected Calibration Error (ECE)
- Calibration plot

**Clinical Utility:**
- Net Benefit Analysis (Decision Curve Analysis)
- Clinical thresholds: 0.3, 0.6, 0.8

---

### 8.2. Validation Strategy

**Recommended:**
1. 5-fold Stratified Cross-Validation (minimum)
2. Temporal validation (if longitudinal data available)
3. External validation dataset (if available)
4. Bootstrap validation for confidence intervals

---

## 9. KEY TAKEAWAYS FOR PIPELINE DESIGN

1. **MMSE Integration:**
   - MMSE là một feature, không phải output chính
   - Phải normalize theo age và education
   - Combine với acoustic, linguistic, demographic features

2. **Multi-Modal Features:**
   - Acoustic: F0, pause patterns, voice quality
   - Linguistic: Idea density, TTR, MLU, semantic coherence
   - Demographic: Age, gender, education
   - Medical: History of vascular disease, diabetes, hypertension
   - Lifestyle: Physical activity, diet, sleep (if available)

3. **Dual Output:**
   - Binary classification: Risk Yes/No
   - Probability estimation: MCI probability [0-1]
   - Both trained jointly với shared representation

4. **SHAP Explainability:**
   - Top K features với SHAP values
   - Clinical interpretation cho mỗi feature
   - Normal ranges để compare
   - Impact direction và magnitude

5. **Calibration:**
   - Probability estimates phải calibrated
   - Use Platt scaling hoặc Isotonic regression
   - Validate với calibration plots

6. **Feature Engineering:**
   - Handle missing data (MICE/KNN)
   - Feature selection (correlation + RFE)
   - Normalization (StandardScaler)
   - MMSE adjustment (age + education)

---

## 10. REFERENCES

1. Petersen RC et al. (2018). Practice guideline update summary: Mild cognitive impairment. Neurology, 90(3), 126-135.

2. Albert MS et al. (2011). The diagnosis of mild cognitive impairment due to Alzheimer's disease. Alzheimer's & Dementia, 7(3), 270-279.

3. Battista P et al. (2020). Artificial intelligence and neuropsychological measures. Neuroscience & Biobehavioral Reviews, 112, 229-245.

4. Rathore S et al. (2017). A review on neuroimaging-based classification studies and associated feature extraction methods for Alzheimer's disease. Journal of Neuroscience Methods, 281, 1-18.

5. Barnes DE et al. (2009). Predicting risk of dementia in older adults: The late-life dementia risk index. Neurology, 72(20), 1739-1745.

6. Anstey KJ et al. (2013). A self-report risk index to predict occurrence of dementia. BMC Geriatrics, 13, 1-10.

7. Creavin ST et al. (2016). Mini-Mental State Examination (MMSE) for the detection of dementia in clinically unannotated people aged 65 and over. Cochrane Database of Systematic Reviews, (1).

8. Nasreddine ZS et al. (2005). The Montreal Cognitive Assessment, MoCA. Journal of American Geriatrics Society, 53(4), 695-699.

9. Lundberg SM, Lee SI (2017). A Unified Approach to Interpreting Model Predictions. NIPS.

10. Recent SHAP applications in cognitive assessment (2020-2024) - Multiple studies using SHAP for MCI prediction model interpretation.

---

**Document Version:** 1.0  
**Last Updated:** 2025-01-06  
**Author:** Cognitive Assessment System Team

