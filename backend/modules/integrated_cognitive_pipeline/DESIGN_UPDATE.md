# DESIGN UPDATE: MMSE LÀ INPUT FEATURE, KHÔNG PHẢI OUTPUT

## 🔄 THAY ĐỔI THIẾT KẾ

### Trước đây (SAI):
- Model estimate MMSE score từ prediction
- MMSE là output của model

### Bây giờ (ĐÚNG):
- **MMSE score từ chatbot test** → Pass vào model như một **input feature**
- Model chỉ output:
  1. **Binary Classification**: Risk Yes/No
  2. **Probability Estimation**: MCI Probability [0-1]

---

## 📋 KIẾN TRÚC MỚI

```
CHATBOT MMSE TEST
    ↓
MMSE Score (0-30) → INPUT FEATURE
    ↓
FEATURE ENGINEERING
    ├── MMSE normalization (age + education)
    ├── Acoustic features
    ├── Linguistic features
    ├── Demographic features
    └── Clinical features
    ↓
DUAL-OUTPUT MODEL
    ├── Head 1: Binary Classifier → Risk Yes/No
    └── Head 2: Probability Estimator → MCI Probability [0-1]
    ↓
OUTPUT
    ├── risk_binary: True/False
    ├── risk_binary_probability: 0-1
    ├── mci_probability: 0-1
    └── predicted_class: Normal/MCI/AD
```

---

## ✅ CHANGES MADE

### 1. Removed MMSE Estimation Logic ✅
- ❌ Removed `_estimate_mmse_from_dual_output()` method
- ✅ Added `_get_mmse_from_features()` method để lấy MMSE từ chatbot test

### 2. Updated AnalysisResult ✅
- Changed `mmse_estimate` → `mmse_score`
- Added comment: MMSE từ chatbot test, KHÔNG phải estimate

### 3. Updated Prediction Flow ✅
- MMSE score được lấy từ `metadata['mmse']` hoặc `features['mmse_raw']`
- MMSE được normalize và pass vào model như feature
- Model output chỉ có: risk_binary và mci_probability

### 4. Updated Documentation ✅
- Added warnings trong code
- Updated comments

---

## 🔍 USAGE

### Pass MMSE từ Chatbot Test

```python
from backend.modules.integration_service import MCIScreeningService

service = MCIScreeningService(use_integrated_pipeline=True)

# MMSE score từ chatbot test
mmse_from_chatbot = 25.0  # Điểm từ bài kiểm tra chatbot

# Pass vào metadata
result = service.analyze(
    transcript="...",
    metadata={
        'age': 70,
        'education': 12,
        'mmse': mmse_from_chatbot  # ✅ MMSE từ chatbot test
    }
)

# Model sẽ:
# 1. Normalize MMSE theo age/education
# 2. Pass MMSE vào model như feature
# 3. Output: risk_binary và mci_probability
```

---

## 📊 OUTPUT FORMAT

### Dual-Output Prediction (New)

```python
{
    # Binary classification
    'risk_binary': True,  # Có nguy cơ hay không
    'risk_binary_probability': 0.72,  # Probability của risk
    
    # Probability estimation
    'mci_probability': 0.65,  # MCI probability [0-1]
    'mci_probability_calibrated': 0.63,  # Calibrated
    
    # Class prediction
    'predicted_class': 'MCI',
    'class_probabilities': {
        'Normal': 0.28,
        'MCI': 0.65,
        'AD': 0.07
    },
    
    # MMSE score (từ chatbot test, KHÔNG phải estimate)
    'mmse_score': 25.0,  # Từ chatbot test
    'severity': 'Bình thường'  # Classify từ MMSE score
}
```

---

## ⚠️ IMPORTANT NOTES

1. **MMSE Source**: MMSE score PHẢI được lấy từ chatbot test, không được estimate từ model.

2. **Feature Engineering**: MMSE score sẽ được normalize theo age và education trước khi pass vào model.

3. **Model Output**: Model chỉ output risk_binary và mci_probability. MMSE score chỉ dùng để classify severity.

4. **Backward Compatibility**: Code vẫn có `mmse_estimate` field trong một số places để backward compatibility, nhưng nó sẽ lấy từ chatbot test, không estimate.

---

**Status:** ✅ **UPDATED**  
**Date:** 2025-01-07

