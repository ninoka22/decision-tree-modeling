# Crash Severity Classification - Sydney

A machine learning project for classifying crash severity levels in Sydney using advanced preprocessing, feature engineering, and imbalanced data handling techniques.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [Feature Engineering](#feature-engineering)
  - [Handling Imbalanced Data](#handling-imbalanced-data)
  - [Feature Selection](#feature-selection)
  - [Model Training](#model-training)
- [Results](#results)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Usage](#usage)

---

## 🎯 Project Overview

This project aims to classify crash severity into three categories:
- **Class 0**: Property Damage Only (PDO)
- **Class 1**: Injury
- **Class 2**: Severe (Fatal or Serious Injury)

The challenge lies in the **significant class imbalance** present in crash data, where injury crashes vastly outnumber severe crashes.

---

## 📊 Dataset

**Source**: Crash data from Sydney, Australia (2016-2023)

| Property | Value |
|----------|-------|
| Total Samples | 5,980 |
| Original Features | 341 |
| Final Features | 30 (after selection) |
| Target Classes | 3 |

### Class Distribution (Original)

| Class | Label | Count | Percentage |
|-------|-------|-------|------------|
| 0 | Property Damage | 1,478 | 24.72% |
| 1 | Injury | 3,362 | 56.22% |
| 2 | Severe | 1,140 | 19.06% |

**Imbalance Ratio**: 2.9:1

---

## 🔬 Methodology

### Data Preprocessing

1. **Missing Value Handling**
   - Amenity columns: Filled with 0 (indicating absence)
   - Distance columns: Removed due to high missing rate (>60%)
   - Columns with 100% missing: Dropped

2. **Feature Type Classification**
   - Numerical features: 269 columns
   - Categorical features: 41 columns
   - Boolean features: 86 columns

3. **Outlier Detection**
   - IQR method applied
   - 52 columns identified with outliers
   - Focus on environmental and weather features

### Feature Engineering

#### Temporal Features
| Feature | Description |
|---------|-------------|
| `season` | Summer/Autumn/Winter/Spring (Southern Hemisphere) |
| `is_weekend` | Binary indicator |
| `hour_sin`, `hour_cos` | Cyclical encoding of hour |
| `month_sin`, `month_cos` | Cyclical encoding of month |
| `is_rush_hour` | 7-9 AM or 5-7 PM indicator |
| `time_of_day` | Night/Morning/Afternoon/Evening |

#### Categorical Aggregation

**RUM (Road User Movement) Categories**:
- Class 1: Rear-end collisions
- Class 2: Pedestrian accidents
- Class 3: Intersection collisions
- Class 4: Other/Off-road

**DCA (Dangerous Crash Analysis) Categories**:
- Similar 4-category grouping

**Vehicle Type Grouping**:
- Car, Heavy, Two-wheel, Pedestrian, Other

#### Feature Reduction
- Constant features removed: 48
- Duplicate features removed: 16
- High correlation features (>0.95): 17 pairs

---

### 🔥 Handling Imbalanced Data

This is the **core focus** of this project. Multiple strategies were explored:

#### Strategy 1: Class Weighting

Custom sample weights assigned to each class during training:

```python
weight_strategies = {
    "No Weight":       {0: 1.0, 1: 1.0, 2: 1.0},
    "Balanced":        {0: 1.35, 1: 0.59, 2: 1.75},
    "Severe x2":       {0: 1.0, 1: 1.0, 2: 2.0},
    "Severe x2.5":     {0: 1.0, 1: 1.0, 2: 2.5},
    "Severe x3":       {0: 1.0, 1: 1.0, 2: 3.0}
}
```

**Results**: No weight strategy performed best (Val F1: 0.5215)

#### Strategy 2: Random Undersampling

Applied **only to training data** to reduce majority class:

| Strategy | Class 0 | Class 1 | Class 2 | Train Size | Val F1 |
|----------|---------|---------|---------|------------|--------|
| No Undersampling | 945 | 2152 | 730 | 3827 | 0.5215 |
| **Mild** | 945 | **1600** | 730 | 3275 | **0.5443** |
| Moderate | 945 | 1400 | 730 | 3075 | 0.5224 |
| Strong | 945 | 1200 | 730 | 2875 | 0.5264 |
| Balanced | 945 | 945 | 730 | 2620 | 0.5138 |

**Best**: Mild undersampling (Val F1: 0.5443, Test F1: 0.5263)

#### Strategy 3: ADASYN (Adaptive Synthetic Sampling)

```python
adasyn = ADASYN(
    sampling_strategy="auto",
    random_state=42,
    n_neighbors=5
)
```

| Metric | Original | After ADASYN |
|--------|----------|--------------|
| Class 0 | 945 | 2,104 |
| Class 1 | 2,152 | 2,152 |
| Class 2 | 730 | 2,009 |
| **Total** | 3,827 | 6,265 |

**Results**: Val F1: 0.4509 (worse than baseline)

#### Strategy 4: Combined Undersampling + Class Weight

25 experiments conducted combining strategies:

| Rank | Undersampling | Class Weight | Val F1 | Severe Recall |
|------|---------------|--------------|--------|---------------|
| 1 | Mild | Severe x1.25 | **0.5208** | 0.4560 |
| 2 | Mild | Severe x1.50 | 0.5193 | 0.5165 |
| 3 | No Undersampling | Severe x1.75 | 0.5190 | 0.4286 |

**Best Combination**: Mild + Severe x1.25

#### Summary of Imbalance Handling Results

| Method | Validation F1 | Test F1 | Notes |
|--------|---------------|---------|-------|
| Baseline (No handling) | 0.5215 | 0.4977 | Reference |
| Class Weight (Balanced) | 0.5073 | - | Slight decrease |
| Undersampling (Mild) | **0.5443** | **0.5263** | Best single method |
| ADASYN | 0.4509 | - | Overfitting |
| Undersampling + Weight | 0.5208 | 0.5214 | Best combined |

**Key Insight**: Mild undersampling of the majority class (Injury) provided the best balance between bias and variance, improving F1 by ~4% over baseline.

---

### Feature Selection

#### Permutation Importance Analysis

Top features by permutation importance (on test data):

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Street lighting_Unknown/not stated | 0.0409 |
| 2 | DCA_category | 0.0319 |
| 3 | RUM_category | 0.0183 |
| 4 | Street lighting_Off | 0.0178 |
| 5 | key_tu_type_group_Two-wheel | 0.0175 |
| 6 | First impact type_Vehicle-Pedestrian | 0.0173 |
| 7 | other_tu_type_group_Two-wheel | 0.0149 |
| 8 | Firstimpacttype_Vehicle-Pedestrian | 0.0140 |
| 9 | Firstimpacttype_Other angle | 0.0131 |
| 10 | other_tu_type_group_Pedestrian | 0.0130 |

#### Feature Selection Curve

Tested Top-K features from 10 to 263:

| Top-K | Train F1 | Val F1 |
|-------|----------|--------|
| 10 | 0.570 | 0.487 |
| **30** | 0.605 | **0.527** |
| 50 | 0.627 | 0.515 |
| 100 | 0.627 | 0.501 |
| 263 | 0.626 | 0.506 |

**Optimal**: Top 30 features (88.59% reduction)

### Model Training

#### Decision Tree
```python
best_params = {
    'criterion': 'entropy',
    'max_depth': 8,
    'min_samples_leaf': 20,
    'min_samples_split': 25
}
```

#### XGBoost
```python
best_params = {
    'n_estimators': 600,
    'max_depth': 2,
    'learning_rate': 0.15,
    'subsample': 0.6,
    'colsample_bytree': 0.9,
    'gamma': 0.5,
    'reg_alpha': 0.1,
    'reg_lambda': 5
}
```

---

## 📈 Results

### Final Model Performance (XGBoost + Mild Undersampling)

| Dataset | Accuracy | Precision | Recall | Macro F1 |
|---------|----------|-----------|--------|----------|
| Train | 86.35% | 87.68% | 85.40% | 0.8635 |
| Validation | 58.10% | 54.14% | 55.02% | **0.5443** |
| Test | 57.02% | 52.28% | 53.62% | **0.5263** |

### Per-Class Performance (Test Set)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Property Damage | 0.531 | 0.662 | 0.590 | 296 |
| Injury | 0.652 | 0.609 | 0.630 | 672 |
| Severe | 0.385 | 0.338 | 0.360 | 228 |

### Confusion Matrix (Test Set)

```
                 Predicted
              PDO   Injury  Severe
Actual PDO  [ 196    90      10 ]
       Inj  [ 150   409     113 ]
       Sev  [  23   128      77 ]
```

---

## 🔍 Key Findings

### SHAP Analysis

#### Global Feature Importance (Mean |SHAP|)

| Rank | Feature | Mean |SHAP| |
|------|---------|------------|
| 1 | Street lighting_Unknown/not stated | 0.563 |
| 2 | Firstimpacttype_Vehicle-Pedestrian | 0.170 |
| 3 | key_tu_type_group_Two-wheel | 0.137 |
| 4 | Latitude | 0.128 |
| 5 | other_tu_type_group_Two-wheel | 0.126 |

#### Severe Class (Class 2) Feature Importance

| Rank | Feature | Mean |SHAP| |
|------|---------|------------|
| 1 | Street lighting_Unknown/not stated | 1.171 |
| 2 | OtherTUtype_Car (sedan/hatch) | 0.162 |
| 3 | Latitude | 0.156 |
| 4 | key_tu_type_group_Two-wheel | 0.132 |
| 5 | cloud_cover_high_PR_mean | 0.118 |

### Critical Factors for Crash Severity

1. **Street Lighting**: Unknown/poor lighting conditions are strongly associated with severe crashes
2. **Vehicle Type**: Two-wheel vehicles (motorcycles, bicycles) significantly increase severity risk
3. **Pedestrian Involvement**: Vehicle-pedestrian impacts are highly predictive of severe outcomes
4. **Location**: Geographic coordinates capture area-specific risk patterns
5. **Weather Conditions**: Cloud cover and wind speed show moderate importance

---

## 📁 Project Structure

```
crash-severity-classification/
├── README.md
├── requirements.txt
├── notebooks/
│   └── ml_classify.ipynb          # Main analysis notebook
├── data/
│   ├── raw/
│   │   └── CrashSeverityData-Sydney.csv
│   └── processed/
│       └── processed_data4.csv
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── imbalance_handling.py
│   └── models.py
├── results/
│   ├── figures/
│   ├── shap_analysis/
│   └── model_results/
└── models/
    └── best_model.pkl
```

---

## 🛠️ Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
imbalanced-learn>=0.9.0
shap>=0.41.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

---

## 🚀 Usage

### 1. Data Preparation

```python
import pandas as pd

# Load raw data
df = pd.read_csv('data/raw/CrashSeverityData-Sydney.csv')

# Create 3-class target
import numpy as np
df['severity_3class'] = np.select(
    [
        (df['No. killed'] > 0) | (df['No. seriously injured'] > 0),
        ((df['No. moderately injured'] > 0) | (df['No. minor-other injured'] > 0))
    ],
    [2, 1],
    default=0
)
```

### 2. Apply Imbalance Handling

```python
from imblearn.under_sampling import RandomUnderSampler

# Define sampling strategy
sampling_strategy = {
    0: 945,   # Keep all PDO
    1: 1600,  # Reduce Injury (from 2152)
    2: 730    # Keep all Severe
}

# Apply to training data only!
sampler = RandomUnderSampler(
    sampling_strategy=sampling_strategy,
    random_state=42
)
X_train_resampled, y_train_resampled = sampler.fit_resample(
    X_train, y_train
)
```

### 3. Train Model

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=600,
    max_depth=2,
    learning_rate=0.15,
    subsample=0.6,
    colsample_bytree=0.9,
    random_state=42
)

model.fit(X_train_resampled, y_train_resampled)
```

---

## 📝 Recommendations

### For Practitioners

1. **Use Mild Undersampling**: Reducing majority class to ~1600 samples improved F1 by 4%
2. **Avoid ADASYN**: Synthetic sampling led to overfitting on this dataset
3. **Focus on Feature Quality**: Top 30 features capture most predictive power
4. **Consider Class Weights**: When undersampling is not feasible, use moderate class weights

### For Future Work

- [ ] Explore SMOTE variants (Borderline-SMOTE, SVM-SMOTE)
- [ ] Implement cost-sensitive learning
- [ ] Test ensemble methods (BalancedBagging, EasyEnsemble)
- [ ] Investigate deep learning approaches with focal loss
- [ ] Add spatial autocorrelation features

---

## 📚 References

1. He, H., et al. (2008). ADASYN: Adaptive synthetic sampling approach for imbalanced learning. IEEE IJCNN.
2. Chawla, N. V., et al. (2002). SMOTE: Synthetic minority over-sampling technique. JAIR.
3. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. KDD.
4. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. NeurIPS.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👥 Contributors

- [kawe saidy](https://github.com/ninoka22)

---

## ⭐ Acknowledgments

- NSW Centre for Road Safety for crash data
- OpenStreetMap contributors for road network data
- Open-Meteo for weather data
