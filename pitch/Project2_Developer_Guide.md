# Project 2: Predicting 30-Day Readmission for Diabetic Inpatients
## Complete Developer Guide

---

## 1. Project Summary

### What We're Building

A binary classification pipeline that predicts whether a diabetic inpatient will be readmitted to a hospital within 30 days of discharge. The model consumes clinical, demographic, and medication data available at discharge time and outputs a probability score that hospitals can use to triage transitional care resources.

### Why It Matters

The CMS Hospital Readmissions Reduction Program (HRRP) penalizes hospitals up to 3% of all Medicare inpatient payments for excess 30-day readmissions. In FY 2024, roughly 80% of evaluated hospitals were penalized, totaling over $560 million industry-wide. A model that identifies high-risk patients before discharge lets hospitals concentrate expensive interventions (medication reconciliation, discharge coaching, 48-hour follow-up calls, home health referrals) on the patients who actually need them rather than applying them uniformly.

### Dataset

- **Source:** UCI Machine Learning Repository, ID 296 (Strack et al., 2014)
- **Size:** 101,766 inpatient encounters x 50 features
- **Span:** 10 years (1999-2008) across 130 US hospitals
- **Target:** `readmitted` column -- values `<30`, `>30`, `NO` → binarized to `<30 = 1, else = 0`
- **License:** CC BY 4.0

### Stakeholders

| Role | How They Use It |
|------|----------------|
| VP of Case Management / UM | Patient-level risk flags to prioritize intensive discharge follow-up |
| CMO | Clinical policy on discharge protocols, benchmarking vs. peers |
| CFO | Quantify HRRP penalty exposure, estimate ROI on intervention programs |
| Hospitalists / Endocrinologists | Risk alerts at discharge with actionable feature insights (e.g., "HbA1c not measured") |
| Quality / Population Health | Aggregate trend monitoring, intervention effectiveness tracking |

### Success Metrics

| Metric | Target | Why |
|--------|--------|-----|
| ROC-AUC | > 0.65 | Primary comparison metric. Published benchmark is 0.64-0.67 for this dataset. |
| Recall | > 0.50 | A missed readmission = a penalty. High recall is more valuable than high precision. |
| F1-Score | Maximize | Balances missed readmissions vs. wasted intervention spend. |
| Cost-Adjusted Net Benefit | Positive | Custom: (TP x penalty_saved) − (FP x intervention_cost). Does deploying this model save money? |

---

## 2. Overall Approach

### Methodology: CRISP-DM

The project follows the Cross-Industry Standard Process for Data Mining:

1. **Business Understanding** → Completed in the pitch document
2. **Data Understanding** → Phase 1 (Profiling) + Phase 2 (EDA)
3. **Data Preparation** → Phase 3 (Cleaning, Feature Engineering, Pipeline)
4. **Modeling** → Phase 4 (Training) + Phase 5 (Tuning)
5. **Evaluation** → Phase 6 (Final Evaluation, SHAP, Business Impact)
6. **Deployment** → Phase 7 (Documentation, Presentation)

### Architecture

```
Raw CSV (101K rows x 50 cols)
    │
    ▼
Phase 1-2: Profile + EDA  →  Hypotheses about what matters
    │
    ▼
Phase 3: Clean + Engineer  →  ~70K rows x ~35 features (after dedup, drops, engineering)
    │
    ▼
Phase 3: sklearn Pipeline  →  ColumnTransformer(numeric | categorical | binary) → Classifier
    │
    ▼
Phase 4-5: Train + Tune   →  4 models x 5-fold CV x hyperparameter search
    │
    ▼
Phase 6: Evaluate          →  Test-set metrics, SHAP plots, cost simulation
    │
    ▼
Phase 7: Communicate       →  Executive presentation, notebook cleanup, GitHub push
```

### Libraries

```
pandas          -- Data manipulation, profiling
numpy           -- Numerical operations
matplotlib      -- Static visualizations
seaborn         -- Statistical visualizations
scikit-learn    -- Pipeline, preprocessing, models (LR, RF, SVM), metrics, CV, tuning
xgboost         -- Gradient-boosted classifier
imbalanced-learn -- SMOTE oversampling within CV folds
shap            -- Model interpretability (summary plots, force plots)
ucimlrepo       -- Dataset download from UCI
jupyter         -- Notebook environment
```

Install command:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap imbalanced-learn ucimlrepo jupyter
```

---

## 3. Phase 1 -- Data Acquisition and Profiling

**Goal:** Load the dataset, understand its structure, quantify data quality issues, and confirm the target distribution before any transformations.

**Status:** Complete (Week 1 notebook, cells 0-35)

### 3.1 Data Loading

```python
DATA_PATH = './Data/diabetic_data.csv'
df = pd.read_csv(DATA_PATH)
# 101,766 rows x 50 columns
```

If the CSV isn't present, download via:
```python
from ucimlrepo import fetch_ucirepo
ds = fetch_ucirepo(id=296)
ds.data.original.to_csv(DATA_PATH, index=False)
```

**Rationale:** The CSV download is a one-time operation. We store the file locally rather than fetching it every run, since the dataset doesn't change and is ~18MB.

### 3.2 Schema Profiling

For every column, compute: dtype, non-null count, null count, null percentage, unique count, and `?`-mark count (the dataset uses `?` as a missing sentinel instead of NaN).

**Key findings:**
- `weight`: ~97% missing → drop
- `payer_code`: ~40% missing, not clinically relevant → drop
- `medical_specialty`: ~50% missing, ~70 unique values → impute as "Unknown", collapse rare categories
- `race`: ~2% missing → impute as "Unknown"
- `diag_1/2/3`: ~700 unique ICD-9 codes each → must group into clinical categories
- 23 medication columns: categorical ("No"/"Steady"/"Up"/"Down"), many near-zero-variance

### 3.3 Target Distribution

Raw values: NO (54.9%), >30 (34.7%), <30 (11.2%). After binarization to `<30 = 1, else = 0`: roughly 11% positive class, 89% negative. Imbalance ratio ~8:1.

**Rationale for binarization:** The business problem is specifically about 30-day readmission (the HRRP penalty window). The `>30` category is clinically and financially distinct from `<30`. Collapsing `>30` and `NO` into a single negative class is standard for this dataset and matches the Strack et al. approach.

### 3.4 Duplicate Patient Check

~101K encounters come from ~71K unique patients. Some patients appear up to 3+ times. This creates a data leakage risk: if the same patient is in both train and test, the model may memorize patient-specific patterns.

**Decision:** Keep one encounter per patient during cleaning (Phase 3).

### 3.5 Deliverables

- [x] Dataset loaded and confirmed (101,766 x 50)
- [x] Every column profiled (dtype, nulls, uniques, `?` counts)
- [x] Target distribution documented (~11% positive)
- [x] Missing data quantified and drop/impute decisions recorded
- [x] Duplicate patients identified (~30K excess rows)
- [x] 4 figures saved to `./figures/`

---

## 4. Phase 2 -- Exploratory Data Analysis

**Goal:** Move beyond profiling into statistical exploration. Understand feature-target relationships, identify interactions, and form hypotheses that will guide feature engineering and model selection.

### 4.1 Numeric Feature Analysis

**What to do:**

Compute correlation matrix for all numeric features against `readmit_30`. Visualize with a heatmap. Identify the strongest univariate correlates.

```python
# Correlation with target
numeric_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
                'num_medications', 'number_outpatient', 'number_emergency',
                'number_inpatient', 'number_diagnoses']

corr_with_target = df[numeric_cols + ['readmit_30']].corr()['readmit_30'].drop('readmit_30')
corr_with_target.sort_values(ascending=False)
```

**Expected:** `number_inpatient` will show the strongest positive correlation. `number_outpatient` and `number_emergency` will also be positive but weaker. `num_lab_procedures` and `num_medications` capture treatment intensity.

**Rationale:** Univariate correlations give a baseline for feature importance before modeling. Features that show zero correlation here are unlikely to matter, though interactions may still surface in tree-based models.

### 4.2 Categorical Feature Deep-Dives

For each key categorical feature, compute the 30-day readmission rate per level and test for significance:

**a) Discharge Disposition**

```python
# Map numeric IDs to readable labels using IDs_mapping.csv
# Compute readmission rate per disposition
# Key finding: "Discharged to home" vs. "Transferred to SNF" vs. "Left AMA"
# will show very different readmission profiles
```

**Rationale:** Discharge disposition is one of the strongest clinical predictors of readmission. Patients discharged to skilled nursing facilities or against medical advice have higher readmission rates. This also identifies the "Expired" codes (11, 19, 20, 21) that must be dropped.

**b) Admission Type and Source**

```python
# admission_type_id: Emergency (1), Urgent (2), Elective (3), etc.
# admission_source_id: Physician referral, ER, Transfer, etc.
# Compute readmission rates per type/source
```

**Rationale:** Emergency admissions may predict higher readmission. Transfer patients have different risk profiles than direct admits.

**c) HbA1c and Glucose Testing**

```python
# Cross-tab: A1Cresult x readmit_30
# Key hypothesis from Strack paper: patients whose HbA1c was measured
# (regardless of result) had lower readmission rates
# This suggests the ACT of measuring is a proxy for better diabetes management
```

**Rationale:** This is the central finding of the original research paper. The `has_A1c_measured` engineered feature directly targets this.

**d) Medication Change Patterns**

```python
# For each of 23 medication columns:
#   - What % of patients had a dosage change (Up or Down)?
#   - Does having ANY change correlate with readmission?
#   - Does the NUMBER of changes correlate?

# Cross-tab: change (yes/no) x diabetesMed (yes/no) x readmit_30
```

**Rationale:** Medication adjustment intensity is a clinical signal. A patient whose medications were heavily modified during the stay may be less stable. Conversely, a patient on diabetes medications whose doses were NOT adjusted may be undertreated.

### 4.3 Feature Interaction Analysis

```python
# Key interactions to test:
# 1. number_inpatient x time_in_hospital (frequent flyers with long stays)
# 2. A1Cresult x change (was diabetes managed AND medication adjusted?)
# 3. number_diagnoses x num_medications (complexity x polypharmacy)
# 4. age x number_inpatient (older frequent flyers)
```

**Rationale:** Tree-based models will learn these interactions automatically, but documenting them in the EDA provides interpretive context for SHAP results later.

### 4.4 Statistical Tests

| Test | Variables | Purpose |
|------|-----------|---------|
| Chi-square | discharge_disposition x readmit_30 | Is disposition associated with readmission? |
| Chi-square | A1Cresult x readmit_30 | Does HbA1c measurement matter? |
| Mann-Whitney U | number_inpatient: readmit vs. not | Do readmitted patients have more prior visits? |
| ANOVA or Kruskal-Wallis | time_in_hospital across readmission groups | Does LOS predict readmission? |
| Point-biserial | Each medication change x readmit_30 | Which medication adjustments associate with readmission? |

**Rationale:** With ~70K+ rows after dedup, all tests will be significant. Report effect sizes (Cramér's V, rank-biserial r, eta-squared) alongside p-values. This mirrors the approach from Project 1 and is an explicit rubric expectation.

### 4.5 Visualizations Checklist

At minimum, produce these charts (each with a dedicated interpretation cell):

1. Target distribution (3-class and binary) -- done in Phase 1
2. Missing data bar chart -- done in Phase 1
3. Readmission rate by age group -- done in Phase 1
4. Readmission rate by prior inpatient visits -- done in Phase 1
5. Readmission rate by HbA1c result -- done in Phase 1
6. Readmission rate by primary diagnosis group -- done in Phase 1
7. Numeric feature distributions (histograms + KDE) -- done in Phase 1
8. Categorical feature distributions -- done in Phase 1
9. Correlation heatmap (numeric features)
10. Discharge disposition readmission rates
11. Admission type readmission rates
12. Medication change analysis (insulin, metformin, overall)
13. HbA1c measured vs. not measured -- readmission rate comparison
14. Feature interaction heatmap (top pairs)
15. Number of diagnoses x readmission (with confidence intervals)
16. Encounters per patient distribution -- done in Phase 1

### 4.6 Deliverables

- [ ] Correlation matrix computed and visualized
- [ ] Chi-square tests on key categorical features with effect sizes
- [ ] Mann-Whitney U on numeric features with rank-biserial r
- [ ] Discharge disposition analysis (with expired codes confirmed)
- [ ] HbA1c measurement analysis (confirms Strack finding)
- [ ] Medication change analysis
- [ ] All visualizations saved to `./figures/`
- [ ] Each chart has a dedicated interpretation cell in the notebook

---

## 5. Phase 3 -- Data Cleaning, Feature Engineering, and Pipeline

**Goal:** Transform raw data into a model-ready matrix through a reproducible scikit-learn Pipeline. Every step must be justified by evidence from Phases 1-2.

### 5.1 Cleaning Steps (In Order)

The order matters. Later steps depend on earlier ones.

**Step 1: Replace `?` with NaN**

```python
df = df.replace('?', np.nan)
```

**Rationale:** The dataset uses `?` as a sentinel for missing categoricals. Replacing globally before any other operation ensures all downstream imputation logic works correctly. Must happen first because pandas operations like `.isnull()` won't detect `?` as missing.

**Step 2: Binarize the target**

```python
df['readmit_30'] = (df['readmitted'] == '<30').astype(int)
df = df.drop(columns=['readmitted'])
```

**Rationale:** The raw target has three classes. For the HRRP business problem, only the <30 window matters. Binarizing early means all subsequent readmission rate calculations use the final target definition, avoiding confusion.

**Step 3: Remove expired patients**

```python
expired_codes = [11, 19, 20, 21]
df = df[~df['discharge_disposition_id'].isin(expired_codes)]
```

**Rationale:** A deceased patient cannot be readmitted. Including them inflates the negative class artificially. The `discharge_disposition_id` codes for expiration are 11 (Expired), 19 (Expired at home, Medicaid hospice), 20 (Expired in medical facility, Medicaid hospice), 21 (Expired, place unknown). This is a universally applied step in published analyses of this dataset.

**Step 4: Deduplicate by patient**

```python
# Keep the encounter with the longest time_in_hospital per patient
df = df.sort_values('time_in_hospital', ascending=False)
df = df.drop_duplicates(subset='patient_nbr', keep='first')
```

**Rationale:** Multiple encounters from the same patient are not independent. If the same patient appears in both train and test sets, the model can memorize patient-specific patterns rather than learning generalizable readmission risk factors. Keeping the longest stay gives the encounter with the most clinical information (more procedures, more medication decisions, more diagnostic data). After dedup, drop `patient_nbr` since it's an identifier, not a feature.

**Step 5: Drop columns**

```python
drop_cols = [
    'encounter_id',      # Row ID -- no predictive value
    'patient_nbr',       # Patient ID -- removed after dedup to prevent leakage
    'weight',            # ~97% missing -- irrecoverable
    'payer_code',        # ~40% missing, not clinically relevant to readmission
]
df = df.drop(columns=drop_cols)
```

**Rationale:** IDs are identifiers, not features. `weight` has too little data to impute meaningfully. `payer_code` describes insurance type, which while potentially correlated with readmission, is (a) heavily missing, and (b) not an actionable feature for clinical intervention. Keep the feature set focused on variables the hospital can observe and act on at discharge.

**Step 6: Group ICD-9 diagnosis codes**

```python
def classify_icd9(code):
    """Map ICD-9 code to 9 clinical categories per Strack et al. Table 2."""
    if pd.isna(code):
        return 'Other'
    try:
        if str(code).startswith(('V', 'E')):
            return 'Other'
        num = float(code)
    except ValueError:
        return 'Other'

    if 390 <= num <= 459 or num == 785:
        return 'Circulatory'
    elif 460 <= num <= 519 or num == 786:
        return 'Respiratory'
    elif 520 <= num <= 579 or num == 787:
        return 'Digestive'
    elif 250 <= num < 251:
        return 'Diabetes'
    elif 800 <= num <= 999:
        return 'Injury'
    elif 710 <= num <= 739:
        return 'Musculoskeletal'
    elif 580 <= num <= 629 or num == 788:
        return 'Genitourinary'
    elif 140 <= num <= 239:
        return 'Neoplasms'
    else:
        return 'Other'

for col in ['diag_1', 'diag_2', 'diag_3']:
    df[col] = df[col].apply(classify_icd9)
```

**Rationale:** Raw ICD-9 codes have ~700 unique values per column. No model can learn meaningful patterns from 700-level categoricals with most levels having < 50 observations. The 9-category grouping from the original Strack et al. paper is clinically validated and reduces dimensionality by ~98% while preserving the clinically meaningful distinctions.

**Step 7: Collapse rare categories**

```python
# medical_specialty: group specialties with < 100 encounters into "Other"
specialty_counts = df['medical_specialty'].value_counts()
rare_specialties = specialty_counts[specialty_counts < 100].index
df['medical_specialty'] = df['medical_specialty'].replace(rare_specialties, 'Other')
df['medical_specialty'] = df['medical_specialty'].fillna('Unknown')
```

**Rationale:** `medical_specialty` has ~70 unique values, many with negligible sample sizes. One-hot encoding all of them wastes dimensionality on noise. Collapsing to a manageable set of ~10-15 meaningful specialties plus "Other" and "Unknown" preserves signal while controlling feature count.

**Step 8: Handle remaining NaN in categoricals**

```python
# race: fill NaN with 'Unknown'
df['race'] = df['race'].fillna('Unknown')

# gender: drop the tiny 'Unknown/Invalid' group or recode
df = df[df['gender'] != 'Unknown/Invalid']

# admission_type_id, discharge_disposition_id, admission_source_id:
# These are numeric codes. Convert to string before encoding so the
# pipeline treats them as categoricals, not continuous numbers.
for col in ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']:
    df[col] = df[col].astype(str)
```

**Rationale:** The ID columns look numeric but are actually categorical codes. If left as integers, the pipeline would scale them and the model would interpret "discharge_disposition_id = 6" as being 3x "discharge_disposition_id = 2", which is meaningless. Converting to string forces the pipeline to one-hot encode them correctly.

### 5.2 Feature Engineering

Seven new features, each with a specific clinical or analytical rationale:

**Feature 1: `medication_change_count`**

```python
med_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
            'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
            'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
            'miglitol', 'troglitazone', 'tolazamide', 'examide',
            'citoglipton', 'insulin', 'glyburide-metformin',
            'glipizide-metformin', 'glimepiride-pioglitazone',
            'metformin-rosiglitazone', 'metformin-pioglitazone']

df['medication_change_count'] = df[med_cols].apply(
    lambda row: ((row == 'Up') | (row == 'Down')).sum(), axis=1
)
```

**Rationale:** Captures the intensity of medication adjustment during the stay. A patient with 3 medication changes is clinically different from one with 0 -- they're either less stable or receiving more aggressive management. This single numeric feature replaces the need to individually model all 23 medication columns.

**Feature 2: `total_visits_prior_year`**

```python
df['total_visits_prior_year'] = (
    df['number_inpatient'] + df['number_outpatient'] + df['number_emergency']
)
```

**Rationale:** A composite utilization index. Patients with high total healthcare utilization in the prior year are higher-risk. Combining the three visit types into one feature captures overall "system engagement" without requiring the model to learn the sum relationship itself.

**Feature 3: `has_A1c_measured`**

```python
df['has_A1c_measured'] = (df['A1Cresult'] != 'None').astype(int)
```

**Rationale:** The Strack et al. paper found that HbA1c measurement during the encounter was independently associated with reduced readmission, regardless of the result. The act of measuring is a proxy for better diabetes management protocols. This is the single most clinically important engineered feature.

**Feature 4: `A1c_elevated`**

```python
df['A1c_elevated'] = df['A1Cresult'].isin(['>7', '>8']).astype(int)
```

**Rationale:** Distinguishes between patients with controlled vs. uncontrolled diabetes. Poorly controlled diabetes (A1c > 7) is a direct clinical risk factor for complications and readmission.

**Feature 5: `diagnosis_is_diabetes`**

```python
df['diagnosis_is_diabetes'] = (df['diag_1'] == 'Diabetes').astype(int)
```

**Rationale:** A patient admitted primarily for diabetes management may have different readmission patterns than one admitted for a heart failure exacerbation who also happens to have diabetes. This flag lets the model distinguish the two scenarios.

**Feature 6: `num_procedures_per_day`**

```python
df['num_procedures_per_day'] = df['num_procedures'] / df['time_in_hospital']
```

**Rationale:** Normalizes procedure count by length of stay. A patient with 3 procedures over 1 day received more intensive treatment than one with 3 procedures over 10 days. Raw procedure count conflates treatment intensity with length of stay.

**Feature 7: `polypharmacy`**

```python
df['polypharmacy'] = (df['num_medications'] > 15).astype(int)
```

**Rationale:** Polypharmacy (concurrent use of many medications) is a recognized clinical risk factor for drug interactions, adverse events, and readmission. The threshold of 15 is a commonly used cutpoint in geriatric medicine literature.

### 5.3 Drop Near-Zero-Variance Medication Columns

After computing `medication_change_count`, the individual medication columns can be dropped. Most of them are > 95% "No" and provide no predictive signal.

```python
# Keep only insulin and metformin as individual features (highest variance)
# Drop the rest -- their signal is captured by medication_change_count
keep_med_cols = ['insulin', 'metformin']
drop_med_cols = [c for c in med_cols if c not in keep_med_cols]
df = df.drop(columns=drop_med_cols)
```

**Rationale:** Examined in Phase 1 -- most medication columns are > 95% "No". One-hot encoding 23 x 4-level categoricals creates ~92 sparse features. The `medication_change_count` composite plus `insulin` and `metformin` (the two with meaningful variance) retains the signal in ~6 features instead of ~92.

### 5.4 Define Feature Groups

```python
# After all cleaning and engineering, define column groups for the pipeline

numeric_features = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses',
    'medication_change_count', 'total_visits_prior_year',
    'num_procedures_per_day'
]

categorical_features = [
    'race', 'gender', 'age',
    'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
    'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed',
    'diag_1', 'diag_2', 'diag_3',
    'medical_specialty', 'insulin', 'metformin'
]

binary_features = [
    'has_A1c_measured', 'A1c_elevated', 'diagnosis_is_diabetes', 'polypharmacy'
]

target = 'readmit_30'
```

### 5.5 Build the scikit-learn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

# Numeric pipeline: median imputation → standard scaling
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline: most-frequent imputation → one-hot encoding
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='if_binary'))
])

# Combine into ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', num_pipeline, numeric_features),
    ('cat', cat_pipeline, categorical_features),
    ('bin', 'passthrough', binary_features),
])
```

**Rationale for each choice:**
- `SimpleImputer(strategy='median')` for numerics: median is robust to the right-skewed distributions seen in Phase 1 (number_inpatient, number_outpatient, etc.)
- `SimpleImputer(strategy='most_frequent')` for categoricals: fills missing categories with the mode. After Phase 3 cleaning most NaNs are already handled, so this is a safety net.
- `OneHotEncoder(handle_unknown='ignore')`: if the test set contains a category not seen in training, it produces a zero vector instead of crashing.
- `drop='if_binary'`: for binary categoricals (gender, change, diabetesMed), drops one level to avoid multicollinearity in logistic regression.
- `'passthrough'` for binary features: already 0/1 integers, no transformation needed.

### 5.6 Train/Test Split

```python
from sklearn.model_selection import train_test_split

X = df[numeric_features + categorical_features + binary_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Train: {X_train.shape[0]:,} rows ({y_train.mean()*100:.1f}% positive)")
print(f"Test:  {X_test.shape[0]:,} rows ({y_test.mean()*100:.1f}% positive)")
```

**Rationale:** 80/20 split with stratification ensures the class balance is preserved in both sets. `random_state=42` for reproducibility. The test set is locked away until Phase 6 -- all tuning happens on training data via cross-validation.

### 5.7 Validation Checkpoint

Before moving to modeling, verify:

```python
# 1. Pipeline transforms without error
X_train_transformed = preprocessor.fit_transform(X_train)
print(f"Transformed shape: {X_train_transformed.shape}")

# 2. No NaN in output
assert not np.any(np.isnan(X_train_transformed)), "NaN found in transformed output"

# 3. Class balance preserved
print(f"Train positive rate: {y_train.mean()*100:.1f}%")
print(f"Test positive rate:  {y_test.mean()*100:.1f}%")

# 4. Feature count
feature_names = preprocessor.get_feature_names_out()
print(f"Feature count after encoding: {len(feature_names)}")
```

### 5.8 Deliverables

- [ ] All 8 cleaning steps applied in order
- [ ] 7 engineered features created with documented rationale
- [ ] Near-zero-variance columns dropped
- [ ] sklearn ColumnTransformer + Pipeline assembled and tested
- [ ] Train/test split created (80/20, stratified)
- [ ] Pipeline transforms without errors
- [ ] Test set locked -- not touched until Phase 6

---

## 6. Phase 4 -- Baseline Model Training

**Goal:** Train four models with default hyperparameters, establish baseline performance, and identify which models warrant further tuning.

### 6.1 Model Definitions

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        solver='lbfgs'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=50,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    ),
    'SVM': SVC(
        kernel='rbf',
        class_weight='balanced',
        probability=True,  # Required for ROC-AUC
        random_state=42
    ),
}
```

**Rationale for each model:**

| Model | Why | Strength | Weakness |
|-------|-----|----------|----------|
| Logistic Regression | Interpretable baseline. Coefficients → odds ratios. | Clinical explainability. Fast. Regularization handles correlated features. | Assumes linear decision boundary. |
| Random Forest | Non-linear interactions without feature crosses. | Built-in feature importance. Robust to outliers. Hard to overfit with depth limits. | Slower inference. Less interpretable than LR. |
| XGBoost | State-of-the-art for tabular data. Published benchmark used XGBoost. | Handles imbalance via `scale_pos_weight`. Built-in regularization. Best expected AUC. | Requires more tuning. Can overfit with too many trees. |
| SVM | Different learning paradigm (margin-based). Adds diversity. | Effective in high-dimensional space after OHE. Kernel trick for non-linearity. | Slow on large datasets. Hyperparameters are sensitive. |

### 6.2 Cross-Validation

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}
for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    scores = cross_val_score(pipeline, X_train, y_train,
                             cv=cv, scoring='roc_auc', n_jobs=-1)

    results[name] = {
        'mean_auc': scores.mean(),
        'std_auc': scores.std(),
        'scores': scores
    }

    print(f"{name:<25} AUC: {scores.mean():.4f} ± {scores.std():.4f}")
```

**Rationale:** 5-fold stratified CV ensures each fold preserves the class balance. `scoring='roc_auc'` because accuracy is misleading with 89/11 imbalance. This gives us a fair comparison across all four models before any tuning.

### 6.3 SMOTE Comparison (Optional)

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Only apply SMOTE within CV folds to prevent data leakage
smote_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', XGBClassifier(...))
])
```

**Rationale:** SMOTE must be inside the CV loop, not applied before splitting. Using `imblearn.pipeline.Pipeline` (not sklearn's) ensures SMOTE only transforms training folds, never the validation fold. Compare SMOTE results against `class_weight='balanced'` results to determine which imbalance strategy works best.

### 6.4 Deliverables

- [ ] All 4 models trained with default hyperparameters
- [ ] 5-fold CV AUC scores logged for each model
- [ ] Baseline comparison table/chart
- [ ] SMOTE vs. class_weight comparison (at least for best model)
- [ ] Best baseline model identified

---

## 7. Phase 5 -- Hyperparameter Tuning

**Goal:** Optimize the top 2-3 models from Phase 4 through systematic hyperparameter search.

### 7.1 Tuning Strategy

Two-stage approach:
1. **RandomizedSearchCV** (50-100 iterations): broad sweep to find the right neighborhood
2. **GridSearchCV**: narrow, exhaustive search around the best region from Stage 1

### 7.2 Parameter Grids

**Logistic Regression:**

```python
lr_params = {
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__solver': ['saga'],  # supports both L1 and L2
}
```

**Random Forest:**

```python
rf_params = {
    'classifier__n_estimators': [100, 200, 300, 500],
    'classifier__max_depth': [5, 10, 15, 20, None],
    'classifier__min_samples_leaf': [20, 50, 100],
    'classifier__max_features': ['sqrt', 'log2', 0.3],
}
```

**XGBoost:**

```python
xgb_params = {
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'classifier__max_depth': [3, 5, 7, 9],
    'classifier__n_estimators': [100, 200, 300, 500],
    'classifier__subsample': [0.7, 0.8, 0.9, 1.0],
    'classifier__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'classifier__min_child_weight': [1, 3, 5, 7],
}
```

**SVM:**

```python
svm_params = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__kernel': ['rbf', 'poly'],
    'classifier__gamma': ['scale', 'auto', 0.01, 0.001],
}
```

### 7.3 Execution

```python
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# Stage 1: Broad sweep
random_search = RandomizedSearchCV(
    pipeline, param_distributions=xgb_params,
    n_iter=80, cv=cv, scoring='roc_auc',
    random_state=42, n_jobs=-1, verbose=1
)
random_search.fit(X_train, y_train)

# Extract best region
best_params = random_search.best_params_
print(f"Best RandomizedSearch AUC: {random_search.best_score_:.4f}")

# Stage 2: Narrow grid around best
narrow_grid = {
    'classifier__learning_rate': [best_lr * 0.5, best_lr, best_lr * 2],
    'classifier__max_depth': [best_depth - 1, best_depth, best_depth + 1],
    # ... etc
}

grid_search = GridSearchCV(
    pipeline, param_grid=narrow_grid,
    cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

print(f"Best GridSearch AUC: {grid_search.best_score_:.4f}")
```

**Rationale for two-stage:** RandomizedSearchCV with 80 iterations over a large space is faster than exhaustive grid search (which would be 4 x 4 x 4 x 4 x 4 x 4 = 4,096 combinations for XGBoost alone). The randomized stage identifies the right region; the grid stage fine-tunes within it.

### 7.4 Document Iterative Improvement

Create a comparison table showing how each round of tuning improved (or didn't improve) the model:

```python
tuning_log = pd.DataFrame([
    {'Model': 'XGBoost', 'Stage': 'Default', 'AUC': 0.XXX},
    {'Model': 'XGBoost', 'Stage': 'RandomizedSearch', 'AUC': 0.XXX},
    {'Model': 'XGBoost', 'Stage': 'GridSearch', 'AUC': 0.XXX},
    # ... etc for each model
])
```

### 7.5 Deliverables

- [ ] RandomizedSearchCV completed for top 2-3 models
- [ ] GridSearchCV completed for the best model
- [ ] Tuning log documenting iterative improvement
- [ ] Best hyperparameters recorded
- [ ] Final model selected based on CV ROC-AUC

---

## 8. Phase 6 -- Final Evaluation and Interpretation

**Goal:** Unlock the test set, evaluate the final model, explain its predictions with SHAP, and translate results into business impact.

### 8.1 Test-Set Evaluation

```python
# Fit the final pipeline on full training data
final_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', best_model)
])
final_pipeline.fit(X_train, y_train)

# Predict on held-out test set
y_pred = final_pipeline.predict(X_test)
y_prob = final_pipeline.predict_proba(X_test)[:, 1]
```

### 8.2 Metrics Suite

```python
from sklearn.metrics import (classification_report, roc_auc_score,
                             roc_curve, precision_recall_curve,
                             confusion_matrix, ConfusionMatrixDisplay)

# ROC-AUC
auc = roc_auc_score(y_test, y_prob)

# Classification report at default 0.5 threshold
print(classification_report(y_test, y_pred, target_names=['Not Readmit', 'Readmit <30']))

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ...)

# ROC curve overlay (all 4 models)
# Precision-recall curve (more informative for imbalanced classes)
```

### 8.3 Threshold Optimization

The default 0.5 threshold is rarely optimal for imbalanced classes. Find the threshold that maximizes the business-relevant metric.

```python
# Find threshold that maximizes F1
from sklearn.metrics import f1_score

thresholds = np.arange(0.1, 0.6, 0.01)
f1_scores = [f1_score(y_test, (y_prob >= t).astype(int)) for t in thresholds]
best_threshold = thresholds[np.argmax(f1_scores)]

print(f"Optimal threshold: {best_threshold:.2f}")
print(f"F1 at optimal threshold: {max(f1_scores):.4f}")
```

**Rationale:** With ~11% positive rate, the default 0.5 threshold biases toward the majority class. Lowering the threshold captures more true positives (higher recall) at the cost of more false positives (lower precision). The optimal point depends on the relative cost of missing a readmission vs. the cost of an unnecessary intervention.

### 8.4 SHAP Analysis

```python
import shap

# For tree-based models (RF, XGBoost):
explainer = shap.TreeExplainer(final_pipeline.named_steps['classifier'])

# Get transformed features for SHAP
X_test_transformed = final_pipeline.named_steps['preprocessor'].transform(X_test)
feature_names = final_pipeline.named_steps['preprocessor'].get_feature_names_out()

shap_values = explainer.shap_values(X_test_transformed)

# Global feature importance: summary plot
shap.summary_plot(shap_values, X_test_transformed,
                  feature_names=feature_names, max_display=20)

# Individual patient explanation: force plot
shap.force_plot(explainer.expected_value, shap_values[0],
                X_test_transformed[0], feature_names=feature_names)
```

**Produce three SHAP visualizations:**

1. **Summary bar plot** -- Top 20 features by mean |SHAP value|. This is the global feature importance ranking.
2. **Summary dot plot** -- Same features but shows the direction and magnitude of each feature's impact. Red dots (high feature value) pulling right (higher readmission probability) vs. left.
3. **Force plot for 2-3 individual patients** -- One high-risk, one low-risk, one borderline. Demonstrates how the model makes decisions at the patient level. This is what clinicians need to see.

### 8.5 Clinical Risk Factor Summary

Translate the top 10 SHAP features into plain-English statements:

```python
# Example (numbers will come from actual SHAP values):
risk_factors = [
    "Patients with 2+ prior inpatient visits in the past year were 2.1x more likely to be readmitted",
    "Patients whose HbA1c was NOT measured during the stay had a 1.4x higher readmission rate",
    "Discharge to home health (vs. home self-care) reduced readmission risk by 22%",
    "Each additional inpatient day was associated with a 5% increase in readmission probability",
    "Patients on insulin with dosage changes were 1.3x more likely to be readmitted",
    # ... etc
]
```

**Rationale:** This is the deliverable the stakeholders actually care about. The CMO doesn't read SHAP plots -- they read sentences. These statements must be hedged with "associated with" language (not causal claims) since this is observational data.

### 8.6 Business Impact Simulation

```python
# Hypothetical hospital: 10,000 Medicare discharges per year
# ~11% readmitted within 30 days = ~1,100 readmissions
# Average HRRP penalty: 0.74% of Medicare revenue
# Average Medicare revenue per discharge: ~$13,000
# Annual penalty exposure: 10,000 x $13,000 x 0.0074 = ~$962,000

# At optimal threshold, model catches X% of readmissions (recall)
# and flags Y total patients (recall + false positives)
# Intervention cost per flagged patient: ~$500 (nurse call + med reconciliation)

# Net benefit = (TP x penalty_saved_per_readmission) - (FP x $500) - (FN x penalty_per_miss)
```

Produce a table showing net benefit at 3-5 different probability thresholds, so the CFO can choose the operating point that matches their risk tolerance.

### 8.7 Deliverables

- [ ] Test-set metrics: ROC-AUC, precision, recall, F1, confusion matrix
- [ ] ROC curve overlay for all 4 models
- [ ] Precision-recall curve for final model
- [ ] Threshold optimization with business justification
- [ ] SHAP summary plot (global importance)
- [ ] SHAP dot plot (directional effects)
- [ ] SHAP force plots for 2-3 individual patients
- [ ] Clinical risk factor summary (plain English)
- [ ] Business impact simulation table at multiple thresholds

---

## 9. Phase 7 -- Documentation and Communication

**Goal:** Clean up the notebook, build the executive presentation, record the video, and push to GitHub.

### 9.1 Notebook Cleanup

- Every code cell has a comment or markdown cell above it explaining what and why
- Remove dead code, debugging cells, and scratch work
- Confirm the notebook runs top-to-bottom without errors: `Kernel → Restart & Run All`
- All figures saved to `./figures/`
- All model artifacts saved to `./models/` (e.g., `joblib.dump(final_pipeline, 'models/final_pipeline.joblib')`)

### 9.2 Executive Presentation

Target audience: hospital administrators who make resource allocation decisions. They understand readmissions and HRRP but not SHAP or AUC.

**Slide structure (10 minutes):**

| Slide | Content | Time |
|-------|---------|------|
| 1 | Title + business problem (HRRP penalties) | 0:45 |
| 2 | Dataset overview + approach (CRISP-DM) | 0:45 |
| 3 | Key EDA findings (prior visits, HbA1c measurement) | 1:30 |
| 4 | Model comparison results (bar chart of AUC scores) | 1:00 |
| 5 | Final model performance (confusion matrix, threshold) | 1:00 |
| 6 | Top risk factors (plain English, from SHAP) | 1:30 |
| 7 | Business impact simulation (cost/benefit table) | 1:30 |
| 8 | Recommendations + limitations + next steps | 1:00 |
| 9 | Notebook walkthrough (screen share, 60 seconds) | 1:00 |

**Key framing:** Lead with the money. "$560 million in annual HRRP penalties" → "our model identifies X% of readmissions before discharge" → "targeted intervention on flagged patients costs $Y but saves $Z" → "net benefit: positive."

### 9.3 GitHub Repository

Final structure:

```
Diabetes_Readmission_Capstone/
├── Data/                          # gitignored
│   ├── diabetic_data.csv
│   └── IDs_mapping.csv
├── notebooks/                     # Exploratory work (optional)
├── figures/                       # All saved PNGs
│   ├── target_distribution.png
│   ├── missing_data.png
│   ├── shap_summary.png
│   └── ...
├── models/                        # gitignored
│   └── final_pipeline.joblib
├── outputs/
│   ├── executive_presentation.pptx
│   └── Project2_Pitch.pdf
├── DS_Capstone_Readmission.ipynb  # Main deliverable
├── .gitignore
└── README.md
```

### 9.4 Deliverables

- [ ] Notebook runs Restart & Run All without errors
- [ ] All code cells commented
- [ ] Presentation slides built
- [ ] Video recorded (10 minutes)
- [ ] GitHub repo pushed and clean
- [ ] README updated with final results

---

## 10. Known Challenges and Mitigations

| Challenge | Impact | Mitigation |
|-----------|--------|------------|
| Class imbalance (~11% positive) | Model biases toward majority class, low recall | Compare SMOTE, class_weight, scale_pos_weight, and threshold tuning. Evaluate on recall and F1, not accuracy. |
| High-cardinality ICD-9 codes | One-hot encoding creates 2,100+ sparse features | Group into 9 clinical categories per Strack et al. methodology before encoding. |
| Multiple encounters per patient | Data leakage if same patient in train and test | Deduplicate by patient_nbr before splitting. Keep longest encounter per patient. |
| Dataset age (1999-2008) | Clinical practices have evolved | Acknowledge as limitation. Frame as pipeline development + benchmarking. The HRRP business case is current even if the data is not. |
| Modest expected AUC (0.65-0.67) | Stakeholders may question model utility | Frame realistically: readmission prediction is hard because social determinants (housing, transportation, support network) aren't in the data. A 0.66 AUC model that costs $500/intervention and saves $13,000/avoided penalty is still net-positive. |
| SVM training time | SVM on ~70K x 100+ features is slow | If SVM takes > 30 minutes, subsample training data for SVM only or reduce the parameter grid. Don't let one model block the entire timeline. |
| SHAP on large datasets | TreeExplainer can be memory-intensive | Use `shap.TreeExplainer` with `model_output='raw'`. If memory is an issue, compute SHAP on a random 5K subsample of the test set. |

---

## 11. Timeline Summary

| Week | Phase | Key Output | Gate |
|------|-------|-----------|------|
| 1 | Profiling (Phase 1) | Notebook cells 0-35, 4 figures | Dataset loads, target confirmed ~11% |
| 1 | EDA (Phase 2) | 8+ additional visualizations, statistical tests | Key hypotheses documented |
| 1 | Cleaning + Pipeline (Phase 3) | 8 cleaning steps, 7 features, pipeline tested | Pipeline transforms without errors, test set locked |
| 2 | Baseline Models (Phase 4) | 4 models x 5-fold CV | All baselines logged, best > 0.5 AUC |
| 2 | Tuning (Phase 5) | RandomizedSearchCV + GridSearchCV | Best model selected, CV AUC maximized |
| 2 | Final Eval + SHAP (Phase 6) | Test-set metrics, SHAP plots, cost simulation | All Phase 6 deliverables complete |
| 3 | Documentation (Phase 7) | Presentation, video, GitHub push | Submitted |
