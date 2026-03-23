# Predicting 30-Day Readmission for Diabetic Inpatients

**Data Science Capstone -- End-to-End ML Pipeline**

## Results

| Metric | Value |
|--------|-------|
| Model | XGBoost (tuned) |
| Test AUC | 0.693 |
| Test Recall | 0.621 |
| Business Impact | ~$4.6M/year net benefit at threshold 0.30 |

Both success targets met: AUC > 0.65 and Recall > 0.50 on held-out test data.

## Business Problem

Under the CMS Hospital Readmissions Reduction Program (HRRP), hospitals face up to a 3% reduction in all Medicare payments for excess 30-day readmissions. This project builds a machine learning pipeline to predict which diabetic patients are most likely to be readmitted within 30 days of discharge, enabling hospitals to target transitional care resources at the highest-risk patients.

## Dataset

- **Source:** [UCI Machine Learning Repository (ID 296)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
- **Size:** 101,766 inpatient encounters x 50 features
- **Span:** 10 years (1999-2008) across 130 US hospitals
- **Target:** Readmitted within 30 days (binary, ~11% positive rate)
- **After cleaning:** ~70,000 unique patients (deduplicated, expired removed)

### Data Setup

Download `diabetic_data.csv` and `IDs_mapping.csv` from the UCI repository and place them in the `Data/` directory.

```bash
pip install ucimlrepo
python -c "
from ucimlrepo import fetch_ucirepo
ds = fetch_ucirepo(id=296)
ds.data.original.to_csv('Data/diabetic_data.csv', index=False)
print(f'Saved {len(ds.data.original)} rows')
"
```

## Project Structure

```
Diabetes_Readmission_Capstone/
├── Data/                          # Raw dataset (gitignored)
│   ├── diabetic_data.csv
│   └── IDs_mapping.csv
├── figures/                       # 27 saved visualizations
├── models/                        # Saved model artifacts (.joblib)
│   └── final_model.joblib         # Final tuned XGBoost pipeline
├── notebooks/                     # Scratch/exploratory work
├── pitch/                         # Peer review documents
│   ├── Project2_Pitch.pdf         # Project pitch deck
│   ├── Project2_Developer_Guide.md
│   ├── Checkpoint_Walkthrough_Phase1-3.pdf
│   └── Checkpoint_Walkthrough_Phase4-7.pdf
├── DS_Capstone_Readmission.ipynb  # Main deliverable (216 cells, 7 phases)
├── requirements.txt               # Python dependencies
├── .gitignore
└── README.md
```

## Methodology

CRISP-DM lifecycle with scikit-learn Pipeline:

1. **Data Profiling** -- Schema validation, target distribution, missing data audit
2. **EDA** -- Correlations, categorical deep-dives, 5 statistical tests, feature interactions
3. **Cleaning & Pipeline** -- 8 cleaning steps, 7 engineered features, ColumnTransformer pipeline
4. **Baseline Models** -- Logistic Regression, Random Forest, XGBoost, SVM with 5-fold stratified CV
5. **Hyperparameter Tuning** -- 2-stage (RandomizedSearchCV + GridSearchCV) for top 3 models
6. **Final Evaluation** -- Test-set unlock, SHAP analysis, threshold optimization, business impact
7. **Documentation** -- Limitations, references, reproducibility notes

## Key Findings (SHAP)

Top 3 risk factors for 30-day readmission:

1. **Length of stay** -- Shorter stays associated with higher readmission risk
2. **Prior inpatient visits** -- Each additional prior hospitalization substantially increases risk
3. **Discharge disposition** -- Discharge to SNF/rehab associated with higher risk vs. home

## Peer Review

Project pitch and checkpoint walkthrough documents are available in the [`pitch/`](pitch/) folder, including the project pitch deck (`Project2_Pitch.pdf`) and phase-by-phase walkthrough presentations.

## Tools

Python -- Pandas -- NumPy -- scikit-learn -- XGBoost -- SHAP -- Matplotlib -- Seaborn -- Joblib

## Citation

Strack, B., DeShazo, J., Gennings, C., Olmo, J.L., Ventura, S., Cios, K.J., and Clore, J.N. (2014). "Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records." *BioMed Research International*, 2014, Article ID 781670. doi:10.1155/2014/781670
