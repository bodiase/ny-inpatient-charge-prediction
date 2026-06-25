# Predicting Hospital Inpatient Charges
### New York State SPARCS 2021 | Supervised Machine Learning Pipeline

Supervised regression pipeline predicting hospital inpatient charges across 50,000 New York State discharge records. Seven models compared — from linear baseline to tuned XGBoost — with SHAP-based interpretability analysis identifying the primary drivers of charge variation.

**Best result:** Tuned XGBoost — R² = 0.814, MAE = $18,011, 31% RMSE reduction vs. linear baseline

---

## Business Problem

Hospital inpatient charges in New York State vary by more than 10× across facilities for patients with similar diagnoses and clinical profiles. This variability creates financial uncertainty for hospitals, insurers, policymakers, and patients.

**Core question:** Can patient demographics, clinical characteristics, and hospital-level features predict inpatient charges with sufficient accuracy to support benchmarking, anomaly detection, and financial planning?

This project builds a data-driven baseline — a model that estimates expected charges from observable features, enabling stakeholders to identify cases where actual charges deviate significantly from predicted norms.

**Stakeholder applications:**
- **Hospitals:** Benchmark actual billing against statewide norms; flag systematic deviations for revenue integrity review
- **Insurers:** Calibrate reimbursement rates and support actuarial modeling
- **Policymakers:** Monitor regional and demographic pricing disparities
- **Patients:** Benefit indirectly from increased pricing transparency

> **Scope note:** This model predicts billed charges, not actual costs or payments received. Predictions should be interpreted as billing benchmarks, not cost estimates.

---

## Dataset

| Attribute | Detail |
|---|---|
| Source | New York State Department of Health — Hospital Inpatient Discharges (SPARCS De-Identified) 2021 |
| Records | 50,000 inpatient hospitalizations (sampled from full statewide census) |
| Features | 33 variables across demographics, clinical classification, hospital characteristics, and visit attributes |
| Target | `total_charges` — total amount billed per inpatient stay |

**Key feature categories:**

| Category | Fields |
|---|---|
| Hospital & Facility | `hospital_service_area`, `hospital_county`, `facility_name` |
| Patient Demographics | `age_group`, `gender`, `race`, `ethnicity`, `zip_code_3_digits` |
| Visit Characteristics | `length_of_stay`, `type_of_admission`, `patient_disposition`, `emergency_department_indicator` |
| Clinical Classification | `ccsr_diagnosis_code`, `apr_drg_code`, `apr_severity_of_illness_code`, `apr_risk_of_mortality` |
| Target | `total_charges` (prediction target) — `total_costs` excluded to prevent target leakage |

A 500-row stratified sample is included in `/data/` for local reproducibility. The full dataset is publicly available from the [New York State SPARCS data portal](https://health.data.ny.gov/Health/Hospital-Inpatient-Discharges-SPARCS-De-Identified/gnzp-ekau).

---

## Tools and Environment

| Category | Tools |
|---|---|
| Language | Python 3 |
| Data manipulation | pandas, NumPy |
| ML pipeline | scikit-learn (Pipeline, ColumnTransformer, TransformedTargetRegressor) |
| Models | scikit-learn (Linear, Ridge, Lasso, Decision Tree, Random Forest), XGBoost |
| Interpretability | SHAP (TreeExplainer) |
| Visualization | matplotlib, seaborn |
| Environment | Google Colab / Jupyter |

---

## Methodology

### Preprocessing Pipeline

The dataset required a carefully differentiated encoding strategy given its mix of numeric, ordinal, and high-cardinality categorical features:

| Feature Type | Treatment | Rationale |
|---|---|---|
| `length_of_stay` (numeric) | Median imputation | Handles 30 missing values from the `120+` cap |
| Severity and mortality codes (ordinal) | OrdinalEncoder | Preserves natural rank ordering (Minor < Moderate < Major < Extreme) |
| Diagnosis, procedure, DRG codes (high-cardinality) | Custom FrequencyEncoder | Avoids sparse one-hot matrix; leak-free compact representation |
| Demographics and admission type (low-cardinality) | OneHotEncoder | Standard treatment for nominal categoricals |
| `total_charges` (target) | Log transformation + Winsorization at 99th percentile | Right-skew requires transformation; top 1% of cases ($556K+) would dominate loss function |

All transformations are fitted on training data only and applied to the test set via scikit-learn's `ColumnTransformer`, preventing data leakage.

### Modeling Strategy

Seven models were trained and evaluated, progressing from a linear baseline to a tuned gradient boosting model. The progression was designed to answer a specific question: is the relationship between features and charges fundamentally linear or nonlinear?

80/20 train/test split (`random_state=42`). All metrics reported on the held-out test set.

---

## Model Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | \$33,061 | \$54,093 | 0.606 |
| Ridge Regression | \$33,061 | \$54,093 | 0.606 |
| Lasso Regression | \$32,985 | \$54,086 | 0.606 |
| Decision Tree | \$24,555 | \$48,421 | 0.684 |
| Random Forest (default) | \$20,010 | \$41,047 | 0.773 |
| XGBoost (default) | \$18,675 | \$38,455 | 0.801 |
| **XGBoost (tuned)** | **\$18,011** | **\$37,166** | **0.814** |

**Reading the progression:**

The plateau across all three linear models (R²=0.606) is not a modeling failure — it confirms that hospital charge variation has meaningful nonlinear structure that regularization alone cannot address. The decision tree's improvement to 0.684 confirms nonlinearity exists; Random Forest's jump to 0.773 shows that variance reduction through bagging captures it more stably; XGBoost's final push to 0.814 reflects gradient boosting's ability to focus on the hardest-to-predict cases.

Tuning XGBoost via `RandomizedSearchCV` (30 iterations, 3-fold CV) produced a modest additional gain — 3.4% RMSE reduction from default. The cross-validated RMSE ($39,116) being close to the test RMSE ($37,166) confirms the model generalizes well without overfitting.

**The residual unexplained variance (~19%) reflects structural data limitations:** hospital-specific charge master pricing, payer-specific negotiation history, and institutional overhead are real determinants of what hospitals bill that are not present in administrative discharge data. No model trained on this feature set can close this gap — and identifying that ceiling is itself a useful analytical finding.

---

## Key Findings

**1. Length of stay is by far the dominant cost driver.**
Lasso coefficient analysis and SHAP global importance agree: `length_of_stay` has substantially larger impact than any other feature. Every additional day in the hospital corresponds to a nonlinear increase in billed charges — the relationship is steeper at shorter stays and levels off at longer stays.

**2. Surgical classification is a strong binary signal.**
Whether a stay is classified as medical vs. surgical is the second-most influential predictor in the Lasso analysis. Surgical cases carry substantially higher charges, reflecting OR time, procedural costs, and specialist involvement.

**3. Geography matters as much as clinical severity.**
`hospital_county` and `apr_severity_of_illness_code` rank third and fourth in both Lasso and SHAP importance. The same diagnosis in Manhattan may be billed very differently than the same diagnosis in a rural upstate county — this geographic component reflects facility-level pricing strategy that cannot be fully explained by clinical factors alone.

**4. The relationship between charges and predictors is nonlinear.**
The 13-point R² gap between linear models and the best tree-based model is driven by interaction effects: the cost of a given length of stay depends on why the patient is staying. SHAP's dependence plot for LOS confirms this — the interaction coloring shows severity modulates the LOS effect significantly.

**5. Preprocessing design matters as much as model selection.**
Without log transformation of the target and Winsorization of the top 1%, the linear-to-XGBoost performance gap narrows substantially — the preprocessing pipeline is not incidental to the results, it is a core part of them.

---

## SHAP Interpretability Analysis

*This section is an independent portfolio extension completed after the original coursework.*

Prediction accuracy alone is insufficient for a healthcare analytics application. SHAP (SHapley Additive exPlanations) was applied to the tuned XGBoost model to explain both global feature importance and individual predictions.

**Four analyses were performed:**

**Global importance (mean |SHAP|):** Confirms `length_of_stay` dominates, with severity, DRG code, county, and MDC code forming the second tier. This ranking is consistent with the Lasso coefficient analysis, validating that both the linear and nonlinear models are capturing the same underlying signal.

**Beeswarm plot (direction and magnitude):** Reveals that all top features have consistent, interpretable directional effects. High LOS pushes predictions up; low severity pushes them down; surgical classification creates a clean two-cluster pattern. The beeswarm also identifies features with conditional effects — where impact depends on the values of other variables.

**Waterfall plots (individual explanations):** Decompose single predictions into additive feature contributions. The high-cost case ($772,665 predicted) and low-cost case ($1,396 predicted) illustrate how the model's reasoning can be communicated to a hospital finance reviewer for a specific patient — the kind of explanation needed for anomaly detection use cases.

**Dependence plot (LOS × interaction):** Shows that the SHAP value for `length_of_stay` increases nonlinearly and is modulated by clinical severity — the cost impact of a 10-day stay differs meaningfully between Minor and Extreme severity cases. This interaction is what prevents linear models from capturing the full signal.

Together, these analyses show that the model's R²=0.814 reflects real clinical and economic signal — not noise — and that the learned relationships are defensible and interpretable to domain stakeholders.

---

## Limitations

- **Unobserved pricing factors:** Hospital charge master pricing, payer-specific contract rates, and institutional billing strategy are not present in SPARCS data. These are real determinants of billed charges and represent a structural ceiling on predictive performance that additional modeling cannot close.
- **Extreme-cost cases:** Winsorization at the 99th percentile excludes the most expensive cases from training. The model systematically underpredicts catastrophic cases (major trauma, extended ICU stays). This is a deliberate tradeoff to improve accuracy across the 99% of cases within the cap.
- **Temporal scope:** Cross-sectional 2021 data. Hospital pricing changes over time; model may not reflect current charge patterns.
- **Geographic scope:** New York State only. Findings reflect NY's specific healthcare market structure and may not generalize to other states.
- **Prediction vs. causation:** This model identifies features that predict charges; it does not establish that these features cause higher charges. Length of stay, for example, is partially a symptom of clinical severity rather than an independent cause of cost.

---

## Repository Structure

```
ny-inpatient-charge-prediction/
│
├── README.md
│
├── notebook/
│   └── inpatient_charge_prediction.ipynb
│
├── data/
│   ├── sample_sparcs_2021_500rows.csv   ← 500-row stratified sample
│   └── DATA_README.md                   ← data source notes
│
├── presentation/
│   └── predicting_hospital_inpatient_charges.pdf
│
└── .gitignore
```

---

## How to Run

**Option 1 — Google Colab (recommended):**

1. Open `notebook/inpatient_charge_prediction.ipynb` in Google Colab
2. Run all cells — data loads automatically from Google Drive
3. SHAP installation is handled in the notebook if not already present

**Option 2 — Local:**

```bash
git clone https://github.com/bodiase/ny-inpatient-charge-prediction.git
cd ny-inpatient-charge-prediction
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap
```

Then open the notebook, uncomment the `LOCAL_PATH` line in the data loading cell, and set it to the path of your local SPARCS CSV file. A 500-row sample is available in `/data/` if you want to verify the pipeline runs correctly before loading the full dataset.

**Runtime notes:**
- Full pipeline (all 7 models + SHAP): approximately 15–25 minutes on a standard Colab CPU
- `RandomizedSearchCV` for XGBoost (30 iterations × 3 folds) accounts for the majority of runtime
- SHAP computation on 10,000 test cases: approximately 1–3 minutes

---

## Skills Demonstrated

| Skill | Where Applied |
|---|---|
| **Supervised ML pipeline design** | scikit-learn Pipeline + ColumnTransformer with differentiated encoding per feature type |
| **Regression modeling** | Linear, Ridge, Lasso, Decision Tree, Random Forest, XGBoost — full model family comparison |
| **Hyperparameter tuning** | RandomizedSearchCV with cross-validation on XGBoost |
| **Feature engineering** | Custom FrequencyEncoder for high-cardinality categoricals; ordinal encoding for severity codes |
| **Target transformation** | TransformedTargetRegressor with log1p/expm1 for skewed financial target |
| **Model interpretability** | SHAP TreeExplainer — global importance, beeswarm, waterfall, and dependence plots |
| **EDA and data quality** | Distribution analysis, missingness handling, target leakage identification, Winsorization |
| **Business communication** | Findings framed for hospital finance, insurance, and policy stakeholder audiences |
| **Analytical honesty** | Explicit performance ceiling analysis; limitations documented at structural level |

---

## Background

This project was developed as part of BA810 Supervised Machine Learning at Boston University's Questrom School of Business (Fall 2025). The SHAP interpretability section is an independent extension completed after the original coursework.

**Original team:** Yangze Li, Yanlun Li, Bill Odiase, Ansh Gupta  
**Independent extension:** Bill Odiase
