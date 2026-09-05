# Hospital Readmission Risk Predictor — Engineering Spec

> **Purpose of this document:** Complete build spec for a coding agent. Every decision is pre-made. Read top-to-bottom and build in the order written. Do not deviate from the stack, folder structure, or decisions described unless a hard technical blocker forces it — in that case, note the deviation in a comment.

---

## 0. What We Are Building

An end-to-end, production-style ML pipeline that predicts whether a diabetic patient will be readmitted to hospital within 30 days of discharge, using the publicly available UCI Diabetes 130-US Hospitals dataset (100K records, 50 features).

The system has five distinct layers:

1. **Data ingestion layer** — download raw CSV, clean it, load into PostgreSQL
2. **SQL analytics layer** — cohort analysis via SQL (readmission rates by diagnosis, payer, LOS)
3. **ML pipeline** — feature engineering → XGBoost baseline → LightGBM final model, with SMOTE and class-weight handling
4. **Evaluation layer** — AUROC, precision-recall curves, SHAP waterfall + beeswarm plots, saved as PNG
5. **Serving layer** — async FastAPI `/predict` endpoint, fully Dockerized with Docker Compose

Everything is in one GitHub repo. The README must be thorough (see Section 8).

---

## 1. Dataset

**Name:** Diabetes 130-US Hospitals for Years 1999-2008  
**Source:** UCI Machine Learning Repository  
**Direct download URL:** `https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip`  
**Fallback (Kaggle mirror):** Search "diabetes 130 us hospitals kaggle" if UCI is down.

**Files inside the zip:**
- `diabetic_data.csv` — main data file (101,766 rows × 50 columns)
- `IDs_mapping.csv` — maps coded values (admission_type_id, discharge_disposition_id, admission_source_id) to their human-readable labels

**Target variable:** Column `readmitted`  
Raw values: `"<30"` (readmitted within 30 days), `">30"` (readmitted after 30 days), `"NO"` (not readmitted)  
**Binary encoding:** `1` if `readmitted == "<30"`, else `0`  
This gives severe class imbalance: ~11% positive class. This is intentional and must be handled explicitly.

**Key columns to know:**
| Column | Type | Notes |
|---|---|---|
| `encounter_id` | int | Unique per visit — drop before training |
| `patient_nbr` | int | Patient ID — one patient can have multiple visits; deduplicate |
| `race` | categorical | Has `"?"` values → treat as `"Unknown"` |
| `gender` | categorical | One value is `"Unknown/Invalid"` → drop those rows |
| `age` | ordinal categorical | Values like `"[0-10)"`, `"[10-20)"` etc. → encode as integers 0–9 |
| `weight` | categorical | ~97% missing → **drop this column entirely** |
| `payer_code` | categorical | ~40% missing → keep, fill with `"Unknown"` |
| `medical_specialty` | categorical | ~50% missing → keep, fill with `"Unknown"` |
| `diag_1`, `diag_2`, `diag_3` | string ICD-9 codes | Map to 9 disease categories (see Section 3.2) |
| `number_inpatient` | int | Strong predictor — keep |
| `number_emergency` | int | Keep |
| `number_outpatient` | int | Keep |
| `time_in_hospital` | int | Days in hospital (1–14) |
| `num_lab_procedures` | int | Keep |
| `num_procedures` | int | Keep |
| `num_medications` | int | Keep |
| `number_diagnoses` | int | Keep |
| `A1Cresult` | categorical | `">8"`, `">7"`, `"Norm"`, `"None"` |
| `change` | binary | Whether diabetes medication was changed |
| `diabetesMed` | binary | Whether diabetes medication was prescribed |
| 24 medication columns (`metformin`, `insulin`, etc.) | categorical | Values: `"No"`, `"Steady"`, `"Up"`, `"Down"` |

---

## 2. Project Structure

```
readmission-risk-predictor/
│
├── data/
│   ├── raw/                     # Downloaded CSVs land here (gitignored)
│   └── processed/               # Cleaned parquet files (gitignored)
│
├── sql/
│   ├── 01_create_tables.sql     # DDL for PostgreSQL tables
│   ├── 02_cohort_analysis.sql   # Analytical queries (readmission rates by group)
│   └── 03_feature_views.sql     # SQL views used as features
│
├── src/
│   ├── __init__.py
│   ├── ingest.py                # Download, unzip, clean raw CSV
│   ├── db.py                    # PostgreSQL connection + load functions
│   ├── features.py              # Feature engineering (pure Python/Pandas)
│   ├── train.py                 # Training: XGBoost + LightGBM + SMOTE
│   ├── evaluate.py              # AUROC, PR curve, confusion matrix, SHAP plots
│   └── predict.py               # Inference logic (used by API)
│
├── api/
│   ├── main.py                  # FastAPI app
│   ├── schema.py                # Pydantic request/response models
│   └── model_loader.py          # Load saved model artifact on startup
│
├── notebooks/
│   └── eda_and_results.ipynb    # EDA + final evaluation plots (for README screenshots)
│
├── models/
│   └── .gitkeep                 # Saved model artifacts go here (.gitignored except .gitkeep)
│
├── plots/
│   └── .gitkeep                 # Saved evaluation PNGs go here
│
├── tests/
│   ├── test_features.py         # Unit tests for feature engineering
│   └── test_api.py              # FastAPI endpoint tests with httpx
│
├── Dockerfile                   # API service
├── docker-compose.yml           # API + PostgreSQL together
├── requirements.txt
├── .env.example                 # DB credentials template
├── .gitignore
└── README.md
```

---

## 3. Layer 1 — Data Ingestion (`src/ingest.py`, `src/db.py`)

### 3.1 Download and Clean

```python
# src/ingest.py — implement these steps exactly

# Step 1: Download ZIP from UCI URL, unzip into data/raw/
# Use requests + zipfile. If data/raw/diabetic_data.csv already exists, skip download.

# Step 2: Load diabetic_data.csv with pandas

# Step 3: Cleaning — apply in this order:
#   a) Drop duplicate patient visits: keep only the FIRST encounter per patient_nbr
#      (sort by encounter_id ascending first, then drop_duplicates on patient_nbr, keep='first')
#      Rationale: multiple visits per patient create data leakage in train/test split
#
#   b) Drop columns: ['weight', 'encounter_id', 'patient_nbr', 'examide', 'citoglipton']
#      (weight: 97% missing; encounter_id/patient_nbr: IDs not features;
#       examide/citoglipton: single-value columns, zero variance)
#
#   c) Replace all '?' values with np.nan across the entire dataframe
#
#   d) Drop rows where gender == 'Unknown/Invalid' (only ~3 rows)
#
#   e) Drop rows where discharge_disposition_id is in [11, 13, 14, 19, 20, 21]
#      (These are patients who died or went to hospice — they cannot be readmitted,
#       keeping them would introduce noise)
#
#   f) Encode target: readmitted_binary = 1 if readmitted == '<30' else 0
#      Drop original 'readmitted' column after encoding
#
#   g) Save cleaned dataframe to data/processed/cleaned.parquet

# Step 4: Load IDs_mapping.csv and use it to add human-readable labels
#   for admission_type_id, discharge_disposition_id, admission_source_id
#   as new columns (keep numeric IDs too for the model)
```

### 3.2 ICD-9 Diagnosis Category Mapping

This is critical for `diag_1`, `diag_2`, `diag_3`. Map raw ICD-9 codes to 9 categories. Apply this mapping inside `src/features.py`:

```python
def map_icd9_to_category(code):
    """
    Maps a raw ICD-9 code string to one of 9 disease categories.
    Returns 'Other' if no match.
    """
    if pd.isna(code):
        return 'Unknown'
    code = str(code)
    if code.startswith('V') or code.startswith('E'):
        return 'External'
    try:
        num = float(code)
    except ValueError:
        return 'Other'

    if 390 <= num <= 459 or num == 785:
        return 'Circulatory'
    elif 460 <= num <= 519 or num == 786:
        return 'Respiratory'
    elif 520 <= num <= 579 or num == 787:
        return 'Digestive'
    elif num == 250:
        return 'Diabetes'          # Primary diabetes code
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
```

Apply this to `diag_1`, `diag_2`, `diag_3` → creates `diag_1_cat`, `diag_2_cat`, `diag_3_cat`. Drop the original raw diag columns after mapping.

---

## 4. Layer 2 — SQL Analytics (`sql/`)

### 4.1 PostgreSQL Setup

**Via Docker Compose** (see Section 7). Database name: `readmission_db`, user: `admin`, password: from `.env`.

### 4.2 DDL — `sql/01_create_tables.sql`

Create two tables:

```sql
CREATE TABLE IF NOT EXISTS patient_encounters (
    encounter_id        SERIAL PRIMARY KEY,
    race                VARCHAR(50),
    gender              VARCHAR(20),
    age                 VARCHAR(20),
    admission_type_id   INT,
    discharge_disposition_id INT,
    admission_source_id INT,
    time_in_hospital    INT,
    payer_code          VARCHAR(20),
    medical_specialty   VARCHAR(100),
    num_lab_procedures  INT,
    num_procedures      INT,
    num_medications     INT,
    number_outpatient   INT,
    number_emergency    INT,
    number_inpatient    INT,
    number_diagnoses    INT,
    diag_1_cat          VARCHAR(50),
    diag_2_cat          VARCHAR(50),
    diag_3_cat          VARCHAR(50),
    insulin             VARCHAR(20),
    change              VARCHAR(10),
    diabetesMed         VARCHAR(10),
    a1c_result          VARCHAR(20),
    readmitted_binary   INT          -- 1 = readmitted <30 days, 0 = not
);

CREATE TABLE IF NOT EXISTS cohort_summary AS SELECT * FROM patient_encounters WHERE 1=0;
-- (populated by analytical queries, not raw insert)
```

### 4.3 Analytical SQL — `sql/02_cohort_analysis.sql`

Write and execute these **4 queries**. Save results as CSVs into `data/processed/`. These are the "SQL analytical work" your resume claims.

```sql
-- Query 1: Readmission rate by primary diagnosis category
SELECT
    diag_1_cat,
    COUNT(*)                                             AS total_encounters,
    SUM(readmitted_binary)                               AS readmissions,
    ROUND(100.0 * SUM(readmitted_binary) / COUNT(*), 2) AS readmission_rate_pct
FROM patient_encounters
GROUP BY diag_1_cat
ORDER BY readmission_rate_pct DESC;

-- Query 2: Readmission rate by payer code (insurance type)
SELECT
    payer_code,
    COUNT(*)                                             AS total_encounters,
    SUM(readmitted_binary)                               AS readmissions,
    ROUND(100.0 * SUM(readmitted_binary) / COUNT(*), 2) AS readmission_rate_pct
FROM patient_encounters
GROUP BY payer_code
ORDER BY readmission_rate_pct DESC;

-- Query 3: Readmission rate by length of stay (time_in_hospital buckets)
SELECT
    CASE
        WHEN time_in_hospital <= 2  THEN '1-2 days'
        WHEN time_in_hospital <= 5  THEN '3-5 days'
        WHEN time_in_hospital <= 9  THEN '6-9 days'
        ELSE '10+ days'
    END                                                  AS los_bucket,
    COUNT(*)                                             AS total_encounters,
    SUM(readmitted_binary)                               AS readmissions,
    ROUND(100.0 * SUM(readmitted_binary) / COUNT(*), 2) AS readmission_rate_pct
FROM patient_encounters
GROUP BY los_bucket
ORDER BY readmission_rate_pct DESC;

-- Query 4: Prior utilization impact — inpatient visits in past year vs readmission
SELECT
    number_inpatient,
    COUNT(*)                                             AS total_encounters,
    SUM(readmitted_binary)                               AS readmissions,
    ROUND(100.0 * SUM(readmitted_binary) / COUNT(*), 2) AS readmission_rate_pct
FROM patient_encounters
WHERE number_inpatient <= 10        -- cap at 10 to avoid outlier distortion
GROUP BY number_inpatient
ORDER BY number_inpatient;
```

**These queries must be run and their outputs printed + saved.** They are referenced in the README and shown in the notebook.

---

## 5. Layer 3 — Feature Engineering + ML Pipeline

### 5.1 Feature Engineering (`src/features.py`)

Start from `data/processed/cleaned.parquet`. Output: feature matrix `X` (DataFrame) and target vector `y` (Series).

**Encoding decisions — apply exactly:**

| Feature | Encoding |
|---|---|
| `age` | Ordinal: `'[0-10)'→0`, `'[10-20)'→1`, ..., `'[90-100)'→9` |
| `gender` | Binary: `'Male'→1`, `'Female'→0` |
| `race` | One-hot encode (5 categories + Unknown) |
| `payer_code` | One-hot encode (fill NaN with `'Unknown'` first) |
| `medical_specialty` | Group rare specialties: keep top 10 by frequency, rest → `'Other'`; then one-hot |
| `diag_1_cat`, `diag_2_cat`, `diag_3_cat` | One-hot encode |
| `admission_type_id`, `discharge_disposition_id`, `admission_source_id` | Treat as categorical → one-hot |
| `A1Cresult` | Ordinal: `'None'→0`, `'Norm'→1`, `'>7'→2`, `'>8'→3` |
| `change`, `diabetesMed` | Binary: `'Ch'/'Yes'→1`, `'No'→0` |
| 24 medication columns | Ordinal: `'No'→0`, `'Steady'→1`, `'Up'→2`, `'Down'→2` (Up and Down both = changed) |

**Engineered features to add (create these explicitly):**

```python
# Number of prior healthcare encounters (proxy for disease burden)
df['total_prior_encounters'] = df['number_inpatient'] + df['number_outpatient'] + df['number_emergency']

# Number of medications changed (sum across all 24 medication columns where value != 'No')
# Compute BEFORE encoding medications
df['num_meds_changed'] = (medication_columns != 'No').sum(axis=1)

# Medication-to-procedure ratio (complexity signal)
df['med_procedure_ratio'] = df['num_medications'] / (df['num_procedures'] + 1)

# Is primary diagnosis diabetes?
df['primary_diag_is_diabetes'] = (df['diag_1_cat'] == 'Diabetes').astype(int)
```

**Final feature list:** All encoded columns + 4 engineered features above. Should produce approximately 80-100 features total after one-hot encoding.

### 5.2 Train/Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y          # CRITICAL: stratify because of class imbalance
)
```

### 5.3 Class Imbalance Handling

Apply **both** strategies and compare:

**Strategy A — SMOTE on training data only:**
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
# X_test is NEVER resampled. Only transform the training set.
```

**Strategy B — Class weights (no resampling):**
```python
# For XGBoost: compute scale_pos_weight
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos   # ~8.0 given ~11% positive rate
```

Train XGBoost with Strategy B, LightGBM with Strategy A (SMOTE). Compare both on test set.

### 5.4 Model 1 — XGBoost Baseline (`src/train.py`)

```python
import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,   # class imbalance handled here
    eval_metric='aucpr',                 # area under PR curve, better for imbalanced
    early_stopping_rounds=30,
    random_state=42,
    n_jobs=-1,
    use_label_encoder=False
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)
```

### 5.5 Model 2 — LightGBM Final (`src/train.py`)

```python
import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(
    n_estimators=1000,
    max_depth=7,
    learning_rate=0.03,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    class_weight='balanced',    # use class_weight here instead of SMOTE
    metric='average_precision',
    random_state=42,
    n_jobs=-1
)

# Train with SMOTE data
lgb_model.fit(
    X_train_smote, y_train_smote,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)
```

### 5.6 Hyperparameter Tuning

Run `Optuna` on LightGBM only (XGBoost is just the baseline). 50 trials. Optimize for AUROC on validation set using 3-fold CV:

```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1,
    }
    model = lgb.LGBMClassifier(**params)
    score = cross_val_score(model, X_train_smote, y_train_smote,
                            cv=3, scoring='roc_auc', n_jobs=-1).mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=True)
```

Retrain final LightGBM with best params from Optuna. **Save this model.**

### 5.7 Model Persistence

```python
import joblib

# Save
joblib.dump(lgb_model, 'models/lgb_final.joblib')
joblib.dump(xgb_model, 'models/xgb_baseline.joblib')

# Save feature names (critical for API)
import json
with open('models/feature_names.json', 'w') as f:
    json.dump(list(X_train.columns), f)

# Save the fitted ColumnTransformer / encoding pipeline too
joblib.dump(preprocessing_pipeline, 'models/preprocessor.joblib')
```

---

## 6. Layer 4 — Evaluation + Visualizations (`src/evaluate.py`)

All plots saved as PNG to `plots/`. All metrics printed to stdout and saved to `plots/metrics.json`.

### 6.1 Metrics to Compute (for both models)

```python
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)

metrics = {
    'auroc':             roc_auc_score(y_test, y_pred_proba),
    'avg_precision':     average_precision_score(y_test, y_pred_proba),
    'classification_report': classification_report(y_test, y_pred, output_dict=True)
}
```

**Threshold for binary prediction:** Do NOT use 0.5. Use the threshold that maximizes F1 on the validation set:
```python
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_val, y_val_proba)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_threshold = thresholds[f1_scores.argmax()]
```

### 6.2 Plots to Generate (save all as PNG)

**Plot 1 — ROC Curve Comparison (`plots/roc_curve.png`)**  
Both XGBoost and LightGBM on the same axes. Label each with its AUROC. Use seaborn style.

**Plot 2 — Precision-Recall Curve (`plots/pr_curve.png`)**  
Both models on same axes. Label with Average Precision score. Include no-skill baseline (horizontal line at positive rate ~0.11).

**Plot 3 — Confusion Matrix (`plots/confusion_matrix.png`)**  
For LightGBM at the optimal threshold. Use seaborn heatmap with annotation.

**Plot 4 — SHAP Beeswarm Plot (`plots/shap_beeswarm.png`)**  
Global feature importance across all test samples. Show top 20 features. Use:
```python
import shap
explainer = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(X_test)
# For binary classification, shap_values is a list; use shap_values[1] for positive class
shap.summary_plot(shap_values[1], X_test, max_display=20, show=False)
plt.savefig('plots/shap_beeswarm.png', bbox_inches='tight', dpi=150)
plt.close()
```

**Plot 5 — SHAP Waterfall for single patient (`plots/shap_waterfall_example.png`)**  
Pick one true positive prediction from the test set. Show its individual SHAP waterfall.
```python
shap.waterfall_plot(shap.Explanation(
    values=shap_values[1][idx],
    base_values=explainer.expected_value[1],
    data=X_test.iloc[idx],
    feature_names=X_test.columns.tolist()
), show=False)
plt.savefig('plots/shap_waterfall_example.png', bbox_inches='tight', dpi=150)
plt.close()
```

**Plot 6 — Cohort Analysis Bar Chart (`plots/cohort_readmission_rates.png`)**  
4-panel figure (2×2) showing the outputs of the 4 SQL queries from Section 4.3. Use matplotlib subplots. Bar chart for each. This demonstrates SQL → visualization pipeline.

---

## 7. Layer 5 — FastAPI Serving + Docker

### 7.1 `api/schema.py` — Pydantic Models

```python
from pydantic import BaseModel, Field
from typing import Optional

class PatientRecord(BaseModel):
    race: Optional[str] = "Unknown"
    gender: str
    age: str                          # e.g. "[50-60)"
    admission_type_id: int
    discharge_disposition_id: int
    admission_source_id: int
    time_in_hospital: int
    payer_code: Optional[str] = "Unknown"
    medical_specialty: Optional[str] = "Unknown"
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    number_diagnoses: int
    diag_1: str
    diag_2: Optional[str] = "Unknown"
    diag_3: Optional[str] = "Unknown"
    insulin: str = "No"
    change: str = "No"
    diabetesMed: str = "No"
    a1c_result: str = "None"
    # All 24 medication columns optional, default "No"
    metformin: str = "No"
    repaglinide: str = "No"
    nateglinide: str = "No"
    chlorpropamide: str = "No"
    glimepiride: str = "No"
    acetohexamide: str = "No"
    glipizide: str = "No"
    glyburide: str = "No"
    tolbutamide: str = "No"
    pioglitazone: str = "No"
    rosiglitazone: str = "No"
    acarbose: str = "No"
    miglitol: str = "No"
    troglitazone: str = "No"
    tolazamide: str = "No"
    examide: str = "No"
    citoglipton: str = "No"
    glyburide_metformin: str = "No"
    glipizide_metformin: str = "No"
    glimepiride_pioglitazone: str = "No"
    metformin_rosiglitazone: str = "No"
    metformin_pioglitazone: str = "No"

class PredictionResponse(BaseModel):
    readmission_probability: float = Field(..., description="Probability of 30-day readmission (0–1)")
    readmission_risk: str             = Field(..., description="LOW / MEDIUM / HIGH based on threshold")
    top_risk_factors: list[str]       = Field(..., description="Top 5 SHAP-based risk factors for this patient")
```

### 7.2 `api/main.py` — FastAPI App

```python
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import uvicorn

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model artifacts once on startup
    app.state.model = load_model()          # from api/model_loader.py
    app.state.preprocessor = load_preprocessor()
    app.state.explainer = load_explainer()  # shap.TreeExplainer
    app.state.feature_names = load_feature_names()
    app.state.threshold = 0.35              # from optimal threshold computed in evaluate.py
    yield
    # cleanup if needed

app = FastAPI(
    title="Hospital Readmission Risk Predictor",
    description="Predicts 30-day readmission risk for diabetic patients",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health():
    return {"status": "healthy", "model": "lgb_final"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(record: PatientRecord):
    try:
        # 1. Convert Pydantic model to DataFrame row
        # 2. Run through preprocessor (same feature engineering as training)
        # 3. Get probability from model
        # 4. Get SHAP values → extract top 5 feature names by |shap_value|
        # 5. Compute risk tier: <0.2 = LOW, 0.2-0.4 = MEDIUM, >0.4 = HIGH
        # 6. Return PredictionResponse
        ...
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def model_info():
    """Returns model metadata: AUROC, training date, feature count."""
    return {
        "model_type": "LightGBM",
        "auroc": 0.82,       # update with actual
        "avg_precision": ..., # update with actual
        "n_features": ...,
        "training_samples": ...,
        "positive_rate": 0.11
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
```

### 7.3 `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Model artifacts must be present at build time
# (or mounted as a volume)
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

### 7.4 `docker-compose.yml`

```yaml
version: "3.9"

services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: readmission_db
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./sql:/docker-entrypoint-initdb.d   # auto-runs SQL files on first start
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U admin -d readmission_db"]
      interval: 10s
      timeout: 5s
      retries: 5

  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://admin:${POSTGRES_PASSWORD}@postgres:5432/readmission_db
    volumes:
      - ./models:/app/models       # model artifacts
      - ./plots:/app/plots         # generated plots accessible from host
    depends_on:
      postgres:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  postgres_data:
```

---

## 8. `requirements.txt`

```
# Data
pandas==2.1.4
numpy==1.26.2
pyarrow==14.0.1          # for parquet
requests==2.31.0

# Database
psycopg2-binary==2.9.9
sqlalchemy==2.0.23

# ML
scikit-learn==1.3.2
xgboost==2.0.2
lightgbm==4.1.0
imbalanced-learn==0.11.0  # SMOTE
optuna==3.4.0

# Explainability
shap==0.44.0

# Serving
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.2
httpx==0.25.2             # for test client

# Visualization
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.18.0

# Utilities
python-dotenv==1.0.0
joblib==1.3.2
tqdm==4.66.1
```

---

## 9. Tests (`tests/`)

### `tests/test_features.py`
Write unit tests for:
- `map_icd9_to_category()` — test each category boundary (e.g., 250.0 → Diabetes, 391 → Circulatory, `'V10'` → External)
- Age encoding — test all 10 brackets
- That feature matrix has no NaN values after preprocessing
- That engineered features (`total_prior_encounters`, `num_meds_changed`) are computed correctly

### `tests/test_api.py`
Use `httpx.AsyncClient` with FastAPI's test client:
- `GET /health` returns 200 and `{"status": "healthy"}`
- `POST /predict` with valid payload returns a `PredictionResponse` with `readmission_probability` between 0 and 1
- `POST /predict` with missing required field returns 422

---

## 10. `.env.example`

```
POSTGRES_PASSWORD=changeme
DATABASE_URL=postgresql://admin:changeme@localhost:5432/readmission_db
```

---

## 11. `.gitignore`

```
data/raw/
data/processed/
models/*.joblib
models/*.json
!models/.gitkeep
plots/*.png
!plots/.gitkeep
.env
__pycache__/
*.pyc
.ipynb_checkpoints/
```

---

## 12. `README.md` — Required Sections

The README is part of the deliverable. It must contain all of the following:

### Sections (in order):

**1. Project Overview** — 3 sentences. What this is, what dataset, what it predicts.

**2. Motivation** — 1 paragraph on why 30-day readmission prediction matters in healthcare. Mention: CMS penalizes hospitals financially for excess readmissions (Hospital Readmission Reduction Program). Reduction = direct financial impact.

**3. Dataset** — Source, size, feature count, target class distribution (show the 11% positive rate explicitly).

**4. Architecture Diagram** — ASCII or embedded image showing: Raw CSV → Ingest → PostgreSQL → Feature Engineering → XGBoost/LightGBM → SHAP → FastAPI → Docker

**5. SQL Cohort Analysis** — Embed the 4 cohort analysis plots. Write 1-2 sentence insight for each (e.g., "Patients with Circulatory diagnoses showed the highest 30-day readmission rate at X.X%, nearly 2× the dataset average").

**6. ML Pipeline Decisions** — Explain in prose (not bullet lists):
- Why two models (baseline vs final)
- Why AUROC and not accuracy (class imbalance reason)
- SMOTE vs class weights tradeoff — explain that SMOTE generates synthetic minority samples in feature space which can introduce noise, while class weights are simpler but may under-represent minority class complexity
- Why optimal threshold instead of 0.5

**7. Results** — Embed all 6 plots. Table comparing XGBoost vs LightGBM:
| Metric | XGBoost | LightGBM |
|---|---|---|
| AUROC | ? | ? |
| Avg Precision | ? | ? |
| F1 (optimal threshold) | ? | ? |

**8. SHAP Explainability** — Embed beeswarm and waterfall. Explain what SHAP values mean and name the top 3 most influential features from the beeswarm.

**9. API Usage** — Show `curl` example for `/predict` endpoint with a real sample payload.

**10. How to Run** — 
```bash
# 1. Clone + setup
git clone ...
cp .env.example .env
# (fill in password)

# 2. Train model
python -m src.ingest
python -m src.features
python -m src.train
python -m src.evaluate

# 3. Start services
docker-compose up --build

# 4. Test
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @tests/sample_payload.json
```

---

## 13. Build Order

Execute in this exact sequence:

1. Create folder structure and `requirements.txt`
2. Implement `src/ingest.py` — download, clean, save parquet
3. Implement `src/db.py` + run `sql/01_create_tables.sql` — load data into PostgreSQL
4. Run `sql/02_cohort_analysis.sql` — generate and save cohort CSVs
5. Implement `src/features.py` — feature engineering, encoding, engineered features
6. Implement `src/train.py` — XGBoost baseline, then LightGBM + SMOTE + Optuna
7. Implement `src/evaluate.py` — all 6 plots, metrics JSON
8. Implement `api/schema.py`, `api/model_loader.py`, `api/main.py`
9. Write `Dockerfile` + `docker-compose.yml`
10. Write `tests/test_features.py` + `tests/test_api.py`
11. Write `README.md` with all sections populated with actual results
12. Final check: `docker-compose up --build` runs clean, `GET /health` returns 200, `POST /predict` returns a valid response

---

## 14. Non-Negotiable Quality Bars

- No NaN values in the feature matrix passed to the model — verify with `assert X_train.isna().sum().sum() == 0`
- Test set is never touched during preprocessing fitting — fit transformers on train, transform test
- SMOTE is applied only to training data — never test data
- Model artifacts are saved with joblib before the API is built
- All 6 plots are generated and saved as PNG before README is written
- Docker Compose must start clean on a fresh machine with only Docker installed (no local Python dependency)
- `GET /health` must return 200 within 30 seconds of `docker-compose up`
