"""Feature engineering and encoding for the readmission model."""

import pandas as pd

from src import ROOT

MEDICATION_COLUMNS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

AGE_MAP = {
    "[0-10)": 0,
    "[10-20)": 1,
    "[20-30)": 2,
    "[30-40)": 3,
    "[40-50)": 4,
    "[50-60)": 5,
    "[60-70)": 6,
    "[70-80)": 7,
    "[80-90)": 8,
    "[90-100)": 9,
}

A1C_MAP = {"None": 0, "Norm": 1, ">7": 2, ">8": 3}
MED_MAP = {"No": 0, "Steady": 1, "Up": 2, "Down": 2}
ID_COLS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]
LABEL_COLS = [
    "admission_type_label",
    "discharge_disposition_label",
    "admission_source_label",
    "max_glu_serum",
]
ONEHOT_COLS = [
    "race",
    "payer_code",
    "medical_specialty",
    "diag_1_cat",
    "diag_2_cat",
    "diag_3_cat",
] + ID_COLS


def map_icd9_to_category(code) -> str:
    """Map a raw ICD-9 code string to one of 9 disease categories."""
    if pd.isna(code):
        return "Unknown"
    code = str(code)
    if code.startswith("V") or code.startswith("E"):
        return "External"
    try:
        num = float(code)
    except ValueError:
        return "Other"

    if 390 <= num <= 459 or num == 785:
        return "Circulatory"
    if 460 <= num <= 519 or num == 786:
        return "Respiratory"
    if 520 <= num <= 579 or num == 787:
        return "Digestive"
    # 250.xx is the ICD-9 diabetes family (spec wrote `num == 250`)
    if 250 <= num < 251:
        return "Diabetes"
    if 800 <= num <= 999:
        return "Injury"
    if 710 <= num <= 739:
        return "Musculoskeletal"
    if 580 <= num <= 629 or num == 788:
        return "Genitourinary"
    if 140 <= num <= 239:
        return "Neoplasms"
    return "Other"


def add_diag_categories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("diag_1", "diag_2", "diag_3"):
        if col in df.columns:
            df[f"{col}_cat"] = df[col].map(map_icd9_to_category)
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 4 engineered features. Call BEFORE encoding medications."""
    df = df.copy()
    df["total_prior_encounters"] = (
        df["number_inpatient"] + df["number_outpatient"] + df["number_emergency"]
    )
    meds = [c for c in MEDICATION_COLUMNS if c in df.columns]
    df["num_meds_changed"] = (df[meds] != "No").sum(axis=1)
    df["med_procedure_ratio"] = df["num_medications"] / (df["num_procedures"] + 1)
    if "diag_1_cat" not in df.columns:
        df = add_diag_categories(df)
    df["primary_diag_is_diabetes"] = (df["diag_1_cat"] == "Diabetes").astype(int)
    return df


class ReadmissionPreprocessor:
    """Fit on train only: specialty grouping + one-hot columns. Transform applies the same encoding."""

    def __init__(self):
        self.top_specialties_ = None
        self.feature_names_ = None

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        drop = [c for c in LABEL_COLS if c in df.columns]
        df = df.drop(columns=drop)

        df = add_diag_categories(df)
        df = df.drop(columns=[c for c in ("diag_1", "diag_2", "diag_3") if c in df.columns])

        df["race"] = df["race"].fillna("Unknown")
        df["payer_code"] = df["payer_code"].fillna("Unknown")
        df["medical_specialty"] = df["medical_specialty"].fillna("Unknown")
        df["A1Cresult"] = df["A1Cresult"].fillna("None")
        for col in MEDICATION_COLUMNS:
            if col in df.columns:
                df[col] = df[col].fillna("No")

        df = add_engineered_features(df)
        return df

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["age"] = df["age"].map(AGE_MAP).fillna(0).astype(int)
        df["gender"] = (df["gender"] == "Male").astype(int)
        df["A1Cresult"] = df["A1Cresult"].map(A1C_MAP).fillna(0).astype(int)
        df["change"] = (df["change"] == "Ch").astype(int)
        df["diabetesMed"] = (df["diabetesMed"] == "Yes").astype(int)

        for col in MEDICATION_COLUMNS:
            if col in df.columns:
                df[col] = df[col].map(MED_MAP).fillna(0).astype(int)

        df["medical_specialty"] = df["medical_specialty"].where(
            df["medical_specialty"].isin(self.top_specialties_), "Other"
        )
        for col in ID_COLS:
            df[col] = df[col].astype(str)

        df = pd.get_dummies(df, columns=ONEHOT_COLS, dummy_na=False)
        return df.astype(float)

    def fit(self, X: pd.DataFrame, y=None):
        prepared = self._prepare(X)
        counts = prepared["medical_specialty"].value_counts()
        self.top_specialties_ = counts.head(10).index.tolist()
        encoded = self._encode(prepared)
        self.feature_names_ = encoded.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.feature_names_ is None:
            raise RuntimeError("Preprocessor is not fitted.")
        encoded = self._encode(self._prepare(X))
        return encoded.reindex(columns=self.feature_names_, fill_value=0.0)

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


def load_cleaned() -> pd.DataFrame:
    path = ROOT / "data" / "processed" / "cleaned.parquet"
    return pd.read_parquet(path)


def run() -> None:
    df = load_cleaned()
    y = df["readmitted_binary"]
    X = df.drop(columns=["readmitted_binary"])
    pre = ReadmissionPreprocessor()
    # Demo-only fit on all rows so `python -m src.features` prints the shape.
    # Training fits a fresh preprocessor on X_train only (see src/train.py).
    Xt = pre.fit_transform(X)
    n_nan = int(Xt.isna().sum().sum())
    print(f"Rows={len(Xt)}  features={Xt.shape[1]}  NaNs={n_nan}  positive_rate={y.mean():.4f}")
    print("Engineered columns present:", all(
        c in pre._prepare(X).columns
        for c in (
            "total_prior_encounters",
            "num_meds_changed",
            "med_procedure_ratio",
            "primary_diag_is_diabetes",
        )
    ))
    if n_nan != 0:
        raise SystemExit("Feature matrix still contains NaN values.")


if __name__ == "__main__":
    run()
