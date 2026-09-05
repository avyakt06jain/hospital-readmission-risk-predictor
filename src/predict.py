"""Inference helpers used by the FastAPI app."""

import numpy as np
import pandas as pd

from src.features import MEDICATION_COLUMNS

API_TO_DATA = {
    "a1c_result": "A1Cresult",
    "glyburide_metformin": "glyburide-metformin",
    "glipizide_metformin": "glipizide-metformin",
    "glimepiride_pioglitazone": "glimepiride-pioglitazone",
    "metformin_rosiglitazone": "metformin-rosiglitazone",
    "metformin_pioglitazone": "metformin-pioglitazone",
}


def record_to_frame(record: dict) -> pd.DataFrame:
    row = dict(record)
    for src, dest in API_TO_DATA.items():
        if src in row:
            row[dest] = row.pop(src)
    row.pop("examide", None)
    row.pop("citoglipton", None)
    for col in MEDICATION_COLUMNS:
        row.setdefault(col, "No")
    return pd.DataFrame([row])


def risk_tier(prob: float) -> str:
    if prob < 0.2:
        return "LOW"
    if prob <= 0.4:
        return "MEDIUM"
    return "HIGH"


def shap_top_features(explainer, X: pd.DataFrame, k: int = 5) -> list[str]:
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        values = np.array(sv[1]).reshape(-1)
    else:
        values = np.array(sv).reshape(-1)
    order = np.argsort(np.abs(values))[::-1][:k]
    names = list(X.columns)
    return [names[i] for i in order]


def predict_one(record: dict, model, preprocessor, explainer, feature_names) -> dict:
    raw = record_to_frame(record)
    X = preprocessor.transform(raw)
    X = X.reindex(columns=feature_names, fill_value=0.0)
    proba = float(model.predict_proba(X)[0, 1])
    return {
        "readmission_probability": proba,
        "readmission_risk": risk_tier(proba),
        "top_risk_factors": shap_top_features(explainer, X),
    }
