import json

import joblib
import shap

from src import ROOT

MODELS = ROOT / "models"


def load_model():
    return joblib.load(MODELS / "lgb_final.joblib")


def load_preprocessor():
    return joblib.load(MODELS / "preprocessor.joblib")


def load_explainer():
    return shap.TreeExplainer(load_model())


def load_feature_names() -> list[str]:
    with open(MODELS / "feature_names.json", encoding="utf-8") as f:
        return json.load(f)


def load_threshold() -> float:
    path = MODELS / "threshold.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["threshold"]
    return 0.35


def load_model_info() -> dict:
    path = MODELS / "train_meta.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}
