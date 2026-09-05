"""AUROC, PR curves, confusion matrix, SHAP plots, cohort bar chart."""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from src import ROOT
from src.features import load_cleaned

PLOTS = ROOT / "plots"
MODELS = ROOT / "models"
PROCESSED = ROOT / "data" / "processed"
sns.set_style("whitegrid")


def _test_split():
    df = load_cleaned()
    y = df["readmitted_binary"].astype(int)
    X = df.drop(columns=["readmitted_binary"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    preprocessor = joblib.load(MODELS / "preprocessor.joblib")
    return (
        preprocessor.transform(X_train),
        preprocessor.transform(X_test),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


def best_f1_threshold(y_true, y_proba) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    # precision_recall_curve returns one extra precision/recall vs thresholds
    if len(thresholds) == 0:
        return 0.35
    return float(thresholds[np.nanargmax(f1_scores[:-1])])


def model_metrics(y_true, y_proba, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, y_proba)),
        "avg_precision": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred)),
        "threshold": float(threshold),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }


def plot_roc(y_test, proba_map: dict) -> None:
    plt.figure(figsize=(8, 6))
    for name, proba in proba_map.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUROC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — 30-day Readmission")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_pr(y_test, proba_map: dict) -> None:
    baseline = float(np.mean(y_test))
    plt.figure(figsize=(8, 6))
    for name, proba in proba_map.items():
        precision, recall, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        plt.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    plt.axhline(baseline, color="gray", linestyle="--", label=f"No-skill ({baseline:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "pr_curve.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion(y_test, y_pred) -> None:
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("LightGBM Confusion Matrix (optimal F1 threshold)")
    plt.tight_layout()
    plt.savefig(PLOTS / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()


def _shap_positive(explainer, X):
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        return sv[1], explainer.expected_value[1]
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)) and np.size(base) > 1:
        base = base[1]
    return sv, float(np.array(base).reshape(-1)[0])


def plot_shap(lgb_model, X_test, y_test, y_pred) -> None:
    # subsample for speed; beeswarm still reflects global importance
    sample = X_test.sample(n=min(1000, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(lgb_model)
    sv, base = _shap_positive(explainer, sample)

    plt.figure()
    shap.summary_plot(sv, sample, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(PLOTS / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    tp = np.where((y_test.values == 1) & (y_pred == 1))[0]
    idx = int(tp[0]) if len(tp) else 0
    row = X_test.iloc[[idx]]
    sv_one, base_one = _shap_positive(explainer, row)
    shap.waterfall_plot(
        shap.Explanation(
            values=np.array(sv_one).reshape(-1),
            base_values=base_one,
            data=row.iloc[0],
            feature_names=X_test.columns.tolist(),
        ),
        show=False,
        max_display=15,
    )
    plt.tight_layout()
    plt.savefig(PLOTS / "shap_waterfall_example.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_cohort() -> None:
    files = [
        ("cohort_by_diagnosis.csv", "diag_1_cat", "By primary diagnosis"),
        ("cohort_by_payer.csv", "payer_code", "By payer code"),
        ("cohort_by_los.csv", "los_bucket", "By length of stay"),
        ("cohort_by_inpatient.csv", "number_inpatient", "By prior inpatient visits"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (fname, xcol, title) in zip(axes.ravel(), files):
        table = pd.read_csv(PROCESSED / fname)
        if xcol == "payer_code":
            table = table.head(12)
        ax.bar(table[xcol].astype(str), table["readmission_rate_pct"], color="#2c7fb8")
        ax.set_title(title)
        ax.set_ylabel("Readmission rate (%)")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    fig.suptitle("SQL cohort analysis — 30-day readmission rates")
    fig.tight_layout()
    fig.savefig(PLOTS / "cohort_readmission_rates.png", dpi=150, bbox_inches="tight")
    plt.close()


def run() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = _test_split()
    xgb_model = joblib.load(MODELS / "xgb_baseline.joblib")
    lgb_model = joblib.load(MODELS / "lgb_final.joblib")

    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
    xgb_threshold = best_f1_threshold(y_train, xgb_model.predict_proba(X_train)[:, 1])
    lgb_threshold = best_f1_threshold(y_train, lgb_model.predict_proba(X_train)[:, 1])
    lgb_pred = (lgb_proba >= lgb_threshold).astype(int)

    metrics = {
        "xgboost": model_metrics(y_test, xgb_proba, xgb_threshold),
        "lightgbm": model_metrics(y_test, lgb_proba, lgb_threshold),
        "threshold": lgb_threshold,
    }
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "classification_report"} if isinstance(v, dict) else v for k, v in metrics.items()}, indent=2))

    plot_roc(y_test, {"XGBoost": xgb_proba, "LightGBM": lgb_proba})
    plot_pr(y_test, {"XGBoost": xgb_proba, "LightGBM": lgb_proba})
    plot_confusion(y_test, lgb_pred)
    plot_shap(lgb_model, X_test, y_test, lgb_pred)
    plot_cohort()

    with open(PLOTS / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(MODELS / "threshold.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": lgb_threshold}, f)
    meta_path = MODELS / "train_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update(
        {
            "auroc": metrics["lightgbm"]["auroc"],
            "avg_precision": metrics["lightgbm"]["avg_precision"],
            "f1": metrics["lightgbm"]["f1"],
            "threshold": lgb_threshold,
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Saved plots to {PLOTS}")


if __name__ == "__main__":
    run()
