"""Train XGBoost baseline and LightGBM final model (SMOTE + Optuna)."""

import json
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score, train_test_split

from src import ROOT
from src.features import ReadmissionPreprocessor, load_cleaned

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

MODELS = ROOT / "models"
N_TRIALS = 20


def _split_xy():
    df = load_cleaned()
    y = df["readmitted_binary"].astype(int)
    X = df.drop(columns=["readmitted_binary"])
    return train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)


def _encode(X_train, X_test):
    preprocessor = ReadmissionPreprocessor()
    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)
    assert X_train_enc.isna().sum().sum() == 0, "NaN in X_train"
    assert X_test_enc.isna().sum().sum() == 0, "NaN in X_test"
    return preprocessor, X_train_enc, X_test_enc


def train_xgboost(X_train, y_train, X_test, y_test):
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / pos
    print(f"XGBoost scale_pos_weight={scale_pos_weight:.3f}")

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)
    return model


def train_lightgbm(X_train, y_train, X_test, y_test, params=None):
    if params is None:
        params = {
            "n_estimators": 1000,
            "max_depth": 7,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
        }
    model = lgb.LGBMClassifier(
        class_weight="balanced",
        metric="average_precision",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        **params,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )
    return model


def tune_lightgbm(X_train_smote, y_train_smote) -> dict:
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        model = lgb.LGBMClassifier(**params)
        score = cross_val_score(
            model, X_train_smote, y_train_smote, cv=3, scoring="roc_auc", n_jobs=1
        ).mean()
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    print(f"Optuna best AUROC={study.best_value:.4f}  params={study.best_params}")
    return study.best_params


def run() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = _split_xy()
    preprocessor, X_train_enc, X_test_enc = _encode(X_train, X_test)
    print(f"Encoded features: {X_train_enc.shape[1]}")

    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_enc, y_train)
    X_train_smote = pd.DataFrame(X_train_smote, columns=X_train_enc.columns)
    print(f"SMOTE train size: {X_train_smote.shape}  pos_rate={np.mean(y_train_smote):.3f}")

    print("\n--- XGBoost baseline (class weights) ---")
    xgb_model = train_xgboost(X_train_enc, y_train, X_test_enc, y_test)

    print("\n--- LightGBM (SMOTE + spec params) ---")
    train_lightgbm(X_train_smote, y_train_smote, X_test_enc, y_test)

    print(f"\n--- Optuna LightGBM ({N_TRIALS} trials) ---")
    best_params = tune_lightgbm(X_train_smote, y_train_smote)

    print("\n--- Final LightGBM (best Optuna params) ---")
    lgb_model = train_lightgbm(X_train_smote, y_train_smote, X_test_enc, y_test, params=best_params)

    joblib.dump(lgb_model, MODELS / "lgb_final.joblib")
    joblib.dump(xgb_model, MODELS / "xgb_baseline.joblib")
    joblib.dump(preprocessor, MODELS / "preprocessor.joblib")
    with open(MODELS / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(list(X_train_enc.columns), f)
    with open(MODELS / "train_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "training_samples": int(len(X_train_enc)),
                "n_features": int(X_train_enc.shape[1]),
                "positive_rate": float(y_train.mean()),
                "best_params": {k: (float(v) if isinstance(v, float) else int(v) if isinstance(v, (int, np.integer)) else v) for k, v in best_params.items()},
            },
            f,
            indent=2,
        )
    print("Saved model artifacts to models/")


if __name__ == "__main__":
    run()
