from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException

from api.model_loader import (
    load_explainer,
    load_feature_names,
    load_model,
    load_model_info,
    load_preprocessor,
    load_threshold,
)
from api.schema import PatientRecord, PredictionResponse
from src.predict import predict_one


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    app.state.preprocessor = load_preprocessor()
    app.state.explainer = load_explainer()
    app.state.feature_names = load_feature_names()
    app.state.threshold = load_threshold()
    app.state.model_info = load_model_info()
    yield


app = FastAPI(
    title="Hospital Readmission Risk Predictor",
    description="Predicts 30-day readmission risk for diabetic patients",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "healthy", "model": "lgb_final"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(record: PatientRecord):
    try:
        return predict_one(
            record.model_dump(),
            app.state.model,
            app.state.preprocessor,
            app.state.explainer,
            app.state.feature_names,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/model-info")
async def model_info():
    info = app.state.model_info
    return {
        "model_type": "LightGBM",
        "auroc": info.get("auroc"),
        "avg_precision": info.get("avg_precision"),
        "n_features": info.get("n_features"),
        "training_samples": info.get("training_samples"),
        "positive_rate": info.get("positive_rate", 0.11),
    }


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
