from fastapi.testclient import TestClient

from api.main import app

SAMPLE = {
    "race": "Caucasian",
    "gender": "Female",
    "age": "[50-60)",
    "admission_type_id": 1,
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
    "time_in_hospital": 4,
    "payer_code": "MC",
    "medical_specialty": "InternalMedicine",
    "num_lab_procedures": 44,
    "num_procedures": 0,
    "num_medications": 16,
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 0,
    "number_diagnoses": 8,
    "diag_1": "428",
    "diag_2": "250.01",
    "diag_3": "401",
    "insulin": "Steady",
    "change": "Ch",
    "diabetesMed": "Yes",
    "a1c_result": "None",
}


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "healthy"
        assert body["model"] == "lgb_final"


def test_predict_valid_payload():
    with TestClient(app) as client:
        r = client.post("/predict", json=SAMPLE)
        assert r.status_code == 200
        body = r.json()
        assert 0.0 <= body["readmission_probability"] <= 1.0
        assert body["readmission_risk"] in {"LOW", "MEDIUM", "HIGH"}
        assert len(body["top_risk_factors"]) == 5


def test_predict_missing_required_field():
    with TestClient(app) as client:
        payload = {k: v for k, v in SAMPLE.items() if k != "gender"}
        r = client.post("/predict", json=payload)
        assert r.status_code == 422
