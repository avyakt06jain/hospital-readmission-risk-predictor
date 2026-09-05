from typing import Optional

from pydantic import BaseModel, Field


class PatientRecord(BaseModel):
    race: Optional[str] = "Unknown"
    gender: str
    age: str
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
    readmission_risk: str = Field(..., description="LOW / MEDIUM / HIGH based on threshold")
    top_risk_factors: list[str] = Field(..., description="Top 5 SHAP-based risk factors for this patient")
