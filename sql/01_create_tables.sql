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
    "diabetesMed"       VARCHAR(10),
    a1c_result          VARCHAR(20),
    readmitted_binary   INT
);

CREATE TABLE IF NOT EXISTS cohort_summary AS SELECT * FROM patient_encounters WHERE 1=0;
