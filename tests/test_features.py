from src.features import (
    AGE_MAP,
    ReadmissionPreprocessor,
    add_engineered_features,
    load_cleaned,
    map_icd9_to_category,
)


def test_icd9_diabetes_and_boundaries():
    assert map_icd9_to_category(250.0) == "Diabetes"
    assert map_icd9_to_category("250.83") == "Diabetes"
    assert map_icd9_to_category(391) == "Circulatory"
    assert map_icd9_to_category(785) == "Circulatory"
    assert map_icd9_to_category(460) == "Respiratory"
    assert map_icd9_to_category(786) == "Respiratory"
    assert map_icd9_to_category(520) == "Digestive"
    assert map_icd9_to_category(800) == "Injury"
    assert map_icd9_to_category(710) == "Musculoskeletal"
    assert map_icd9_to_category(580) == "Genitourinary"
    assert map_icd9_to_category(140) == "Neoplasms"
    assert map_icd9_to_category("V10") == "External"
    assert map_icd9_to_category("E950") == "External"
    assert map_icd9_to_category(None) == "Unknown"
    assert map_icd9_to_category("ZZZ") == "Other"


def test_age_encoding_all_brackets():
    expected = {
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
    assert AGE_MAP == expected


def test_feature_matrix_has_no_nan():
    df = load_cleaned()
    X = df.drop(columns=["readmitted_binary"])
    pre = ReadmissionPreprocessor()
    Xt = pre.fit_transform(X.head(500))
    assert Xt.isna().sum().sum() == 0


def test_engineered_features():
    df = load_cleaned().head(20)
    out = add_engineered_features(df)
    expected_prior = df["number_inpatient"] + df["number_outpatient"] + df["number_emergency"]
    assert (out["total_prior_encounters"] == expected_prior).all()
    assert (out["num_meds_changed"] >= 0).all()
    assert out["num_meds_changed"].notna().all()
    assert (out["med_procedure_ratio"] == df["num_medications"] / (df["num_procedures"] + 1)).all()
