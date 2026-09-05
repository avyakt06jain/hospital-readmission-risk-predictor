"""PostgreSQL load + cohort analysis CSVs."""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from src import ROOT
from src.features import add_diag_categories, load_cleaned

load_dotenv(ROOT / ".env")

PROCESSED = ROOT / "data" / "processed"
SQL_DIR = ROOT / "sql"
TABLE_COLS = [
    "race",
    "gender",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "payer_code",
    "medical_specialty",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "diag_1_cat",
    "diag_2_cat",
    "diag_3_cat",
    "insulin",
    "change",
    "diabetesMed",
    "a1c_result",
    "readmitted_binary",
]


def encounters_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = add_diag_categories(df)
    out["payer_code"] = out["payer_code"].fillna("Unknown")
    out["medical_specialty"] = out["medical_specialty"].fillna("Unknown")
    out["race"] = out["race"].fillna("Unknown")
    out["a1c_result"] = out["A1Cresult"] if "A1Cresult" in out.columns else "None"
    return out[TABLE_COLS]


def cohort_csvs_from_df(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Same four aggregations as sql/02_cohort_analysis.sql, via pandas."""
    PROCESSED.mkdir(parents=True, exist_ok=True)

    by_diag = (
        df.groupby("diag_1_cat", dropna=False)
        .agg(total_encounters=("readmitted_binary", "size"), readmissions=("readmitted_binary", "sum"))
        .reset_index()
    )
    by_diag["readmission_rate_pct"] = (100.0 * by_diag["readmissions"] / by_diag["total_encounters"]).round(2)
    by_diag = by_diag.sort_values("readmission_rate_pct", ascending=False)

    by_payer = (
        df.groupby("payer_code", dropna=False)
        .agg(total_encounters=("readmitted_binary", "size"), readmissions=("readmitted_binary", "sum"))
        .reset_index()
    )
    by_payer["readmission_rate_pct"] = (100.0 * by_payer["readmissions"] / by_payer["total_encounters"]).round(2)
    by_payer = by_payer.sort_values("readmission_rate_pct", ascending=False)

    los = df.copy()
    los["los_bucket"] = pd.cut(
        los["time_in_hospital"],
        bins=[0, 2, 5, 9, 100],
        labels=["1-2 days", "3-5 days", "6-9 days", "10+ days"],
        include_lowest=True,
    )
    by_los = (
        los.groupby("los_bucket", observed=True)
        .agg(total_encounters=("readmitted_binary", "size"), readmissions=("readmitted_binary", "sum"))
        .reset_index()
    )
    by_los["readmission_rate_pct"] = (100.0 * by_los["readmissions"] / by_los["total_encounters"]).round(2)
    by_los = by_los.sort_values("readmission_rate_pct", ascending=False)

    inp = df[df["number_inpatient"] <= 10]
    by_inp = (
        inp.groupby("number_inpatient")
        .agg(total_encounters=("readmitted_binary", "size"), readmissions=("readmitted_binary", "sum"))
        .reset_index()
    )
    by_inp["readmission_rate_pct"] = (100.0 * by_inp["readmissions"] / by_inp["total_encounters"]).round(2)
    by_inp = by_inp.sort_values("number_inpatient")

    results = {
        "cohort_by_diagnosis.csv": by_diag,
        "cohort_by_payer.csv": by_payer,
        "cohort_by_los.csv": by_los,
        "cohort_by_inpatient.csv": by_inp,
    }
    for name, table in results.items():
        path = PROCESSED / name
        table.to_csv(path, index=False)
        print(f"\n=== {name} ===")
        print(table.to_string(index=False))
        print(f"Saved {path}")
    return results


def get_engine():
    url = os.getenv("DATABASE_URL")
    if not url:
        return None
    return create_engine(url)


def _statements(sql_text: str) -> list[str]:
    stmts = []
    for chunk in sql_text.split(";"):
        lines = [ln for ln in chunk.splitlines() if not ln.strip().startswith("--")]
        stmt = "\n".join(lines).strip()
        if stmt:
            stmts.append(stmt)
    return stmts


def _run_sql_file(conn, path: Path) -> None:
    for stmt in _statements(path.read_text(encoding="utf-8")):
        conn.execute(text(stmt))


def load_postgres(df: pd.DataFrame) -> bool:
    engine = get_engine()
    if engine is None:
        print("DATABASE_URL not set — skipping Postgres load.")
        return False
    try:
        with engine.begin() as conn:
            _run_sql_file(conn, SQL_DIR / "01_create_tables.sql")
            conn.execute(text("TRUNCATE patient_encounters RESTART IDENTITY;"))
        df.to_sql(
            "patient_encounters",
            engine,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=5000,
        )
        with engine.begin() as conn:
            _run_sql_file(conn, SQL_DIR / "03_feature_views.sql")
        print(f"Loaded {len(df)} rows into patient_encounters.")
        return True
    except Exception as exc:
        print(f"Postgres load skipped ({exc.__class__.__name__}: {exc})")
        return False


def run_sql_cohort(engine) -> None:
    sql_text = (SQL_DIR / "02_cohort_analysis.sql").read_text(encoding="utf-8")
    statements = [s.strip() for s in sql_text.split(";") if s.strip() and not s.strip().startswith("--")]
    names = [
        "cohort_by_diagnosis.csv",
        "cohort_by_payer.csv",
        "cohort_by_los.csv",
        "cohort_by_inpatient.csv",
    ]
    for name, stmt in zip(names, statements):
        table = pd.read_sql(text(stmt), engine)
        table.to_csv(PROCESSED / name, index=False)
        print(f"\n=== SQL {name} ===")
        print(table.to_string(index=False))


def run() -> None:
    df = encounters_frame(load_cleaned())
    cohort_csvs_from_df(df)
    if load_postgres(df):
        run_sql_cohort(get_engine())


if __name__ == "__main__":
    run()
