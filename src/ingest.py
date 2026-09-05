"""Download (if needed), clean the UCI diabetes dataset, and save parquet."""

from pathlib import Path
import zipfile
from io import BytesIO

import numpy as np
import pandas as pd
import requests

from src import ROOT

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip"
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
CSV_PATH = RAW_DIR / "diabetic_data.csv"
MAPPING_PATH = RAW_DIR / "IDS_mapping.csv"
# Kaggle/UCI zips sometimes use this filename
MAPPING_PATH_ALT = RAW_DIR / "IDs_mapping.csv"

DROP_COLS = ["weight", "encounter_id", "patient_nbr", "examide", "citoglipton"]
HOSPICE_OR_EXPIRED_IDS = [11, 13, 14, 19, 20, 21]


def download_raw() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if CSV_PATH.exists():
        print(f"Raw CSV already present at {CSV_PATH} — skipping download.")
        return

    print(f"Downloading dataset from {UCI_URL}")
    response = requests.get(UCI_URL, timeout=120)
    response.raise_for_status()
    with zipfile.ZipFile(BytesIO(response.content)) as zf:
        zf.extractall(RAW_DIR)
    print(f"Extracted files to {RAW_DIR}")


def _mapping_file() -> Path:
    if MAPPING_PATH.exists():
        return MAPPING_PATH
    if MAPPING_PATH_ALT.exists():
        return MAPPING_PATH_ALT
    raise FileNotFoundError("IDS_mapping.csv not found in data/raw/")


def parse_id_mapping(path: Path) -> dict[str, dict[int, str]]:
    """Parse the sectioned IDs_mapping.csv into {id_column: {id: label}}."""
    maps: dict[str, dict[int, str]] = {}
    current_key = None
    current: dict[int, str] = {}

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line == ",":
                continue
            if line.endswith(",description"):
                if current_key is not None:
                    maps[current_key] = current
                current_key = line.split(",")[0]
                current = {}
                continue
            parts = line.split(",", 1)
            if current_key and parts[0].isdigit():
                current[int(parts[0])] = parts[1].strip().strip('"')

    if current_key is not None:
        maps[current_key] = current
    return maps


def add_id_labels(df: pd.DataFrame) -> pd.DataFrame:
    maps = parse_id_mapping(_mapping_file())
    df = df.copy()
    df["admission_type_label"] = df["admission_type_id"].map(maps.get("admission_type_id", {}))
    df["discharge_disposition_label"] = df["discharge_disposition_id"].map(
        maps.get("discharge_disposition_id", {})
    )
    df["admission_source_label"] = df["admission_source_id"].map(
        maps.get("admission_source_id", {})
    )
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # a) one encounter per patient — earliest visit only (avoids leakage)
    df = df.sort_values("encounter_id", ascending=True)
    df = df.drop_duplicates(subset="patient_nbr", keep="first")

    # b) drop unused / ID / zero-variance columns
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # c) missing values are coded as '?'
    df = df.replace("?", np.nan)

    # d) invalid gender
    df = df[df["gender"] != "Unknown/Invalid"]

    # e) expired / hospice — cannot be readmitted
    df = df[~df["discharge_disposition_id"].isin(HOSPICE_OR_EXPIRED_IDS)]

    # f) binary 30-day readmission target
    df["readmitted_binary"] = (df["readmitted"] == "<30").astype(int)
    df = df.drop(columns=["readmitted"])

    df = df.reset_index(drop=True)
    return df


def run() -> pd.DataFrame:
    download_raw()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH, low_memory=False)
    print(f"Loaded raw data: {df.shape}")

    df = clean(df)
    df = add_id_labels(df)

    out_path = PROCESSED_DIR / "cleaned.parquet"
    df.to_parquet(out_path, index=False)
    pos_rate = df["readmitted_binary"].mean()
    print(f"Cleaned data: {df.shape}  positive rate={pos_rate:.4f}")
    print(f"Saved {out_path}")
    return df


if __name__ == "__main__":
    run()
