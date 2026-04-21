from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


LABEL_CANDIDATES = ("label", "labels", "class", "category", "target", "v1")
TEXT_CANDIDATES = ("message", "messages", "text", "sms", "body", "content", "v2")


def _read_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".csv"}:
        return pd.read_csv(path)
    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")

    for sep in ("\t", ","):
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] >= 2:
                return df
        except Exception:
            continue

    raise ValueError(f"Unable to parse dataset file: {path}")


def _find_column(columns: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    normalized = {column.lower().strip(): column for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def load_sms_dataset(path: str | Path) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    raw_df = _read_dataset(dataset_path)
    if raw_df.shape[1] < 2:
        raise ValueError("Dataset must contain at least two columns.")

    label_column = _find_column(raw_df.columns, LABEL_CANDIDATES) or raw_df.columns[0]
    text_column = _find_column(raw_df.columns, TEXT_CANDIDATES)

    if text_column is None:
        remaining = [column for column in raw_df.columns if column != label_column]
        text_column = remaining[0]

    df = raw_df[[label_column, text_column]].copy()
    df.columns = ["label", "text"]
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["text"] = df["text"].fillna("").astype(str).str.strip()

    df = df[df["text"] != ""].drop_duplicates().reset_index(drop=True)

    numeric_labels = set(df["label"].unique()).issubset({"0", "1"})
    if numeric_labels:
        df["label"] = df["label"].map({"0": "ham", "1": "spam"})

    valid_labels = {"ham", "spam"}
    if not set(df["label"].unique()).issubset(valid_labels):
        raise ValueError(
            "Unsupported labels found. Expected ham/spam or 0/1 style labels."
        )

    df["is_anomaly"] = (df["label"] == "spam").astype(int)
    return df

