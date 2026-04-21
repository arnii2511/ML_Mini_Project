from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


DEFAULT_PREDICTION_COLUMNS = [
    "isolation_forest",
    "one_class_svm",
    "lof",
    "ensemble_prediction",
    "logistic_regression",
    "multinomial_nb",
]

MODEL_FAMILY = {
    "isolation_forest": "unsupervised",
    "one_class_svm": "unsupervised",
    "lof": "unsupervised",
    "ensemble_prediction": "ensemble",
    "logistic_regression": "supervised",
    "multinomial_nb": "supervised",
}


def evaluate_predictions(
    y_true: pd.Series,
    prediction_frame: pd.DataFrame,
    scenario: str,
    prediction_columns: list[str] | None = None,
) -> pd.DataFrame:
    columns = prediction_columns or [
        column for column in DEFAULT_PREDICTION_COLUMNS if column in prediction_frame.columns
    ]
    if len(y_true) == 0:
        return pd.DataFrame(
            columns=[
                "scenario",
                "model",
                "model_family",
                "precision",
                "recall",
                "f1_score",
                "accuracy",
                "flagged_fraction",
            ]
        )

    metrics = []
    for column in columns:
        y_pred = prediction_frame[column]
        metrics.append(
            {
                "scenario": scenario,
                "model": column,
                "model_family": MODEL_FAMILY.get(column, "other"),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1_score": f1_score(y_true, y_pred, zero_division=0),
                "accuracy": accuracy_score(y_true, y_pred),
                "flagged_fraction": float(np.mean(y_pred)),
            }
        )
    return (
        pd.DataFrame(metrics)
        .sort_values(by=["scenario", "f1_score"], ascending=[True, False])
        .reset_index(drop=True)
    )


def summarize_agreement(predictions: pd.DataFrame) -> pd.DataFrame:
    summary = predictions.copy()
    summary["votes_for_anomaly"] = (
        summary["isolation_forest"] + summary["one_class_svm"] + summary["lof"]
    )

    grouped = (
        summary.groupby(["votes_for_anomaly", "label"])
        .size()
        .reset_index(name="count")
        .sort_values(["votes_for_anomaly", "label"])
        .reset_index(drop=True)
    )
    return grouped


def heuristic_explanation(row: pd.Series, feature_builder) -> str:
    signals = []
    for feature_name in feature_builder.manual_feature_names_:
        std = float(feature_builder.reference_stds_[feature_name]) or 1.0
        z_score = (float(row[feature_name]) - float(feature_builder.reference_means_[feature_name])) / std
        if z_score >= 1.5:
            pretty_name = feature_name.replace("_", " ")
            signals.append(f"high {pretty_name} (z={z_score:.1f})")

    rare_terms = feature_builder.top_rare_terms(row["clean_text"])
    if rare_terms:
        signals.append("rare terms: " + ", ".join(rare_terms))

    if not signals:
        return "Weak structural anomaly; message may mainly differ in TF-IDF distribution."
    return "; ".join(signals[:4])


def build_case_studies(predictions: pd.DataFrame, feature_builder, max_cases: int = 10) -> pd.DataFrame:
    frame = predictions.copy()
    frame["votes_for_anomaly"] = (
        frame["isolation_forest"] + frame["one_class_svm"] + frame["lof"]
    )
    frame["category"] = "other"
    frame.loc[frame["votes_for_anomaly"] == 3, "category"] = "all_agree_anomaly"
    frame.loc[frame["votes_for_anomaly"] == 0, "category"] = "all_agree_normal"
    frame.loc[frame["votes_for_anomaly"].isin([1, 2]), "category"] = "model_disagreement"

    selected = pd.concat(
        [
            frame[frame["category"] == "model_disagreement"].head(max_cases // 2),
            frame[frame["category"] == "all_agree_anomaly"].head(max_cases // 4),
            frame[frame["category"] == "all_agree_normal"].head(max_cases // 4),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["text"])

    if selected.empty:
        return selected

    selected["heuristic_reason"] = selected.apply(
        lambda row: heuristic_explanation(row, feature_builder),
        axis=1,
    )

    ordered_columns = [
        "category",
        "source",
        "label",
        "split",
        "text",
        "agreement_score",
        "isolation_forest",
        "one_class_svm",
        "lof",
        "ensemble_prediction",
        "logistic_regression",
        "multinomial_nb",
        "heuristic_reason",
    ]
    available_columns = [column for column in ordered_columns if column in selected.columns]
    return selected[available_columns].head(max_cases).reset_index(drop=True)


def save_run_config(path: Path, config: Any) -> None:
    payload = asdict(config) if is_dataclass(config) else dict(config)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
