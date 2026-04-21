from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from anomaly_sms.data import load_sms_dataset
from anomaly_sms.features import SMSFeatureBuilder, clean_text
from anomaly_sms.models import AnomalyModelSuite, SupervisedModelSuite
from anomaly_sms.reporting import build_case_studies, evaluate_predictions, summarize_agreement
from anomaly_sms.shift import build_synthetic_shift_frame


@dataclass
class PipelineConfig:
    data_path: str
    output_dir: str = "outputs"
    train_strategy: str = "ham_only"
    supervised_spam_fraction: float = 1.0
    synthetic_spam_count: int = 24
    test_size: float = 0.30
    random_state: int = 42
    max_features: int = 1500
    min_df: int = 2
    contamination: float = 0.12
    nu: float = 0.12
    n_neighbors: int = 35
    ensemble_threshold: float = 2 / 3
    stop_words: str | None = "english"
    max_case_studies: int = 10


@dataclass
class PipelineArtifacts:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    agreement_summary: pd.DataFrame
    case_studies: pd.DataFrame


def _assign_splits(df: pd.DataFrame, test_size: float, random_state: int) -> pd.DataFrame:
    indices = df.index
    try:
        train_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=df["label"],
        )
    except ValueError:
        train_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
        )

    df = df.copy()
    df["split"] = "train"
    df.loc[test_indices, "split"] = "test"
    return df


def _limit_supervised_training_spam(
    train_df: pd.DataFrame,
    spam_fraction: float,
    random_state: int,
) -> pd.DataFrame:
    if not 0 < spam_fraction <= 1:
        raise ValueError("supervised_spam_fraction must be in the range (0, 1].")

    ham_df = train_df[train_df["label"] == "ham"]
    spam_df = train_df[train_df["label"] == "spam"]
    if spam_df.empty or spam_fraction >= 1.0:
        return train_df.copy()

    sample_size = max(1, int(round(len(spam_df) * spam_fraction)))
    sampled_spam = spam_df.sample(n=sample_size, random_state=random_state)
    limited_df = pd.concat([ham_df, sampled_spam], axis=0)
    return limited_df.sample(frac=1.0, random_state=random_state)


def run_pipeline(config: PipelineConfig) -> PipelineArtifacts:
    df = load_sms_dataset(config.data_path).reset_index(drop=True)
    df["clean_text"] = df["text"].map(clean_text)
    df = _assign_splits(df, test_size=config.test_size, random_state=config.random_state)
    df["source"] = "original"

    synthetic_shift_df = build_synthetic_shift_frame(config.synthetic_spam_count)
    if not synthetic_shift_df.empty:
        synthetic_shift_df["clean_text"] = synthetic_shift_df["text"].map(clean_text)
        df = pd.concat([df, synthetic_shift_df], ignore_index=True)

    fit_mask = df["split"] == "train"
    if config.train_strategy == "ham_only":
        fit_mask = fit_mask & (df["label"] == "ham")
    elif config.train_strategy != "all":
        raise ValueError("train_strategy must be 'ham_only' or 'all'.")

    feature_fit_df = df[(df["split"] == "train") & (df["source"] == "original")]
    fit_df = df.loc[fit_mask]
    if fit_df.empty:
        raise ValueError("No training rows available for fitting the unsupervised models.")
    if feature_fit_df.empty:
        raise ValueError("No original training rows available to fit the feature extractor.")

    feature_builder = SMSFeatureBuilder(
        max_features=config.max_features,
        min_df=config.min_df,
        stop_words=config.stop_words,
    )
    feature_builder.fit(feature_fit_df["text"], feature_fit_df["clean_text"])

    combined_sparse_matrix = feature_builder.transform_sparse(df["text"], df["clean_text"])
    feature_matrix = combined_sparse_matrix.toarray()
    text_matrix = feature_builder.transform_text(df["clean_text"])
    fit_matrix = feature_matrix[fit_mask.to_numpy()]

    model_suite = AnomalyModelSuite(
        contamination=config.contamination,
        random_state=config.random_state,
        nu=config.nu,
        n_neighbors=config.n_neighbors,
    )
    model_suite.fit(fit_matrix)

    supervised_train_df = _limit_supervised_training_spam(
        feature_fit_df,
        spam_fraction=config.supervised_spam_fraction,
        random_state=config.random_state,
    )
    supervised_row_indices = supervised_train_df.index.to_list()
    supervised_combined = combined_sparse_matrix[supervised_row_indices]
    supervised_text = text_matrix[supervised_row_indices]
    supervised_labels = supervised_train_df["is_anomaly"].to_numpy()

    supervised_suite = SupervisedModelSuite(random_state=config.random_state)
    supervised_suite.fit(supervised_combined, supervised_text, supervised_labels)

    manual_features = feature_builder.manual_features(df["text"])
    predictions = model_suite.predict(feature_matrix)
    supervised_predictions = supervised_suite.predict(combined_sparse_matrix, text_matrix)
    scores = model_suite.decision_scores(feature_matrix)
    supervised_scores = supervised_suite.probability_scores(combined_sparse_matrix, text_matrix)

    results = pd.concat(
        [
            df.reset_index(drop=True),
            manual_features.reset_index(drop=True),
            predictions.reset_index(drop=True),
            supervised_predictions.reset_index(drop=True),
            scores.reset_index(drop=True),
            supervised_scores.reset_index(drop=True),
        ],
        axis=1,
    )
    results["agreement_score"] = results[
        ["isolation_forest", "one_class_svm", "lof"]
    ].mean(axis=1)
    results["ensemble_prediction"] = (
        results["agreement_score"] >= config.ensemble_threshold
    ).astype(int)

    original_test_results = results[
        (results["split"] == "test") & (results["source"] == "original")
    ].reset_index(drop=True)
    synthetic_test_results = results[results["source"] == "synthetic_shift"].reset_index(drop=True)
    shifted_test_results = results[results["split"] == "test"].reset_index(drop=True)

    metrics = pd.concat(
        [
            evaluate_predictions(
                original_test_results["is_anomaly"],
                original_test_results,
                scenario="original_test",
            ),
            evaluate_predictions(
                synthetic_test_results["is_anomaly"],
                synthetic_test_results,
                scenario="synthetic_shift",
            ),
            evaluate_predictions(
                shifted_test_results["is_anomaly"],
                shifted_test_results,
                scenario="shifted_test_all",
            ),
        ],
        ignore_index=True,
    )
    agreement_summary = summarize_agreement(results)
    case_studies = build_case_studies(
        shifted_test_results,
        feature_builder=feature_builder,
        max_cases=config.max_case_studies,
    )

    return PipelineArtifacts(
        predictions=results,
        metrics=metrics,
        agreement_summary=agreement_summary,
        case_studies=case_studies,
    )


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
