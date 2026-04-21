from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from anomaly_sms.data import load_sms_dataset
from anomaly_sms.features import SMSFeatureBuilder, clean_text
from anomaly_sms.models import AnomalyModelSuite, SupervisedModelSuite
from anomaly_sms.pipeline import _assign_splits, _limit_supervised_training_spam, PipelineConfig
from anomaly_sms.reporting import evaluate_predictions
from anomaly_sms.shift import generate_synthetic_shift_messages


@dataclass
class TrainedSMSDemo:
    config: PipelineConfig
    feature_builder: SMSFeatureBuilder
    anomaly_suite: AnomalyModelSuite
    supervised_suite: SupervisedModelSuite
    training_summary: dict[str, object]
    performance_metrics: pd.DataFrame

    def predict_message(self, text: str) -> dict[str, object]:
        clean = clean_text(text)
        combined_sparse = self.feature_builder.transform_sparse([text], [clean])
        combined_dense = combined_sparse.toarray()
        text_sparse = self.feature_builder.transform_text([clean])

        unsupervised_predictions = self.anomaly_suite.predict(combined_dense).iloc[0]
        supervised_predictions = self.supervised_suite.predict(combined_sparse, text_sparse).iloc[0]
        unsupervised_scores = self.anomaly_suite.decision_scores(combined_dense).iloc[0]
        supervised_scores = self.supervised_suite.probability_scores(combined_sparse, text_sparse).iloc[0]
        manual_features = self.feature_builder.manual_features([text]).iloc[0]

        agreement_score = float(
            unsupervised_predictions[["isolation_forest", "one_class_svm", "lof"]].mean()
        )
        ensemble_prediction = int(agreement_score >= self.config.ensemble_threshold)

        explanation_row = pd.Series(
            {
                "text": text,
                "clean_text": clean,
                **manual_features.to_dict(),
            }
        )

        from anomaly_sms.reporting import heuristic_explanation

        known_spam_percent = round(
            (
                float(supervised_scores["logistic_regression_score"])
                + float(supervised_scores["multinomial_nb_score"])
            )
            / 2
            * 100,
            2,
        )
        new_pattern_percent = round(agreement_score * 100, 2)
        final_scam_percent = round(max(known_spam_percent, new_pattern_percent), 2)

        if known_spam_percent >= 70 and new_pattern_percent >= 67:
            final_reason = "Matches known spam wording and also looks suspicious in a new-pattern check."
        elif known_spam_percent >= 70:
            final_reason = "Matches known spam or scam wording strongly."
        elif new_pattern_percent >= 67:
            final_reason = "Does not match a common known pattern strongly, but still looks suspicious."
        elif known_spam_percent >= 40 or new_pattern_percent >= 33:
            final_reason = "Shows some suspicious signals. Handle carefully."
        else:
            final_reason = "Does not strongly match scam or spam behavior."

        return {
            "text": text,
            "clean_text": clean,
            "manual_features": manual_features.to_dict(),
            "unsupervised_predictions": unsupervised_predictions.to_dict(),
            "unsupervised_scores": unsupervised_scores.to_dict(),
            "supervised_predictions": supervised_predictions.to_dict(),
            "supervised_scores": supervised_scores.to_dict(),
            "agreement_score": agreement_score,
            "ensemble_prediction": ensemble_prediction,
            "heuristic_reason": heuristic_explanation(explanation_row, self.feature_builder),
            "risk_percent": new_pattern_percent,
            "risk_level": risk_level_from_score(agreement_score),
            "supervised_spam_percent": known_spam_percent,
            "known_spam_percent": known_spam_percent,
            "new_pattern_percent": new_pattern_percent,
            "final_scam_percent": final_scam_percent,
            "final_scam_label": final_scam_label(final_scam_percent),
            "final_reason": final_reason,
        }

    def score_messages(self, texts: list[str]) -> pd.DataFrame:
        rows = []
        for text in texts:
            result = self.predict_message(text)
            rows.append(
                {
                    "message": text,
                    "risk_percent": result["final_scam_percent"],
                    "risk_level": result["final_scam_label"],
                    "agreement_score": result["agreement_score"],
                    "anomaly_votes": sum(result["unsupervised_predictions"].values()),
                    "known_spam_percent": result["known_spam_percent"],
                    "new_pattern_percent": result["new_pattern_percent"],
                }
            )
        return pd.DataFrame(rows).sort_values(
            by=["risk_percent", "anomaly_votes"],
            ascending=[False, False],
        ).reset_index(drop=True)


def risk_level_from_score(score: float) -> str:
    if score >= 0.99:
        return "Very High"
    if score >= 2 / 3:
        return "High"
    if score >= 1 / 3:
        return "Medium"
    if score > 0:
        return "Low"
    return "Very Low"


def final_scam_label(score: float) -> str:
    if score >= 80:
        return "Very High"
    if score >= 60:
        return "High"
    if score >= 40:
        return "Medium"
    if score >= 20:
        return "Low"
    return "Very Low"


def train_demo_system(config: PipelineConfig) -> TrainedSMSDemo:
    df = load_sms_dataset(config.data_path).reset_index(drop=True)
    df["clean_text"] = df["text"].map(clean_text)
    df = _assign_splits(df, test_size=config.test_size, random_state=config.random_state)

    df["source"] = "original"

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
    dense_feature_matrix = combined_sparse_matrix.toarray()
    text_matrix = feature_builder.transform_text(df["clean_text"])
    fit_matrix = dense_feature_matrix[fit_mask.to_numpy()]

    anomaly_suite = AnomalyModelSuite(
        contamination=config.contamination,
        random_state=config.random_state,
        nu=config.nu,
        n_neighbors=config.n_neighbors,
    )
    anomaly_suite.fit(fit_matrix)

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

    unsupervised_predictions = anomaly_suite.predict(dense_feature_matrix)
    supervised_predictions = supervised_suite.predict(combined_sparse_matrix, text_matrix)
    results = pd.concat(
        [
            df.reset_index(drop=True),
            unsupervised_predictions.reset_index(drop=True),
            supervised_predictions.reset_index(drop=True),
        ],
        axis=1,
    )
    results["agreement_score"] = results[
        ["isolation_forest", "one_class_svm", "lof"]
    ].mean(axis=1)
    results["ensemble_prediction"] = (
        results["agreement_score"] >= config.ensemble_threshold
    ).astype(int)
    test_results = results[results["split"] == "test"].reset_index(drop=True)
    performance_metrics = evaluate_predictions(
        test_results["is_anomaly"],
        test_results,
        scenario="held_out_test",
    )

    training_summary = {
        "total_messages": int(len(df)),
        "train_messages": int(len(feature_fit_df)),
        "test_messages": int((df["split"] == "test").sum()),
        "anomaly_train_messages": int(len(fit_df)),
        "anomaly_train_ham": int((fit_df["label"] == "ham").sum()),
        "anomaly_train_spam": int((fit_df["label"] == "spam").sum()),
        "supervised_train_messages": int(len(supervised_train_df)),
        "supervised_train_ham": int((supervised_train_df["label"] == "ham").sum()),
        "supervised_train_spam": int((supervised_train_df["label"] == "spam").sum()),
        "tfidf_features": int(len(feature_builder.vectorizer.vocabulary_)),
        "unsupervised_training": config.train_strategy,
        "supervised_spam_fraction": config.supervised_spam_fraction,
    }

    return TrainedSMSDemo(
        config=config,
        feature_builder=feature_builder,
        anomaly_suite=anomaly_suite,
        supervised_suite=supervised_suite,
        training_summary=training_summary,
        performance_metrics=performance_metrics,
    )


def build_demo_risk_examples(demo: TrainedSMSDemo, count: int = 40) -> pd.DataFrame:
    examples = generate_synthetic_shift_messages(count)
    return demo.score_messages(examples)
