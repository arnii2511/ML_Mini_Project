from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from anomaly_sms.inference import build_demo_risk_examples, train_demo_system
from anomaly_sms.pipeline import PipelineConfig


DEFAULT_DATASET_CANDIDATES = [
    Path("data/raw/SMSSpamCollection"),
    Path("data/raw/sms_spam.csv"),
    Path("data/raw/SMSSpamCollection.csv"),
]

SAMPLE_MESSAGES = [
    "Hey, I will reach by 6 pm. Please wait near the gate.",
    "URGENT: verify your UPI profile immediately to avoid payment block click http://alert.secure-check.example",
    "Congratulations! You won a crypto reward, claim now with OTP 482991.",
]


def _default_dataset_path() -> str:
    for candidate in DEFAULT_DATASET_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return str(DEFAULT_DATASET_CANDIDATES[0])


@st.cache_resource(show_spinner=True)
def load_demo_system(
    data_path: str,
    train_strategy: str,
    max_features: int,
    contamination: float,
    supervised_spam_fraction: float,
) -> object:
    config = PipelineConfig(
        data_path=data_path,
        train_strategy=train_strategy,
        max_features=max_features,
        contamination=contamination,
        supervised_spam_fraction=supervised_spam_fraction,
        synthetic_spam_count=0,
    )
    return train_demo_system(config)


def _format_binary_label(value: int, anomaly_text: str, normal_text: str) -> str:
    return anomaly_text if int(value) == 1 else normal_text


def _summary_value(summary: dict[str, object], key: str, fallback: object = 0) -> object:
    return summary.get(key, fallback)


def main() -> None:
    st.set_page_config(page_title="SMS Risk Checker", layout="wide")
    st.title("SMS Risk Checker")
    st.caption("Train on SMS data first, then check how risky a new message looks.")

    with st.sidebar:
        st.subheader("Train Model")
        data_path = st.text_input("Dataset path", value=_default_dataset_path())
        train_strategy = st.selectbox("Train anomaly model on", options=["ham_only", "all"], index=0)
        max_features = st.slider("TF-IDF features", min_value=300, max_value=3000, value=1500, step=100)
        contamination = st.slider("Expected anomaly ratio", min_value=0.01, max_value=0.30, value=0.12, step=0.01)
        supervised_spam_fraction = st.slider(
            "Spam kept for supervised comparison",
            min_value=0.10,
            max_value=1.00,
            value=1.00,
            step=0.05,
        )
        train_now = st.button("Train Model", type="primary", use_container_width=True)

    st.markdown(
        """
        **What this app does**

        1. It learns from your SMS dataset.
        2. It understands what normal messages usually look like.
        3. You can then paste a new message and see its risk level.
        """
    )

    if train_now:
        try:
            demo = load_demo_system(
                data_path=data_path,
                train_strategy=train_strategy,
                max_features=max_features,
                contamination=contamination,
                supervised_spam_fraction=supervised_spam_fraction,
            )
            st.session_state["demo_model"] = demo
            st.session_state["risk_examples"] = build_demo_risk_examples(demo, count=40)
        except Exception as exc:
            st.error(f"Could not train the model: {exc}")
            st.info("Place the SMS Spam Collection dataset in data/raw/ and verify the dataset path.")
            return

    demo = st.session_state.get("demo_model")
    risk_examples = st.session_state.get("risk_examples")

    if demo is None:
        st.warning("Train the model first from the left panel.")
        return

    summary = demo.training_summary
    st.success("Model training complete.")

    st.subheader("What Was Used For Training")
    st.caption("Important: `ham_only` applies only to the anomaly models. Supervised comparison models still use spam messages unless you reduce the spam fraction slider.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total messages", summary["total_messages"])
    col2.metric("Train split size", summary["train_messages"])
    col3.metric("Test split size", summary["test_messages"])

    left_info, right_info = st.columns(2)
    with left_info:
        st.markdown("**Anomaly model training**")
        st.write(f"- Mode: `{_summary_value(summary, 'unsupervised_training', 'unknown')}`")
        st.write(f"- Messages used: {_summary_value(summary, 'anomaly_train_messages', 'please retrain')}")
        st.write(f"- Ham used: {_summary_value(summary, 'anomaly_train_ham', 'please retrain')}")
        st.write(f"- Spam used: {_summary_value(summary, 'anomaly_train_spam', 'please retrain')}")

    with right_info:
        st.markdown("**Supervised comparison training**")
        st.write(f"- Messages used: {_summary_value(summary, 'supervised_train_messages', 'please retrain')}")
        st.write(f"- Ham used: {_summary_value(summary, 'supervised_train_ham', 'please retrain')}")
        st.write(f"- Spam used: {_summary_value(summary, 'supervised_train_spam', 'please retrain')}")
        spam_fraction = float(_summary_value(summary, "supervised_spam_fraction", 0.0))
        st.write(f"- Spam fraction setting: {spam_fraction:.2f}")

    st.subheader("How Well The Models Performed")
    st.caption("These scores are on the held-out test set from your dataset.")

    if hasattr(demo, "performance_metrics"):
        metrics = demo.performance_metrics.copy()
        metrics["model_name"] = metrics["model"].replace(
            {
                "isolation_forest": "Isolation Forest",
                "one_class_svm": "One-Class SVM",
                "lof": "LOF",
                "ensemble_prediction": "Ensemble",
                "logistic_regression": "Logistic Regression",
                "multinomial_nb": "Naive Bayes",
            }
        )
        chart_df = metrics.set_index("model_name")[["accuracy", "f1_score"]]
        st.bar_chart(chart_df)
        st.dataframe(
            metrics[["model_name", "model_family", "accuracy", "precision", "recall", "f1_score"]].rename(
                columns={
                    "model_name": "Model",
                    "model_family": "Type",
                    "accuracy": "Accuracy",
                    "precision": "Precision",
                    "recall": "Recall",
                    "f1_score": "F1 Score",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("This model was trained with an older app state. Click `Train Model` once to refresh the statistics.")

    st.subheader("Try Your Own Message")
    selected_sample = st.selectbox("Quick sample", options=["Custom"] + SAMPLE_MESSAGES, index=0)
    default_text = "" if selected_sample == "Custom" else selected_sample
    text = st.text_area("Enter a new SMS", value=default_text, height=140, placeholder="Paste a message here...")

    if st.button("Check Risk", use_container_width=True):
        if not text.strip():
            st.warning("Enter a message first.")
        else:
            result = demo.predict_message(text)
            risk_percent = result["risk_percent"]
            risk_level = result["risk_level"]
            progress_value = min(max(int(round(risk_percent)), 0), 100)

            st.progress(progress_value)
            st.subheader(f"Risk: {risk_percent:.0f}% ({risk_level})")

            if result["ensemble_prediction"] == 1:
                st.error("This message looks suspicious.")
            else:
                st.success("This message does not look strongly suspicious.")

            st.write("Reason:", result["heuristic_reason"])

            simple_votes = pd.DataFrame(
                [
                    {
                        "Model": "Isolation Forest",
                        "Decision": _format_binary_label(
                            result["unsupervised_predictions"]["isolation_forest"],
                            "Suspicious",
                            "Normal",
                        ),
                    },
                    {
                        "Model": "One-Class SVM",
                        "Decision": _format_binary_label(
                            result["unsupervised_predictions"]["one_class_svm"],
                            "Suspicious",
                            "Normal",
                        ),
                    },
                    {
                        "Model": "LOF",
                        "Decision": _format_binary_label(
                            result["unsupervised_predictions"]["lof"],
                            "Suspicious",
                            "Normal",
                        ),
                    },
                ]
            )
            st.dataframe(simple_votes, use_container_width=True, hide_index=True)

            with st.expander("More details"):
                supervised_frame = pd.DataFrame(
                    [
                        {
                            "Model": "Logistic Regression",
                            "Prediction": _format_binary_label(
                                result["supervised_predictions"]["logistic_regression"],
                                "Spam",
                                "Ham",
                            ),
                            "Score": round(
                                float(result["supervised_scores"]["logistic_regression_score"]) * 100,
                                2,
                            ),
                        },
                        {
                            "Model": "Naive Bayes",
                            "Prediction": _format_binary_label(
                                result["supervised_predictions"]["multinomial_nb"],
                                "Spam",
                                "Ham",
                            ),
                            "Score": round(
                                float(result["supervised_scores"]["multinomial_nb_score"]) * 100,
                                2,
                            ),
                        },
                    ]
                )
                st.dataframe(supervised_frame, use_container_width=True, hide_index=True)

                feature_frame = pd.DataFrame(
                    [{"Feature": key, "Value": value} for key, value in result["manual_features"].items()]
                )
                st.dataframe(feature_frame, use_container_width=True, hide_index=True)

    st.subheader("40 Example Suspicious Messages")
    st.caption("These are generated examples. The table shows how risky the trained anomaly model thinks each one is.")

    if risk_examples is not None:
        display_examples = risk_examples.rename(
            columns={
                "message": "Message",
                "risk_percent": "Risk %",
                "risk_level": "Risk level",
                "anomaly_votes": "Model votes",
            }
        )
        st.dataframe(
            display_examples[["Message", "Risk %", "Risk level", "Model votes"]],
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
