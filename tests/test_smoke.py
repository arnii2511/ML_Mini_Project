from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from anomaly_sms.inference import train_demo_system
from anomaly_sms.pipeline import PipelineConfig, run_pipeline


def _build_synthetic_dataset() -> pd.DataFrame:
    ham_messages = [
        "Hey are we meeting after class today",
        "Please call me when you arrive home",
        "I will send the report by evening",
        "Lunch at 1 pm works for me",
        "Can you pick up groceries on the way",
        "Your appointment is confirmed for tomorrow",
        "Let us review the slides before the demo",
        "Happy birthday hope you have a great day",
        "The cab will reach in ten minutes",
        "Do not forget the meeting notes",
        "I am outside the station now",
        "Thanks for helping with the assignment",
        "We can reschedule to next week",
        "The package has arrived safely",
        "See you at the library after lunch",
        "Dinner is ready come downstairs",
        "Your OTP for login is 483921",
        "Please update the spreadsheet tonight",
        "The train is delayed by fifteen minutes",
        "Send me the venue once you reach",
    ]

    spam_messages = [
        "WIN CASH NOW!!! Click http://free-prize.example and claim 5000 today!!!",
        "URGENT you have won a FREE holiday call 999999 right now",
        "Exclusive offer buy now get 90 percent off visit www.fake-deals.example",
        "Congratulations claim your reward by sending BANK details immediately",
        "You are selected for a lucky draw click http://spam.example to confirm",
        "FREE entry in contest text WIN to 80808 now",
        "Lowest loan rates guaranteed apply today $$$",
        "Act fast your account will close verify at http://phish.example",
    ]

    rows = [{"label": "ham", "message": text} for text in ham_messages]
    rows.extend({"label": "spam", "message": text} for text in spam_messages)
    return pd.DataFrame(rows)


class PipelineSmokeTest(unittest.TestCase):
    def test_pipeline_generates_predictions_and_metrics(self) -> None:
        dataset = _build_synthetic_dataset()
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "sms.csv"
            dataset.to_csv(dataset_path, index=False)

            config = PipelineConfig(
                data_path=str(dataset_path),
                output_dir=str(Path(temp_dir) / "outputs"),
                max_features=300,
                min_df=1,
                contamination=0.20,
                nu=0.20,
                n_neighbors=5,
                synthetic_spam_count=6,
                test_size=0.25,
                max_case_studies=6,
            )
            artifacts = run_pipeline(config)

            self.assertFalse(artifacts.metrics.empty)
            self.assertIn("ensemble_prediction", artifacts.predictions.columns)
            self.assertIn("agreement_score", artifacts.predictions.columns)
            self.assertIn("logistic_regression", artifacts.predictions.columns)
            self.assertIn("multinomial_nb", artifacts.predictions.columns)
            self.assertTrue(
                set(artifacts.metrics["model"])
                >= {"ensemble_prediction", "lof", "logistic_regression", "multinomial_nb"}
            )
            self.assertIn("synthetic_shift", set(artifacts.metrics["scenario"]))
            self.assertIn("synthetic_shift", set(artifacts.predictions["source"]))

    def test_demo_predictor_returns_live_message_output(self) -> None:
        dataset = _build_synthetic_dataset()
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "sms.csv"
            dataset.to_csv(dataset_path, index=False)

            demo = train_demo_system(
                PipelineConfig(
                    data_path=str(dataset_path),
                    output_dir=str(Path(temp_dir) / "outputs"),
                    max_features=300,
                    min_df=1,
                    contamination=0.20,
                    nu=0.20,
                    n_neighbors=5,
                    synthetic_spam_count=4,
                    test_size=0.25,
                    max_case_studies=6,
                )
            )

            result = demo.predict_message("URGENT verify your UPI profile now at http://fraud.example")

            self.assertIn("agreement_score", result)
            self.assertIn("ensemble_prediction", result)
            self.assertIn("supervised_predictions", result)
            self.assertIn("unsupervised_predictions", result)
            self.assertIn("heuristic_reason", result)


if __name__ == "__main__":
    unittest.main()
