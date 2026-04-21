from __future__ import annotations

import argparse

from anomaly_sms.pipeline import PipelineConfig, ensure_output_dir, run_pipeline
from anomaly_sms.reporting import save_run_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SMS spam detection with supervised, unsupervised, and ensemble comparison."
    )
    parser.add_argument("--data", required=True, help="Path to the SMS dataset file.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for saved artifacts.")
    parser.add_argument(
        "--train-strategy",
        default="ham_only",
        choices=["ham_only", "all"],
        help="Fit models on only ham training messages or on the full training split.",
    )
    parser.add_argument(
        "--supervised-spam-fraction",
        type=float,
        default=1.0,
        help="Fraction of spam labels kept in supervised training to simulate limited known attacks.",
    )
    parser.add_argument(
        "--synthetic-spam-count",
        type=int,
        default=24,
        help="Number of synthetic unseen spam messages added to the evaluation split.",
    )
    parser.add_argument("--test-size", type=float, default=0.30, help="Evaluation split size.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-features", type=int, default=1500, help="Maximum TF-IDF features.")
    parser.add_argument("--min-df", type=int, default=2, help="Minimum document frequency for TF-IDF.")
    parser.add_argument("--contamination", type=float, default=0.12, help="Expected anomaly fraction.")
    parser.add_argument("--nu", type=float, default=0.12, help="One-Class SVM nu parameter.")
    parser.add_argument("--n-neighbors", type=int, default=35, help="LOF neighborhood size.")
    parser.add_argument(
        "--ensemble-threshold",
        type=float,
        default=2 / 3,
        help="Agreement threshold for the final ensemble anomaly label.",
    )
    parser.add_argument(
        "--no-stopwords",
        action="store_true",
        help="Disable English stop-word filtering in TF-IDF.",
    )
    parser.add_argument("--max-case-studies", type=int, default=10, help="Number of analysis examples.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = PipelineConfig(
        data_path=args.data,
        output_dir=args.output_dir,
        train_strategy=args.train_strategy,
        supervised_spam_fraction=args.supervised_spam_fraction,
        synthetic_spam_count=args.synthetic_spam_count,
        test_size=args.test_size,
        random_state=args.random_state,
        max_features=args.max_features,
        min_df=args.min_df,
        contamination=args.contamination,
        nu=args.nu,
        n_neighbors=args.n_neighbors,
        ensemble_threshold=args.ensemble_threshold,
        stop_words=None if args.no_stopwords else "english",
        max_case_studies=args.max_case_studies,
    )

    output_dir = ensure_output_dir(config.output_dir)
    artifacts = run_pipeline(config)

    artifacts.metrics.to_csv(output_dir / "metrics.csv", index=False)
    artifacts.predictions.to_csv(output_dir / "predictions.csv", index=False)
    artifacts.agreement_summary.to_csv(output_dir / "agreement_summary.csv", index=False)
    artifacts.case_studies.to_csv(output_dir / "case_studies.csv", index=False)
    save_run_config(output_dir / "run_config.json", config)

    print("Saved run artifacts:")
    print(f"- {output_dir / 'metrics.csv'}")
    print(f"- {output_dir / 'predictions.csv'}")
    print(f"- {output_dir / 'agreement_summary.csv'}")
    print(f"- {output_dir / 'case_studies.csv'}")
    print()
    print(artifacts.metrics.to_string(index=False))


if __name__ == "__main__":
    main()
