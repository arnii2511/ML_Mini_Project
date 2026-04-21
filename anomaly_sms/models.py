from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


@dataclass
class AnomalyModelSuite:
    contamination: float = 0.12
    random_state: int = 42
    nu: float = 0.12
    n_neighbors: int = 35
    models: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        self.models = {
            "isolation_forest": IsolationForest(
                n_estimators=300,
                contamination=self.contamination,
                random_state=self.random_state,
                n_jobs=1,
            ),
            "one_class_svm": OneClassSVM(
                kernel="rbf",
                gamma="scale",
                nu=self.nu,
            ),
            "lof": LocalOutlierFactor(
                n_neighbors=self.n_neighbors,
                contamination=self.contamination,
                novelty=True,
                n_jobs=1,
            ),
        }

    def fit(self, features) -> "AnomalyModelSuite":
        sample_count = len(features)
        if sample_count <= 2:
            raise ValueError("At least three training samples are required to fit the models.")

        lof_neighbors = max(2, min(self.n_neighbors, sample_count - 1))
        self.models["lof"].set_params(n_neighbors=lof_neighbors)

        for model in self.models.values():
            model.fit(features)
        return self

    def predict(self, features) -> pd.DataFrame:
        outputs = {}
        for name, model in self.models.items():
            outputs[name] = (model.predict(features) == -1).astype(int)
        return pd.DataFrame(outputs)

    def decision_scores(self, features) -> pd.DataFrame:
        scores = {}
        for name, model in self.models.items():
            scores[f"{name}_score"] = -model.decision_function(features)
        return pd.DataFrame(scores)


@dataclass
class SupervisedModelSuite:
    random_state: int = 42
    logistic_regression: LogisticRegression = field(init=False)
    multinomial_nb: MultinomialNB = field(init=False)

    def __post_init__(self) -> None:
        self.logistic_regression = LogisticRegression(
            solver="liblinear",
            max_iter=2000,
            class_weight="balanced",
            random_state=self.random_state,
        )
        self.multinomial_nb = MultinomialNB(alpha=1.0)

    def fit(self, combined_features, text_features, labels) -> "SupervisedModelSuite":
        unique_labels = pd.Series(labels).nunique()
        if unique_labels < 2:
            raise ValueError("Supervised models require both ham and spam labels in training.")

        self.logistic_regression.fit(combined_features, labels)
        self.multinomial_nb.fit(text_features, labels)
        return self

    def predict(self, combined_features, text_features) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "logistic_regression": self.logistic_regression.predict(combined_features),
                "multinomial_nb": self.multinomial_nb.predict(text_features),
            }
        )

    def probability_scores(self, combined_features, text_features) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "logistic_regression_score": self.logistic_regression.predict_proba(
                    combined_features
                )[:, 1],
                "multinomial_nb_score": self.multinomial_nb.predict_proba(text_features)[:, 1],
            }
        )
