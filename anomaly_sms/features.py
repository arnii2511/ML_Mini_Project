from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler, StandardScaler


URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
SPECIAL_CHAR_PATTERN = re.compile(r"[!$%&*@#?]")
NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9\s]")
MULTISPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = str(text)
    lowered = text.lower()
    lowered = URL_PATTERN.sub(" urltoken ", lowered)
    lowered = NON_ALNUM_PATTERN.sub(" ", lowered)
    lowered = MULTISPACE_PATTERN.sub(" ", lowered).strip()
    return lowered


def extract_structural_features(texts: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for text in texts:
        value = str(text)
        alphabetic_count = max(sum(char.isalpha() for char in value), 1)
        uppercase_count = sum(char.isupper() for char in value)
        rows.append(
            {
                "message_length": float(len(value)),
                "word_count": float(len(value.split())),
                "digit_count": float(sum(char.isdigit() for char in value)),
                "url_count": float(len(URL_PATTERN.findall(value))),
                "special_char_count": float(len(SPECIAL_CHAR_PATTERN.findall(value))),
                "uppercase_count": float(uppercase_count),
                "uppercase_ratio": float(uppercase_count / alphabetic_count),
            }
        )
    return pd.DataFrame(rows)


@dataclass
class SMSFeatureBuilder:
    max_features: int = 1500
    min_df: int = 2
    stop_words: str | None = "english"
    ngram_range: tuple[int, int] = (1, 2)
    vectorizer: TfidfVectorizer = field(init=False)
    manual_scaler: StandardScaler = field(init=False)
    final_scaler: MaxAbsScaler = field(init=False)
    manual_feature_names_: list[str] = field(init=False, default_factory=list)
    reference_means_: pd.Series = field(init=False)
    reference_stds_: pd.Series = field(init=False)

    def __post_init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            stop_words=self.stop_words,
            ngram_range=self.ngram_range,
            sublinear_tf=True,
        )
        self.manual_scaler = StandardScaler()
        self.final_scaler = MaxAbsScaler()

    def fit(self, raw_texts: Iterable[str], clean_texts: Iterable[str]) -> "SMSFeatureBuilder":
        manual_df = extract_structural_features(raw_texts)
        tfidf_matrix = self.vectorizer.fit_transform(clean_texts)

        self.manual_scaler.fit(manual_df)
        manual_scaled = self.manual_scaler.transform(manual_df).astype(np.float32)
        combined = sparse.hstack(
            [tfidf_matrix.astype(np.float32), sparse.csr_matrix(manual_scaled)],
            format="csr",
        )
        self.final_scaler.fit(combined)

        self.manual_feature_names_ = manual_df.columns.tolist()
        self.reference_means_ = manual_df.mean()
        stds = manual_df.std().replace(0, 1.0)
        self.reference_stds_ = stds.fillna(1.0)
        return self

    def transform(self, raw_texts: Iterable[str], clean_texts: Iterable[str]) -> np.ndarray:
        scaled = self.transform_sparse(raw_texts, clean_texts)
        return scaled.toarray().astype(np.float32)

    def transform_sparse(self, raw_texts: Iterable[str], clean_texts: Iterable[str]):
        manual_df = extract_structural_features(raw_texts)
        manual_scaled = self.manual_scaler.transform(manual_df).astype(np.float32)
        tfidf_matrix = self.transform_text(clean_texts)
        combined = sparse.hstack(
            [tfidf_matrix, sparse.csr_matrix(manual_scaled)],
            format="csr",
        )
        return self.final_scaler.transform(combined)

    def manual_features(self, raw_texts: Iterable[str]) -> pd.DataFrame:
        return extract_structural_features(raw_texts)

    def transform_text(self, clean_texts: Iterable[str]):
        return self.vectorizer.transform(clean_texts).astype(np.float32)

    def top_rare_terms(self, clean_text: str, top_k: int = 3) -> list[str]:
        analyzer = self.vectorizer.build_analyzer()
        tokens = analyzer(clean_text)
        if not tokens:
            return []

        vocabulary = self.vectorizer.vocabulary_
        idf_values = self.vectorizer.idf_
        unique_tokens = []
        seen = set()
        for token in tokens:
            if token in vocabulary and token not in seen:
                seen.add(token)
                unique_tokens.append(token)

        ranked = sorted(
            unique_tokens,
            key=lambda token: idf_values[vocabulary[token]],
            reverse=True,
        )
        return ranked[:top_k]
