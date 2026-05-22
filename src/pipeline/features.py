from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .config import VectorizerConfig


class FeatureExtractionError(Exception):
    pass


def build_vectorizer(cfg: VectorizerConfig) -> Any:
    kwargs = dict(
        ngram_range=cfg.ngram_range,
        max_features=cfg.max_features,
        min_df=cfg.min_df,
    )
    if cfg.type == "tfidf":
        return TfidfVectorizer(**kwargs)
    if cfg.type == "count":
        return CountVectorizer(**kwargs)
    # validate_config should have caught this; defensive
    raise FeatureExtractionError(f"unknown vectorizer type: {cfg.type}")


def fit_vectorizer(cfg: VectorizerConfig, train_texts: list[str]) -> Any:
    vec = build_vectorizer(cfg)
    try:
        vec.fit(train_texts)
    except ValueError as e:
        # Most commonly: "empty vocabulary; perhaps the documents only contain stop words"
        raise FeatureExtractionError(
            f"vectorizer fit failed (vocabulary may be empty after applying min_df={cfg.min_df}): {e}. "
            f"Hint: lower min_df to 1 in config.json."
        ) from e
    # Defensive: explicitly check empty vocabulary
    if not getattr(vec, "vocabulary_", None):
        raise FeatureExtractionError(
            f"vectorizer produced an empty vocabulary (min_df={cfg.min_df}, "
            f"ngram_range={cfg.ngram_range}). Lower min_df to 1 in config.json."
        )
    return vec
