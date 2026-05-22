from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


VALID_SELECTION_METRICS = {"accuracy", "macro_precision", "macro_recall", "macro_f1"}
VALID_VECTORIZER_TYPES = {"tfidf", "count"}


class ConfigError(Exception):
    pass


@dataclass
class VectorizerConfig:
    type: str = "tfidf"
    ngram_range: tuple[int, int] = (1, 2)
    max_features: int | None = 5000
    min_df: int | float = 1


@dataclass
class CrossValidationConfig:
    enabled: bool = False
    folds: int = 5


@dataclass
class PipelineConfig:
    random_seed: int = 42
    validation_split: float = 0.2
    models: list[str] = field(default_factory=lambda: ["logistic_regression", "linear_svm", "naive_bayes"])
    vectorizer: VectorizerConfig = field(default_factory=VectorizerConfig)
    selection_metric: str = "macro_f1"
    top_k_error_examples: int = 10
    cross_validation: CrossValidationConfig = field(default_factory=CrossValidationConfig)

    @property
    def raw(self) -> dict[str, Any]:
        return {
            "random_seed": self.random_seed,
            "validation_split": self.validation_split,
            "models": list(self.models),
            "vectorizer": {
                "type": self.vectorizer.type,
                "ngram_range": list(self.vectorizer.ngram_range),
                "max_features": self.vectorizer.max_features,
                "min_df": self.vectorizer.min_df,
            },
            "selection_metric": self.selection_metric,
            "top_k_error_examples": self.top_k_error_examples,
            "cross_validation": {
                "enabled": self.cross_validation.enabled,
                "folds": self.cross_validation.folds,
            },
        }


def load_config(path: str | Path) -> PipelineConfig:
    """Load and validate config.json. Missing file is a hard failure (exit 2)."""
    p = Path(path)
    if not p.exists():
        raise ConfigError(
            f"config.json not found at '{p}'. The pipeline requires a configuration "
            f"file at this path; create one based on the project README."
        )
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ConfigError(f"config.json is not valid JSON: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError("config.json must contain a JSON object at the top level")

    cfg = PipelineConfig()

    if "random_seed" in raw:
        if not isinstance(raw["random_seed"], int) or isinstance(raw["random_seed"], bool):
            raise ConfigError("random_seed must be an integer")
        cfg.random_seed = raw["random_seed"]

    if "validation_split" in raw:
        vs = raw["validation_split"]
        if not isinstance(vs, (int, float)) or isinstance(vs, bool):
            raise ConfigError("validation_split must be a number")
        vs = float(vs)
        if not (0.0 < vs < 1.0):
            raise ConfigError(
                f"validation_split must satisfy 0.0 < x < 1.0; got {vs}. "
                f"The pipeline uses the configured fraction exactly — no silent rescaling."
            )
        cfg.validation_split = vs

    if "models" in raw:
        if not isinstance(raw["models"], list) or not all(isinstance(m, str) for m in raw["models"]):
            raise ConfigError("models must be a list of strings")
        seen: list[str] = []
        for m in raw["models"]:
            if m not in seen:
                seen.append(m)
        cfg.models = seen

    if "vectorizer" in raw:
        vraw = raw["vectorizer"]
        if not isinstance(vraw, dict):
            raise ConfigError("vectorizer must be an object")
        v = VectorizerConfig()
        if "type" in vraw:
            if vraw["type"] not in VALID_VECTORIZER_TYPES:
                raise ConfigError(
                    f"vectorizer.type must be one of {sorted(VALID_VECTORIZER_TYPES)}, got {vraw['type']!r}"
                )
            v.type = vraw["type"]
        if "ngram_range" in vraw:
            ng = vraw["ngram_range"]
            if (not isinstance(ng, list) or len(ng) != 2
                    or not all(isinstance(x, int) for x in ng)
                    or ng[0] < 1 or ng[1] < ng[0]):
                raise ConfigError("vectorizer.ngram_range must be [low, high] with 1 <= low <= high")
            v.ngram_range = (ng[0], ng[1])
        if "max_features" in vraw:
            mf = vraw["max_features"]
            if mf is not None and (not isinstance(mf, int) or mf < 1):
                raise ConfigError("vectorizer.max_features must be a positive integer or null")
            v.max_features = mf
        if "min_df" in vraw:
            md = vraw["min_df"]
            if isinstance(md, bool) or not isinstance(md, (int, float)):
                raise ConfigError("vectorizer.min_df must be a number")
            if isinstance(md, float) and not (0.0 < md <= 1.0):
                raise ConfigError("vectorizer.min_df as float must be in (0.0, 1.0]")
            if isinstance(md, int) and md < 1:
                raise ConfigError("vectorizer.min_df as int must be >= 1")
            v.min_df = md
        cfg.vectorizer = v

    if "selection_metric" in raw:
        if raw["selection_metric"] not in VALID_SELECTION_METRICS:
            raise ConfigError(
                f"selection_metric must be one of {sorted(VALID_SELECTION_METRICS)}, "
                f"got {raw['selection_metric']!r}"
            )
        cfg.selection_metric = raw["selection_metric"]

    if "top_k_error_examples" in raw:
        tk = raw["top_k_error_examples"]
        if isinstance(tk, bool) or not isinstance(tk, int) or tk < 0:
            raise ConfigError("top_k_error_examples must be a non-negative integer")
        cfg.top_k_error_examples = tk

    if "cross_validation" in raw:
        cv_raw = raw["cross_validation"]
        if not isinstance(cv_raw, dict):
            raise ConfigError("cross_validation must be an object")
        cv = CrossValidationConfig()
        if "enabled" in cv_raw:
            if not isinstance(cv_raw["enabled"], bool):
                raise ConfigError("cross_validation.enabled must be boolean")
            cv.enabled = cv_raw["enabled"]
        if "folds" in cv_raw:
            f = cv_raw["folds"]
            if isinstance(f, bool) or not isinstance(f, int) or f < 2:
                raise ConfigError("cross_validation.folds must be an integer >= 2")
            cv.folds = f
        cfg.cross_validation = cv

    return cfg


def validate_model_set(cfg: PipelineConfig, registry: dict[str, Any]) -> None:
    """Enforce the spec's model-set rules at config-load time.

    Raises ConfigError if:
      - any model name is unknown
      - fewer than 3 distinct models after dedupe
      - no linear or no probabilistic family represented
    """
    unknown = [m for m in cfg.models if m not in registry]
    if unknown:
        raise ConfigError(
            f"unknown model name(s) in config: {unknown}. "
            f"Valid options: {sorted(registry.keys())}"
        )
    if len(cfg.models) < 3:
        raise ConfigError(
            f"config.models must contain at least 3 distinct entries (got {len(cfg.models)}: {cfg.models}). "
            f"The build spec requires at least 3 baseline models."
        )
    families: set[str] = set()
    for name in cfg.models:
        families |= registry[name].families
    missing = []
    if "linear" not in families:
        missing.append("linear")
    if "probabilistic" not in families:
        missing.append("probabilistic")
    if missing:
        raise ConfigError(
            f"config.models must include at least one model from each required family. "
            f"Missing: {missing}. Family tags by model: "
            f"{ {n: sorted(registry[n].families) for n in cfg.models} }"
        )
