from __future__ import annotations

from typing import Any

from .models import MODEL_REGISTRY


class TrainingError(Exception):
    pass


def train_models(
    model_names: list[str],
    X_train: Any,
    y_train: Any,
    seed: int,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Train each named model. Returns (fitted_models, failures).

    Per-model failures are isolated: a failure in one model does not stop the others.
    """
    fitted: dict[str, Any] = {}
    failures: dict[str, str] = {}
    for name in model_names:
        entry = MODEL_REGISTRY.get(name)
        if entry is None:
            failures[name] = f"unknown model: {name}"
            continue
        try:
            model = entry.build(seed)
            model.fit(X_train, y_train)
            fitted[name] = model
        except Exception as e:
            failures[name] = f"{type(e).__name__}: {e}"
    return fitted, failures


def enforce_post_training_requirements(
    fitted: dict[str, Any],
    failures: dict[str, str],
) -> None:
    """After training, enforce: >=3 successful models, and the surviving set still
    spans both 'linear' and 'probabilistic' families.
    """
    if len(fitted) < 3:
        raise TrainingError(
            f"fewer than 3 models trained successfully (got {len(fitted)}). "
            f"Successes: {sorted(fitted.keys())}. Failures: {failures}"
        )
    families: set[str] = set()
    for name in fitted:
        families |= MODEL_REGISTRY[name].families
    missing = []
    if "linear" not in families:
        missing.append("linear")
    if "probabilistic" not in families:
        missing.append("probabilistic")
    if missing:
        raise TrainingError(
            f"after training failures, the surviving model set no longer covers required families. "
            f"Missing: {missing}. Survivors: {sorted(fitted.keys())}. Failures: {failures}"
        )
