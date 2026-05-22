from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io_paths import (
    ARTIFACTS_MANIFEST,
    CROSS_VALIDATION_REPORT,
    DATA_VALIDATION_REPORT,
    ERROR_ANALYSIS,
    METRICS,
    MODEL_SELECTION_REPORT,
    PREPROCESSING_PREVIEW,
    SAFEGUARDS_REPORT,
    SPLIT_REPORT,
    TEST_PREDICTIONS,
    VECTORIZER_PATH,
    WINNER_META_PATH,
    WINNER_PATH,
    model_path,
    relative_to_root,
)


def _env_info() -> dict[str, Any]:
    info: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("sklearn", "pandas", "numpy", "joblib"):
        try:
            mod = __import__(name)
            info[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info[name] = "missing"
    return info


def build_run_manifest(
    *,
    random_seed: int,
    config_snapshot: dict[str, Any],
    files_read: dict[str, Any],
    stages_completed: list[str],
    models_trained: list[str],
    models_failed: list[str],
    winning_model: str,
    selection_metric: str,
    key_metrics: dict[str, Any],
    cv_enabled: bool,
    encoding_used: str,
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {
        "data_validation_report": relative_to_root(DATA_VALIDATION_REPORT),
        "preprocessing_preview": relative_to_root(PREPROCESSING_PREVIEW),
        "split_report": relative_to_root(SPLIT_REPORT),
        "metrics": relative_to_root(METRICS),
        "model_selection_report": relative_to_root(MODEL_SELECTION_REPORT),
        "artifacts_manifest": relative_to_root(ARTIFACTS_MANIFEST),
        "error_analysis": relative_to_root(ERROR_ANALYSIS),
        "safeguards_report": relative_to_root(SAFEGUARDS_REPORT),
        "cross_validation_report": (
            relative_to_root(CROSS_VALIDATION_REPORT) if cv_enabled else None
        ),
        "test_predictions": relative_to_root(TEST_PREDICTIONS),
        "vectorizer": relative_to_root(VECTORIZER_PATH),
        "trained_models": {
            name: relative_to_root(model_path(name)) for name in models_trained
        },
        "winning_model": relative_to_root(WINNER_PATH),
        "winner_meta": relative_to_root(WINNER_META_PATH),
    }

    return {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "random_seed": random_seed,
        "config_snapshot": config_snapshot,
        "files_read": files_read,
        "encoding_used": encoding_used,
        "stages_completed": stages_completed,
        "models_trained": models_trained,
        "models_failed": models_failed,
        "winning_model": winning_model,
        "selection_metric": selection_metric,
        "key_metrics": key_metrics,
        "artifacts": artifacts,
        "environment": _env_info(),
    }
