from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = PROJECT_ROOT / "models"

CONFIG_PATH = PROJECT_ROOT / "config.json"
TRAIN_CSV = PROJECT_ROOT / "train.csv"
TEST_CSV = PROJECT_ROOT / "test.csv"

# Artifact paths
DATA_VALIDATION_REPORT = ARTIFACTS_DIR / "data_validation_report.json"
PREPROCESSING_PREVIEW = ARTIFACTS_DIR / "preprocessing_preview.json"
SPLIT_REPORT = ARTIFACTS_DIR / "split_report.json"
METRICS = ARTIFACTS_DIR / "metrics.json"
MODEL_SELECTION_REPORT = ARTIFACTS_DIR / "model_selection_report.json"
ARTIFACTS_MANIFEST = ARTIFACTS_DIR / "artifacts_manifest.json"
ERROR_ANALYSIS = ARTIFACTS_DIR / "error_analysis.json"
SAFEGUARDS_REPORT = ARTIFACTS_DIR / "safeguards_report.json"
CROSS_VALIDATION_REPORT = ARTIFACTS_DIR / "cross_validation_report.json"
RUN_MANIFEST = ARTIFACTS_DIR / "run_manifest.json"
TEST_PREDICTIONS = ARTIFACTS_DIR / "test_predictions.csv"

# Model paths
VECTORIZER_PATH = MODELS_DIR / "vectorizer.joblib"
WINNER_PATH = MODELS_DIR / "winner.joblib"
WINNER_META_PATH = MODELS_DIR / "winner.meta.json"


def model_path(name: str) -> Path:
    return MODELS_DIR / f"model_{name}.joblib"


def ensure_dirs() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def to_posix(p: Path | str) -> str:
    return str(p).replace(os.sep, "/")


def relative_to_root(p: Path) -> str:
    try:
        return to_posix(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return to_posix(p)


def write_json_atomic(path: Path, obj: Any) -> None:
    """Write JSON atomically: write to tmp file in same dir, then os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=False, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
