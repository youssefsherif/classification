from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import joblib

from . import io_paths
from .io_paths import (
    ARTIFACTS_MANIFEST,
    DATA_VALIDATION_REPORT,
    METRICS,
    MODEL_SELECTION_REPORT,
    MODELS_DIR,
    PREPROCESSING_PREVIEW,
    SAFEGUARDS_REPORT,
    SPLIT_REPORT,
    VECTORIZER_PATH,
    WINNER_META_PATH,
    WINNER_PATH,
    model_path,
    relative_to_root,
    sha256_file,
    write_json_atomic,
)
from .preprocessing import PROBE_EXPECTED, PROBE_INPUT


class ArtifactsError(Exception):
    pass


def save_artifacts(
    vectorizer: Any,
    fitted_models: dict[str, Any],
    winner_name: str,
    selection_metric: str,
    labels: list[str],
) -> dict[str, Any]:
    """Sole binary writer. Persists vectorizer + each fitted model + winner copy + winner.meta.

    Returns the artifacts_manifest dict that was written.
    """
    if winner_name not in fitted_models:
        raise ArtifactsError(
            f"winner '{winner_name}' is not in fitted_models {sorted(fitted_models)}"
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. vectorizer
    joblib.dump(vectorizer, VECTORIZER_PATH)

    # 2. each fitted model
    model_entries: list[dict[str, Any]] = []
    for name, model in fitted_models.items():
        p = model_path(name)
        joblib.dump(model, p)
        model_entries.append({
            "name": name,
            "path": relative_to_root(p),
            "sha256": sha256_file(p),
        })

    # 3. winner copy
    shutil.copyfile(model_path(winner_name), WINNER_PATH)

    # 4. winner.meta.json
    winner_meta = {
        "winner_name": winner_name,
        "selection_metric": selection_metric,
        "labels": list(labels),
        "vectorizer_path": relative_to_root(VECTORIZER_PATH),
        "winner_path": relative_to_root(WINNER_PATH),
        "preprocessing_module": "src.pipeline.preprocessing",
        "preprocessing_function": "preprocess_text",
        "preprocessing_probe": {"input": PROBE_INPUT, "expected": PROBE_EXPECTED},
    }
    write_json_atomic(WINNER_META_PATH, winner_meta)

    # 5. artifacts_manifest.json with SHA-256 of binaries
    manifest = {
        "stage": "ARTIFACTS_SAVED",
        "vectorizer": {
            "path": relative_to_root(VECTORIZER_PATH),
            "sha256": sha256_file(VECTORIZER_PATH),
        },
        "models": model_entries,
        "winner": {
            "name": winner_name,
            "path": relative_to_root(WINNER_PATH),
            "sha256": sha256_file(WINNER_PATH),
        },
        "winner_meta": {"path": relative_to_root(WINNER_META_PATH)},
        "reports": [
            relative_to_root(DATA_VALIDATION_REPORT),
            relative_to_root(PREPROCESSING_PREVIEW),
            relative_to_root(SPLIT_REPORT),
            relative_to_root(METRICS),
            relative_to_root(MODEL_SELECTION_REPORT),
            relative_to_root(SAFEGUARDS_REPORT),
        ],
    }
    write_json_atomic(ARTIFACTS_MANIFEST, manifest)

    # 6. existence + non-empty assertion
    required_paths: list[Path] = [
        VECTORIZER_PATH,
        WINNER_PATH,
        WINNER_META_PATH,
        ARTIFACTS_MANIFEST,
    ] + [model_path(n) for n in fitted_models]
    for p in required_paths:
        if not p.exists() or p.stat().st_size == 0:
            raise ArtifactsError(f"required artifact missing or empty after save: {p}")

    return manifest
