from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np

from .io_paths import VECTORIZER_PATH, WINNER_META_PATH, WINNER_PATH
from .preprocessing import preprocess_text


class InferenceError(Exception):
    pass


@dataclass
class InferenceResult:
    label: str
    confidence: float | None
    score: float | None
    score_type: str  # "proba" | "margin" | "decision" | "none"
    model_name: str
    per_class_scores: dict[str, float] | None = None
    raw_margin: float | None = None  # for binary decision_function classifiers


_CACHE: dict[str, Any] = {}


def _load() -> tuple[Any, Any, dict[str, Any]]:
    if "loaded" in _CACHE:
        return _CACHE["loaded"]
    for p in (VECTORIZER_PATH, WINNER_PATH, WINNER_META_PATH):
        if not p.exists():
            raise InferenceError(
                f"required artifact missing: {p}. Run run_pipeline.py first."
            )
    try:
        vectorizer = joblib.load(VECTORIZER_PATH)
        model = joblib.load(WINNER_PATH)
    except Exception as e:
        raise InferenceError(
            f"failed to load model artifacts (possibly sklearn version mismatch): "
            f"{type(e).__name__}: {e}. Retrain by running run_pipeline.py."
        ) from e
    meta = json.loads(WINNER_META_PATH.read_text(encoding="utf-8"))
    _CACHE["loaded"] = (vectorizer, model, meta)
    return vectorizer, model, meta


def score_for(model: Any, X: Any) -> dict[str, Any]:
    """Compute per-sample score info for a single row X (already vectorized, shape (1, n_features)).

    Returns dict with keys: score_type, label_for_index (callable), best_index, best_score,
    per_class_scores, raw_margin.
    """
    classes = list(getattr(model, "classes_", []))
    n_classes = len(classes)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        best_idx = int(np.argmax(proba))
        per_class = {str(classes[i]): float(proba[i]) for i in range(n_classes)}
        return {
            "score_type": "proba",
            "best_index": best_idx,
            "best_score": float(proba[best_idx]),
            "per_class_scores": per_class,
            "raw_margin": None,
        }
    if hasattr(model, "decision_function"):
        df = model.decision_function(X)
        df = np.asarray(df)
        if n_classes == 2:
            # binary: scalar margin against classes_[1]
            margin = float(df.ravel()[0])
            per_class = {
                str(classes[0]): -margin,
                str(classes[1]): margin,
            }
            best_idx = 1 if margin >= 0 else 0
            return {
                "score_type": "margin",
                "best_index": best_idx,
                "best_score": float(per_class[str(classes[best_idx])]),
                "per_class_scores": per_class,
                "raw_margin": margin,
            }
        # multiclass: shape (1, n_classes)
        row = df[0] if df.ndim == 2 else df
        best_idx = int(np.argmax(row))
        per_class = {str(classes[i]): float(row[i]) for i in range(n_classes)}
        return {
            "score_type": "decision",
            "best_index": best_idx,
            "best_score": float(row[best_idx]),
            "per_class_scores": per_class,
            "raw_margin": None,
        }
    # no scoring available
    y_pred = model.predict(X)
    pred_label = str(y_pred[0])
    best_idx = classes.index(pred_label) if pred_label in classes else 0
    return {
        "score_type": "none",
        "best_index": best_idx,
        "best_score": None,
        "per_class_scores": None,
        "raw_margin": None,
    }


def infer_one(text: str) -> InferenceResult:
    """Single-text inference. Shared by predict.py and prediction.py (test-set scoring).

    Raises InferenceError on missing artifacts, load failures, or empty text after preprocessing.
    """
    vectorizer, model, meta = _load()
    processed = preprocess_text(text)
    if processed == "":
        raise InferenceError("text is empty after preprocessing")
    X = vectorizer.transform([processed])
    s = score_for(model, X)
    classes = list(getattr(model, "classes_", []))
    label = str(classes[s["best_index"]]) if classes else str(model.predict(X)[0])
    return InferenceResult(
        label=label,
        confidence=s["best_score"] if s["score_type"] == "proba" else None,
        score=s["best_score"] if s["score_type"] in ("margin", "decision") else None,
        score_type=s["score_type"],
        model_name=meta.get("winner_name", "unknown"),
        per_class_scores=s["per_class_scores"],
        raw_margin=s["raw_margin"],
    )


def batch_predict(texts: list[str]) -> tuple[list[str], list[dict[str, Any]]]:
    """Batched version for test-set scoring. Returns (labels, score_dicts).

    Empty-after-preprocessing texts raise InferenceError (the caller is responsible
    for catching this at the validation stage; this is a safety net).
    """
    vectorizer, model, _meta = _load()
    processed = [preprocess_text(t) for t in texts]
    empty_indices = [i for i, p in enumerate(processed) if p == ""]
    if empty_indices:
        raise InferenceError(
            f"{len(empty_indices)} text(s) empty after preprocessing at indices {empty_indices[:10]}"
        )
    X = vectorizer.transform(processed)
    preds = [str(p) for p in model.predict(X)]
    score_dicts: list[dict[str, Any]] = []
    classes = list(getattr(model, "classes_", []))
    # We compute scores row-by-row to reuse score_for() and stay consistent with the CLI.
    for i in range(X.shape[0]):
        s = score_for(model, X[i])
        score_dicts.append(s)
        # Cross-check: predict() label should match score_for's best_index for safety
        if classes:
            best_lbl = str(classes[s["best_index"]])
            if best_lbl != preds[i]:
                # Defensive: trust model.predict() (vector-level argmax),
                # but log mismatch via the score dict
                s["predict_label_mismatch"] = {"predict": preds[i], "score_best": best_lbl}
    return preds, score_dicts
