from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .inference import score_for


def _reason_for(score_type: str, score: float | None, is_proba: bool) -> str:
    if score is None:
        return "Model does not expose a confidence score; useful to manually inspect"
    if is_proba:
        if score < 0.6:
            return f"Low confidence prediction ({score:.3f}); review for ambiguous wording"
        if score >= 0.9:
            return f"High confidence wrong prediction ({score:.3f}); potential labeling issue or hard sample"
        return f"Mid confidence wrong prediction ({score:.3f}); worth inspecting"
    # margin or decision
    if score < 0:
        return f"Negative decision score ({score:.3f}); model was leaning the other way"
    return f"Positive decision score ({score:.3f}) but wrong label; worth inspecting"


def build_error_analysis(
    winner_name: str,
    model: Any,
    vectorizer: Any,
    val_df: pd.DataFrame,
    val_texts_processed: list[str],
    top_k: int,
) -> dict[str, Any]:
    if len(val_df) == 0:
        return {
            "winning_model": winner_name,
            "examples": [],
            "note": "validation split is empty",
        }
    X_val = vectorizer.transform(val_texts_processed)
    y_pred = [str(p) for p in model.predict(X_val)]
    y_true = val_df["label"].astype(str).tolist()
    ids = val_df["id"].astype(str).tolist()
    texts = val_df["text"].astype(str).tolist()

    # Compute per-row score info using shared helper
    classes = list(getattr(model, "classes_", []))
    rows: list[dict[str, Any]] = []
    for i in range(X_val.shape[0]):
        if y_pred[i] == y_true[i]:
            continue
        s = score_for(model, X_val[i])
        # confidence_or_score = the score corresponding to the predicted label
        # which is s["best_score"] when score_for picked the same index as model.predict.
        # In the rare mismatch case, fall back to per-class lookup.
        score = s["best_score"]
        if s["per_class_scores"] is not None and y_pred[i] in s["per_class_scores"]:
            score = float(s["per_class_scores"][y_pred[i]])
        is_proba = s["score_type"] == "proba"
        rows.append({
            "id": ids[i],
            "text": texts[i],
            "true_label": y_true[i],
            "predicted_label": y_pred[i],
            "confidence_or_score": score,
            "score_type": s["score_type"],
            "per_class_scores": s["per_class_scores"],
            "raw_margin": s["raw_margin"],
            "reason": _reason_for(s["score_type"], score, is_proba),
        })

    if not rows:
        return {
            "winning_model": winner_name,
            "examples": [],
            "note": "no misclassifications on validation split",
        }

    # Sort by ascending confidence (= least sure first)
    def sort_key(r: dict[str, Any]) -> float:
        s = r["confidence_or_score"]
        if s is None:
            return float("inf")
        return float(s)

    rows.sort(key=sort_key)
    return {
        "winning_model": winner_name,
        "examples": rows[: max(0, int(top_k))],
    }
