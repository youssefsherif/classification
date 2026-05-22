from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def evaluate_model(model: Any, X_val: Any, y_val: Any, labels: list[str]) -> dict[str, Any]:
    y_pred = model.predict(X_val)
    acc = float(accuracy_score(y_val, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_val, y_pred, average="macro", zero_division=0, labels=labels
    )
    p_pc, r_pc, f1_pc, support_pc = precision_recall_fscore_support(
        y_val, y_pred, average=None, zero_division=0, labels=labels
    )
    per_class: dict[str, Any] = {}
    for i, lbl in enumerate(labels):
        per_class[lbl] = {
            "precision": float(p_pc[i]),
            "recall": float(r_pc[i]),
            "f1": float(f1_pc[i]),
            "support": int(support_pc[i]),
        }
    cm = confusion_matrix(y_val, y_pred, labels=labels)
    return {
        "status": "ok",
        "accuracy": acc,
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": list(labels),
            "matrix": [[int(v) for v in row] for row in np.asarray(cm).tolist()],
        },
    }


def evaluate_all(
    fitted_models: dict[str, Any],
    failures: dict[str, str],
    X_val: Any,
    y_val: Any,
    labels: list[str],
    validation_size: int,
) -> dict[str, Any]:
    models_block: dict[str, Any] = {}
    for name, model in fitted_models.items():
        try:
            models_block[name] = evaluate_model(model, X_val, y_val, labels)
        except Exception as e:
            models_block[name] = {
                "status": "failed",
                "error": f"{type(e).__name__}: {e}",
            }
    for name, err in failures.items():
        models_block[name] = {"status": "failed", "error": err}

    return {
        "validation_size": int(validation_size),
        "labels": list(labels),
        "models": models_block,
    }
