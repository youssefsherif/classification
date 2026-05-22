from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from .config import PipelineConfig
from .evaluation import evaluate_model
from .features import build_vectorizer
from .models import MODEL_REGISTRY


def run_cross_validation(
    cfg: PipelineConfig,
    train_texts_processed: list[str],
    train_labels: list[str],
) -> dict[str, Any]:
    requested_folds = cfg.cross_validation.folds
    labels_series = pd.Series(train_labels)
    min_class = int(labels_series.value_counts().min()) if len(labels_series) else 0
    n_samples = len(train_labels)

    adjustment_notes: list[str] = []
    folds = requested_folds

    # Need at least 2 folds and at least 2 samples to do CV at all.
    if n_samples < 2:
        return {
            "requested_folds": requested_folds,
            "effective_folds": 0,
            "stratified": False,
            "adjustment_note": (
                f"insufficient data for cross-validation (n_samples={n_samples}); CV skipped"
            ),
            "labels": sorted(set(train_labels)),
            "folds": [],
            "aggregate": {},
        }

    if min_class < 2:
        # cannot stratify reliably
        adjustment_notes.append(
            f"a class has only {min_class} samples; falling back to KFold (non-stratified)"
        )
        # KFold requires folds <= n_samples
        if folds > n_samples:
            adjustment_notes.append(
                f"requested folds={requested_folds} > n_samples={n_samples}; reduced to {n_samples}"
            )
            folds = n_samples
        if folds < 2:
            return {
                "requested_folds": requested_folds,
                "effective_folds": 0,
                "stratified": False,
                "adjustment_note": (
                    f"after clamping, folds would be {folds} (<2); CV skipped. "
                    f"n_samples={n_samples}"
                ),
                "labels": sorted(set(train_labels)),
                "folds": [],
                "aggregate": {},
            }
        splitter = KFold(n_splits=folds, shuffle=True, random_state=cfg.random_seed)
        stratified = False
    else:
        # StratifiedKFold requires folds <= min_class size
        if folds > min_class:
            adjustment_notes.append(
                f"requested folds={requested_folds} > smallest class size {min_class}; "
                f"reduced to {min_class}"
            )
            folds = min_class
        if folds < 2:
            adjustment_notes.append(
                f"folds<2 after clamp; falling back to KFold with folds={min(requested_folds, n_samples)}"
            )
            folds = min(requested_folds, n_samples)
            if folds < 2:
                return {
                    "requested_folds": requested_folds,
                    "effective_folds": 0,
                    "stratified": False,
                    "adjustment_note": "; ".join(adjustment_notes) or "folds<2; CV skipped",
                    "labels": sorted(set(train_labels)),
                    "folds": [],
                    "aggregate": {},
                }
            splitter = KFold(n_splits=folds, shuffle=True, random_state=cfg.random_seed)
            stratified = False
        else:
            splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cfg.random_seed)
            stratified = True

    adjustment_note: str | None = "; ".join(adjustment_notes) if adjustment_notes else None

    label_universe = sorted(set(train_labels))
    fold_results: list[dict[str, Any]] = []
    per_model_aggregate: dict[str, dict[str, list[float]]] = {
        name: {"accuracy": [], "macro_precision": [], "macro_recall": [], "macro_f1": []}
        for name in cfg.models
    }

    X_full = np.array(train_texts_processed)
    y_full = np.array(train_labels)
    indices = np.arange(len(X_full))

    for fold_idx, (tr_idx, va_idx) in enumerate(splitter.split(indices, y_full if stratified else None)):
        fold_block: dict[str, Any] = {"fold": fold_idx, "train_size": int(len(tr_idx)), "val_size": int(len(va_idx)), "models": {}}
        vec = build_vectorizer(cfg.vectorizer)
        try:
            X_tr = vec.fit_transform(X_full[tr_idx].tolist())
        except ValueError as e:
            fold_block["error"] = f"vectorizer fit failed: {e}"
            fold_results.append(fold_block)
            continue
        X_va = vec.transform(X_full[va_idx].tolist())
        y_tr = y_full[tr_idx].tolist()
        y_va = y_full[va_idx].tolist()
        for name in cfg.models:
            entry = MODEL_REGISTRY.get(name)
            if entry is None:
                fold_block["models"][name] = {"status": "failed", "error": "unknown model"}
                continue
            try:
                m = entry.build(cfg.random_seed)
                m.fit(X_tr, y_tr)
                metrics = evaluate_model(m, X_va, y_va, label_universe)
                fold_block["models"][name] = {
                    k: metrics[k]
                    for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1")
                }
                for k in per_model_aggregate[name]:
                    per_model_aggregate[name][k].append(float(metrics[k]))
            except Exception as e:
                fold_block["models"][name] = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
        fold_results.append(fold_block)

    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    for name, metric_lists in per_model_aggregate.items():
        agg: dict[str, dict[str, float]] = {}
        for metric_name, values in metric_lists.items():
            if values:
                agg[metric_name] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
            else:
                agg[metric_name] = {"mean": float("nan"), "std": float("nan")}
        aggregate[name] = agg

    return {
        "requested_folds": requested_folds,
        "effective_folds": folds,
        "stratified": stratified,
        "adjustment_note": adjustment_note,
        "labels": label_universe,
        "folds": fold_results,
        "aggregate": aggregate,
    }
