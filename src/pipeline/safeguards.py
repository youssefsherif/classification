from __future__ import annotations

from typing import Any

import pandas as pd

from .preprocessing import preprocess_text


CLASS_IMBALANCE_THRESHOLD = 5.0


def build_safeguards_report(
    train_split: pd.DataFrame,
    val_split: pd.DataFrame,
) -> dict[str, Any]:
    warnings: list[dict[str, Any]] = []

    # Class imbalance in training split
    train_counts = train_split["label"].value_counts()
    if len(train_counts) > 0:
        max_c = int(train_counts.max())
        min_c = int(train_counts.min())
        if min_c == 0:
            ratio = float("inf")
        else:
            ratio = max_c / min_c
        if ratio > CLASS_IMBALANCE_THRESHOLD:
            warnings.append({
                "code": "CLASS_IMBALANCE",
                "severity": "warn",
                "details": {
                    "ratio": ratio if ratio != float("inf") else None,
                    "max_class": train_counts.idxmax(),
                    "min_class": train_counts.idxmin(),
                    "counts": {str(k): int(v) for k, v in train_counts.items()},
                },
            })

    # Class missing from validation
    train_labels = set(train_split["label"].astype(str).unique())
    val_labels = set(val_split["label"].astype(str).unique())
    missing_in_val = sorted(train_labels - val_labels)
    if missing_in_val:
        warnings.append({
            "code": "CLASS_MISSING_FROM_VALIDATION",
            "severity": "warn",
            "details": {"labels": missing_in_val},
        })

    # Empty validation set
    if len(val_split) == 0:
        warnings.append({
            "code": "EMPTY_VALIDATION_SET",
            "severity": "warn",
            "details": {},
        })

    # Train/validation duplicate texts (after preprocessing)
    train_processed = train_split["text"].astype(str).map(preprocess_text)
    val_processed = val_split["text"].astype(str).map(preprocess_text)
    train_set = set(train_processed.tolist())
    overlap_mask = val_processed.isin(train_set)
    overlap_count = int(overlap_mask.sum())
    if overlap_count > 0:
        example_ids = val_split.loc[overlap_mask, "id"].astype(str).tolist()[:10]
        warnings.append({
            "code": "DUPLICATE_TEXTS_ACROSS_SPLITS",
            "severity": "warn",
            "details": {"count": overlap_count, "example_ids": example_ids},
        })

    return {
        "warnings": warnings,
        "status": "ok" if not warnings else "ok_with_warnings",
    }
