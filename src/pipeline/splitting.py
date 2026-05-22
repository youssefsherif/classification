from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


def split_train_validation(
    df: pd.DataFrame,
    validation_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Stratified train/validation split with deterministic fallback.

    Returns (train_split_df, val_split_df, split_report_dict).
    """
    warnings: list[str] = []
    label_counts = df["label"].value_counts().to_dict()
    too_small = [lbl for lbl, c in label_counts.items() if c < 2]

    stratify = df["label"] if not too_small else None
    if too_small:
        warnings.append(
            f"stratification disabled: classes with <2 samples: {too_small}"
        )

    n = len(df)
    if n < 2:
        raise ValueError(f"cannot split a dataset of size {n}")

    # Ensure at least 1 sample in each side for the tiny-dataset edge case.
    expected_val = int(round(n * validation_fraction))
    if expected_val < 1:
        expected_val = 1
        warnings.append(
            f"validation_fraction={validation_fraction} would produce 0 rows; "
            f"forced validation_size=1"
        )
    if expected_val >= n:
        expected_val = n - 1
        warnings.append(
            f"validation_fraction={validation_fraction} would leave 0 train rows; "
            f"forced validation_size={expected_val}"
        )
    effective_test_size = expected_val / n

    train_split, val_split = train_test_split(
        df,
        test_size=effective_test_size,
        random_state=seed,
        stratify=stratify,
        shuffle=True,
    )
    train_split = train_split.reset_index(drop=True)
    val_split = val_split.reset_index(drop=True)

    train_label_counts = train_split["label"].value_counts().to_dict()
    val_label_counts = val_split["label"].value_counts().to_dict()
    all_labels = sorted(set(train_label_counts) | set(val_label_counts))
    label_breakdown = {
        lbl: {
            "train": int(train_label_counts.get(lbl, 0)),
            "validation": int(val_label_counts.get(lbl, 0)),
        }
        for lbl in all_labels
    }

    missing_in_val = [lbl for lbl in train_label_counts if lbl not in val_label_counts]
    if missing_in_val:
        warnings.append(
            f"labels present in train but missing from validation: {missing_in_val}"
        )

    report = {
        "random_seed": seed,
        "validation_fraction": validation_fraction,
        "effective_validation_fraction": effective_test_size,
        "stratified": stratify is not None,
        "train_size": int(len(train_split)),
        "validation_size": int(len(val_split)),
        "labels": label_breakdown,
        "warnings": warnings,
    }
    return train_split, val_split, report
