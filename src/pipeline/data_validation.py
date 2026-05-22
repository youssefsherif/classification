from __future__ import annotations

from typing import Any

import pandas as pd

from .data_loading import REQUIRED_TEST_COLS, REQUIRED_TRAIN_COLS


class DataValidationError(Exception):
    def __init__(self, message: str, report: dict[str, Any]):
        super().__init__(message)
        self.report = report


def _check_required_columns(df: pd.DataFrame, required: tuple[str, ...]) -> dict[str, Any]:
    missing = [c for c in required if c not in df.columns]
    return {"passed": len(missing) == 0, "missing": missing}


def _check_empty_text(df: pd.DataFrame) -> dict[str, Any]:
    if "text" not in df.columns:
        return {"passed": False, "empty_count": -1, "empty_ids": [], "note": "text column missing"}
    stripped = df["text"].fillna("").astype(str).str.strip()
    empty_mask = stripped == ""
    empty_ids = df.loc[empty_mask, "id"].astype(str).tolist() if "id" in df.columns else []
    return {
        "passed": int(empty_mask.sum()) == 0,
        "empty_count": int(empty_mask.sum()),
        "empty_ids": empty_ids[:20],
    }


def _check_unique_ids(df: pd.DataFrame) -> dict[str, Any]:
    if "id" not in df.columns:
        return {"passed": False, "duplicate_count": -1, "duplicates": [], "note": "id column missing"}
    ids = df["id"].astype(str)
    dup_mask = ids.duplicated(keep=False)
    duplicates = ids[dup_mask].drop_duplicates().tolist()
    return {
        "passed": len(duplicates) == 0,
        "duplicate_count": int(dup_mask.sum()),
        "duplicates": duplicates[:20],
    }


def _check_distinct_labels(df: pd.DataFrame) -> dict[str, Any]:
    if "label" not in df.columns:
        return {"passed": False, "count": 0, "labels": [], "note": "label column missing"}
    labels = sorted(df["label"].dropna().astype(str).unique().tolist())
    return {"passed": len(labels) >= 2, "count": len(labels), "labels": labels}


def build_report(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, Any]:
    checks = {
        "required_columns_train": _check_required_columns(train_df, REQUIRED_TRAIN_COLS),
        "required_columns_test": _check_required_columns(test_df, REQUIRED_TEST_COLS),
        "distinct_labels": _check_distinct_labels(train_df),
        "non_empty_text_train": _check_empty_text(train_df),
        "non_empty_text_test": _check_empty_text(test_df),
        "unique_ids_train": _check_unique_ids(train_df),
        "unique_ids_test": _check_unique_ids(test_df),
    }
    errors: list[str] = []
    for name, check in checks.items():
        if not check.get("passed", False):
            errors.append(name)
    return {
        "status": "ok" if not errors else "failed",
        "checks": checks,
        "errors": errors,
    }


def assert_ok(report: dict[str, Any]) -> None:
    if report["status"] != "ok":
        raise DataValidationError(
            f"data validation failed: {report['errors']}", report=report
        )
