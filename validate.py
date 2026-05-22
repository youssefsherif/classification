#!/usr/bin/env python
"""Repo-level validation. Exits 0 on full pass, 1 on any failure."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.pipeline.config import load_config
from src.pipeline.data_loading import REQUIRED_TEST_COLS, REQUIRED_TRAIN_COLS
from src.pipeline.io_paths import (
    ARTIFACTS_MANIFEST,
    DATA_VALIDATION_REPORT,
    ERROR_ANALYSIS,
    METRICS,
    MODEL_SELECTION_REPORT,
    PREPROCESSING_PREVIEW,
    PROJECT_ROOT,
    RUN_MANIFEST,
    SAFEGUARDS_REPORT,
    SPLIT_REPORT,
    TEST_CSV,
    TEST_PREDICTIONS,
    TRAIN_CSV,
    VECTORIZER_PATH,
    WINNER_META_PATH,
    WINNER_PATH,
    model_path,
)
from src.pipeline.models import MODEL_REGISTRY
from src.pipeline.preprocessing import preprocess_text
from src.pipeline.selection import select_winner


PREDICT_PY = PROJECT_ROOT / "predict.py"


class CheckResult:
    def __init__(self, name: str, passed: bool, detail: str = ""):
        self.name = name
        self.passed = passed
        self.detail = detail

    def __str__(self) -> str:
        tag = "[PASS]" if self.passed else "[FAIL]"
        return f"{tag} {self.name}" + (f" — {self.detail}" if self.detail else "")


REQUIRED_ARTIFACTS = [
    DATA_VALIDATION_REPORT,
    PREPROCESSING_PREVIEW,
    SPLIT_REPORT,
    METRICS,
    MODEL_SELECTION_REPORT,
    ARTIFACTS_MANIFEST,
    ERROR_ANALYSIS,
    SAFEGUARDS_REPORT,
    RUN_MANIFEST,
    TEST_PREDICTIONS,
    VECTORIZER_PATH,
    WINNER_PATH,
    WINNER_META_PATH,
]


def check_artifacts_exist() -> CheckResult:
    missing = [str(p) for p in REQUIRED_ARTIFACTS if not p.exists() or p.stat().st_size == 0]
    if missing:
        return CheckResult("required artifacts exist", False,
                           f"missing/empty: {missing}")
    return CheckResult("required artifacts exist", True, f"{len(REQUIRED_ARTIFACTS)} files")


def check_json_valid() -> CheckResult:
    bad: list[str] = []
    json_files: list[Path] = [
        DATA_VALIDATION_REPORT, PREPROCESSING_PREVIEW, SPLIT_REPORT, METRICS,
        MODEL_SELECTION_REPORT, ARTIFACTS_MANIFEST, ERROR_ANALYSIS, SAFEGUARDS_REPORT,
        RUN_MANIFEST, WINNER_META_PATH,
    ]
    expected_keys: dict[Path, set[str]] = {
        DATA_VALIDATION_REPORT: {"status", "checks"},
        SPLIT_REPORT: {"train_size", "validation_size", "labels"},
        METRICS: {"models", "labels"},
        MODEL_SELECTION_REPORT: {"winner", "primary_metric", "tie_break_applied"},
        ARTIFACTS_MANIFEST: {"vectorizer", "models", "winner"},
        ERROR_ANALYSIS: {"winning_model"},
        SAFEGUARDS_REPORT: {"warnings", "status"},
        RUN_MANIFEST: {"timestamp_utc", "random_seed", "artifacts"},
        WINNER_META_PATH: {"winner_name", "preprocessing_probe"},
        PREPROCESSING_PREVIEW: {"samples"},
    }
    for p in json_files:
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            need = expected_keys.get(p)
            if need and not need.issubset(obj.keys()):
                bad.append(f"{p.name}: missing keys {sorted(need - set(obj.keys()))}")
        except Exception as e:
            bad.append(f"{p.name}: {e}")
    if bad:
        return CheckResult("JSON artifacts are valid", False, "; ".join(bad))
    return CheckResult("JSON artifacts are valid", True, f"{len(json_files)} files")


def check_dataset_columns() -> CheckResult:
    try:
        train = pd.read_csv(TRAIN_CSV, dtype=str, keep_default_na=False)
        test = pd.read_csv(TEST_CSV, dtype=str, keep_default_na=False)
    except Exception as e:
        return CheckResult("required dataset columns", False, str(e))
    missing_train = [c for c in REQUIRED_TRAIN_COLS if c not in train.columns]
    missing_test = [c for c in REQUIRED_TEST_COLS if c not in test.columns]
    if missing_train or missing_test:
        return CheckResult("required dataset columns", False,
                           f"train missing {missing_train}, test missing {missing_test}")
    return CheckResult("required dataset columns", True)


def check_preprocessing_probe() -> CheckResult:
    try:
        meta = json.loads(WINNER_META_PATH.read_text(encoding="utf-8"))
        probe = meta["preprocessing_probe"]
        got = preprocess_text(probe["input"])
        if got != probe["expected"]:
            return CheckResult("preprocessing unchanged since training", False,
                               f"probe mismatch: input={probe['input']!r} "
                               f"expected={probe['expected']!r} got={got!r}")
        # idempotence
        s = "  Hello   WORLD\tfoo\nbar  "
        if preprocess_text(preprocess_text(s)) != preprocess_text(s):
            return CheckResult("preprocessing unchanged since training", False,
                               "preprocess_text is not idempotent")
        return CheckResult("preprocessing unchanged since training", True)
    except Exception as e:
        return CheckResult("preprocessing unchanged since training", False, str(e))


def _run_cli(text: str, use_stdin: bool = False) -> dict[str, Any]:
    cmd = [sys.executable, str(PREDICT_PY), "--json"]
    if use_stdin:
        cmd.append("--stdin")
        proc = subprocess.run(cmd, input=text, capture_output=True, text=True,
                              cwd=str(PROJECT_ROOT), timeout=60)
    else:
        cmd += ["--text", text]
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              cwd=str(PROJECT_ROOT), timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"CLI exit {proc.returncode}: stderr={proc.stderr.strip()}; stdout={proc.stdout.strip()}")
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except json.JSONDecodeError as e:
        raise RuntimeError(f"CLI did not emit JSON: {proc.stdout!r}") from e


def check_cli_preprocessing_consistency() -> CheckResult:
    """Three-probe behavioural test: identical preprocessed forms must yield identical outputs."""
    try:
        a = _run_cli("  Hello   WORLD  ")
        b = _run_cli("hello world")
        c = _run_cli("  THE\tApp\nIs   Easy")
        d = _run_cli("the app is easy")
    except Exception as e:
        return CheckResult("CLI uses the same preprocessing as training", False, str(e))
    fields = ("label", "score_type", "confidence", "score")
    mismatches: list[str] = []
    if any(a.get(k) != b.get(k) for k in fields):
        mismatches.append(f"'  Hello   WORLD  ' vs 'hello world' -> {a} vs {b}")
    if any(c.get(k) != d.get(k) for k in fields):
        mismatches.append(f"whitespace-heavy probe disagrees with normalized form -> {c} vs {d}")
    if mismatches:
        return CheckResult("CLI uses the same preprocessing as training", False,
                           "; ".join(mismatches))
    return CheckResult("CLI uses the same preprocessing as training", True)


def check_predict_py_static_guard() -> CheckResult:
    src = PREDICT_PY.read_text(encoding="utf-8")
    if "infer_one" not in src or "src.pipeline.inference" not in src:
        return CheckResult("predict.py imports the shared inference helper", False,
                           "expected import of infer_one from src.pipeline.inference")
    if "TfidfVectorizer" in src or "CountVectorizer" in src:
        return CheckResult("predict.py imports the shared inference helper", False,
                           "predict.py must not import a vectorizer directly")
    if re.search(r"def\s+preprocess_text\s*\(", src):
        return CheckResult("predict.py imports the shared inference helper", False,
                           "predict.py must not redefine preprocess_text")
    if re.search(r"re\.sub\s*\(", src):
        return CheckResult("predict.py imports the shared inference helper", False,
                           "predict.py must not call re.sub on input text")
    return CheckResult("predict.py imports the shared inference helper", True)


def check_three_models_with_families() -> CheckResult:
    try:
        metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    except Exception as e:
        return CheckResult(">=3 models trained with required families", False, str(e))
    ok_names = [n for n, m in metrics.get("models", {}).items() if m.get("status") == "ok"]
    if len(ok_names) < 3:
        return CheckResult(">=3 models trained with required families", False,
                           f"only {len(ok_names)} successful models: {ok_names}")
    # joblib files
    missing = [n for n in ok_names if not model_path(n).exists()]
    if missing:
        return CheckResult(">=3 models trained with required families", False,
                           f"missing joblib files for: {missing}")
    families: set[str] = set()
    for n in ok_names:
        entry = MODEL_REGISTRY.get(n)
        if entry is None:
            return CheckResult(">=3 models trained with required families", False,
                               f"unknown model in metrics.json: {n}")
        families |= entry.families
    missing_fams = [f for f in ("linear", "probabilistic") if f not in families]
    if missing_fams:
        return CheckResult(">=3 models trained with required families", False,
                           f"missing family/families: {missing_fams}")
    return CheckResult(">=3 models trained with required families", True,
                       f"{len(ok_names)} models, families={sorted(families)}")


def check_winner_deterministic() -> CheckResult:
    try:
        metrics = json.loads(METRICS.read_text(encoding="utf-8"))
        selection = json.loads(MODEL_SELECTION_REPORT.read_text(encoding="utf-8"))
        meta = json.loads(WINNER_META_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        return CheckResult("winner is deterministic", False, str(e))
    primary_metric = selection["primary_metric"]
    recomputed = select_winner(metrics, primary_metric)
    if recomputed["winner"] != selection["winner"]:
        return CheckResult("winner is deterministic", False,
                           f"select_winner recomputed {recomputed['winner']} vs saved {selection['winner']}")
    if meta["winner_name"] != selection["winner"]:
        return CheckResult("winner is deterministic", False,
                           f"winner.meta.json disagrees with selection report")
    return CheckResult("winner is deterministic", True,
                       f"{selection['winner']} ({primary_metric}={selection['primary_value']:.6f})")


def check_test_predictions_complete() -> CheckResult:
    try:
        test = pd.read_csv(TEST_CSV, dtype=str, keep_default_na=False)
        preds = pd.read_csv(TEST_PREDICTIONS, dtype=str, keep_default_na=False)
    except Exception as e:
        return CheckResult("test predictions cover all rows", False, str(e))
    if list(preds.columns) != ["id", "predicted_label"]:
        return CheckResult("test predictions cover all rows", False,
                           f"unexpected columns: {list(preds.columns)}")
    if len(preds) != len(test):
        return CheckResult("test predictions cover all rows", False,
                           f"row counts: predictions={len(preds)}, test={len(test)}")
    if set(preds["id"].tolist()) != set(test["id"].astype(str).tolist()):
        return CheckResult("test predictions cover all rows", False, "id sets differ")
    return CheckResult("test predictions cover all rows", True, f"{len(preds)} rows")


def check_cli_smoke() -> CheckResult:
    try:
        out = _run_cli("hello world")
    except Exception as e:
        return CheckResult("CLI runs end-to-end", False, str(e))
    if "label" not in out:
        return CheckResult("CLI runs end-to-end", False, f"missing label in output: {out}")
    try:
        metrics = json.loads(METRICS.read_text(encoding="utf-8"))
        labels = set(metrics.get("labels", []))
    except Exception:
        labels = set()
    if labels and out["label"] not in labels:
        return CheckResult("CLI runs end-to-end", False,
                           f"label {out['label']!r} not in metrics labels {sorted(labels)}")
    return CheckResult("CLI runs end-to-end", True, f"label={out['label']}")


def main() -> int:
    checks = [
        check_artifacts_exist(),
        check_json_valid(),
        check_dataset_columns(),
        check_preprocessing_probe(),
        check_predict_py_static_guard(),
        check_cli_preprocessing_consistency(),
        check_three_models_with_families(),
        check_winner_deterministic(),
        check_test_predictions_complete(),
        check_cli_smoke(),
    ]
    failed = 0
    for c in checks:
        print(c)
        if not c.passed:
            failed += 1
    total = len(checks)
    if failed == 0:
        print(f"OK ({total} checks passed)")
        return 0
    print(f"FAILED ({failed} of {total} checks failed)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
