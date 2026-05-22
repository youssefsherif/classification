#!/usr/bin/env python
"""Top-level pipeline entry point. Runs the full state machine end-to-end."""

from __future__ import annotations

import json
import random
import sys
import traceback
from pathlib import Path

import numpy as np

from src.pipeline import inference as _inference_mod  # to reset cache
from src.pipeline.artifacts import save_artifacts
from src.pipeline.config import ConfigError, load_config, validate_model_set
from src.pipeline.cross_validation import run_cross_validation
from src.pipeline.data_loading import DataLoadError, load_csvs
from src.pipeline.data_validation import (
    DataValidationError,
    assert_ok,
    build_report,
)
from src.pipeline.error_analysis import build_error_analysis
from src.pipeline.evaluation import evaluate_all
from src.pipeline.features import FeatureExtractionError, fit_vectorizer
from src.pipeline.io_paths import (
    ARTIFACTS_DIR,
    CONFIG_PATH,
    CROSS_VALIDATION_REPORT,
    DATA_VALIDATION_REPORT,
    ERROR_ANALYSIS,
    METRICS,
    MODEL_SELECTION_REPORT,
    MODELS_DIR,
    PREPROCESSING_PREVIEW,
    RUN_MANIFEST,
    SAFEGUARDS_REPORT,
    SPLIT_REPORT,
    TEST_CSV,
    TEST_PREDICTIONS,
    TRAIN_CSV,
    ensure_dirs,
    relative_to_root,
    sha256_file,
    write_json_atomic,
)
from src.pipeline.logging_setup import get_logger, log_stage
from src.pipeline.models import MODEL_REGISTRY
from src.pipeline.prediction import predict_test_set
from src.pipeline.preprocessing import preprocess_text
from src.pipeline.reporting import build_run_manifest
from src.pipeline.safeguards import build_safeguards_report
from src.pipeline.selection import SelectionError, select_winner
from src.pipeline.splitting import split_train_validation
from src.pipeline.state import PipelineState, StateContext
from src.pipeline.training import (
    TrainingError,
    enforce_post_training_requirements,
    train_models,
)


EXIT_OK = 0
EXIT_GENERIC = 1
EXIT_VALIDATION = 2
EXIT_TRAINING = 3


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _preprocess_df_texts(df, log, stage_name):
    df = df.copy()
    df["text_processed"] = df["text"].astype(str).map(preprocess_text)
    empty_mask = df["text_processed"] == ""
    if empty_mask.any():
        ids = df.loc[empty_mask, "id"].astype(str).tolist()[:20]
        log_stage(log, stage_name,
                  f"ABORT: {int(empty_mask.sum())} row(s) became empty after preprocessing: ids={ids}")
        raise DataValidationError(
            f"{int(empty_mask.sum())} row(s) empty after preprocessing in {stage_name}",
            report={
                "stage": stage_name,
                "post_preprocessing_empty_text": {"count": int(empty_mask.sum()), "ids": ids},
            },
        )
    return df


def main() -> int:
    log = get_logger()
    ctx = StateContext()
    ensure_dirs()

    # --- INIT ---
    log_stage(log, "INIT", f"project root = {TRAIN_CSV.parent}")
    try:
        cfg = load_config(CONFIG_PATH)
        validate_model_set(cfg, MODEL_REGISTRY)
    except ConfigError as e:
        log_stage(log, "INIT", f"config error: {e}")
        return EXIT_VALIDATION
    _set_seeds(cfg.random_seed)
    ctx.config = cfg
    log_stage(log, "INIT", f"config loaded; seed={cfg.random_seed}, models={cfg.models}")

    # Reset inference cache in case a stale model was loaded earlier
    _inference_mod._CACHE.clear()

    # --- DATA_LOADED ---
    try:
        train_df, test_df, encoding_used = load_csvs(TRAIN_CSV, TEST_CSV)
    except DataLoadError as e:
        log_stage(log, "DATA_LOADED", f"load error: {e}")
        return EXIT_VALIDATION
    ctx.train_df = train_df
    ctx.test_df = test_df
    ctx.encoding_used = encoding_used
    ctx.file_hashes = {
        "train_csv": sha256_file(TRAIN_CSV),
        "test_csv": sha256_file(TEST_CSV),
        "config_json": sha256_file(CONFIG_PATH),
    }
    ctx.advance(PipelineState.DATA_LOADED)
    log_stage(log, "DATA_LOADED",
              f"train.csv rows={len(train_df)}, test.csv rows={len(test_df)}, encoding={encoding_used}")

    # --- DATA_VALIDATED ---
    report = build_report(train_df, test_df)
    write_json_atomic(DATA_VALIDATION_REPORT, report)
    try:
        assert_ok(report)
    except DataValidationError as e:
        log_stage(log, "DATA_VALIDATED", f"FAILED: {e}; see {DATA_VALIDATION_REPORT}")
        return EXIT_VALIDATION
    ctx.advance(PipelineState.DATA_VALIDATED)
    log_stage(log, "DATA_VALIDATED",
              f"{report['checks']['distinct_labels']['count']} distinct labels, "
              f"0 empty rows, 0 duplicate ids")

    # --- TEXT_PREPROCESSED ---
    try:
        train_df_pp = _preprocess_df_texts(train_df, log, "TEXT_PREPROCESSED")
        test_df_pp = _preprocess_df_texts(test_df, log, "TEXT_PREPROCESSED")
    except DataValidationError as e:
        # append to data_validation_report so the evaluator sees it
        report["status"] = "failed"
        report["errors"].append("post_preprocessing_empty_text")
        report["post_preprocessing"] = e.report
        write_json_atomic(DATA_VALIDATION_REPORT, report)
        return EXIT_VALIDATION
    ctx.train_df = train_df_pp
    ctx.test_df = test_df_pp

    # preprocessing preview
    preview = {
        "samples": [
            {
                "id": str(train_df_pp.iloc[i]["id"]),
                "original": str(train_df_pp.iloc[i]["text"]),
                "processed": str(train_df_pp.iloc[i]["text_processed"]),
                "source": "train",
            }
            for i in range(min(5, len(train_df_pp)))
        ] + [
            {
                "id": str(test_df_pp.iloc[i]["id"]),
                "original": str(test_df_pp.iloc[i]["text"]),
                "processed": str(test_df_pp.iloc[i]["text_processed"]),
                "source": "test",
            }
            for i in range(min(3, len(test_df_pp)))
        ]
    }
    write_json_atomic(PREPROCESSING_PREVIEW, preview)
    ctx.advance(PipelineState.TEXT_PREPROCESSED)
    log_stage(log, "TEXT_PREPROCESSED",
              f"preprocessed {len(train_df_pp)} train + {len(test_df_pp)} test rows")

    # --- SPLIT_CREATED ---
    train_split, val_split, split_report = split_train_validation(
        train_df_pp, cfg.validation_split, cfg.random_seed
    )
    write_json_atomic(SPLIT_REPORT, split_report)
    ctx.train_split_df = train_split
    ctx.val_split_df = val_split
    ctx.advance(PipelineState.SPLIT_CREATED)
    log_stage(log, "SPLIT_CREATED",
              f"train={split_report['train_size']}, val={split_report['validation_size']}, "
              f"stratified={split_report['stratified']}")

    # Safeguards (after split, before training)
    sg_report = build_safeguards_report(train_split, val_split)
    write_json_atomic(SAFEGUARDS_REPORT, sg_report)
    if sg_report["warnings"]:
        log_stage(log, "SAFEGUARDS",
                  f"{len(sg_report['warnings'])} warning(s): "
                  f"{[w['code'] for w in sg_report['warnings']]}")
    else:
        log_stage(log, "SAFEGUARDS", "no warnings")

    # --- FEATURES_FIT (in-memory only) ---
    train_texts = train_split["text_processed"].astype(str).tolist()
    val_texts = val_split["text_processed"].astype(str).tolist()
    try:
        vectorizer = fit_vectorizer(cfg.vectorizer, train_texts)
    except FeatureExtractionError as e:
        log_stage(log, "FEATURES_FIT", f"FAILED: {e}")
        return EXIT_TRAINING
    X_train = vectorizer.transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    y_train = train_split["label"].astype(str).tolist()
    y_val = val_split["label"].astype(str).tolist()
    ctx.vectorizer = vectorizer
    ctx.X_train = X_train
    ctx.X_val = X_val
    ctx.y_train = y_train
    ctx.y_val = y_val
    ctx.advance(PipelineState.FEATURES_FIT)
    log_stage(log, "FEATURES_FIT",
              f"vocabulary_size={len(vectorizer.vocabulary_)}, "
              f"X_train.shape={X_train.shape}, X_val.shape={X_val.shape} (in-memory only)")

    # --- MODELS_TRAINED (in-memory only) ---
    fitted, failures = train_models(cfg.models, X_train, y_train, cfg.random_seed)
    for name, err in failures.items():
        log_stage(log, "MODELS_TRAINED", f"model '{name}' failed: {err}")
    try:
        enforce_post_training_requirements(fitted, failures)
    except TrainingError as e:
        log_stage(log, "MODELS_TRAINED", f"FAILED: {e}")
        # still emit a partial metrics.json so the evaluator can see what happened
        partial = {
            "validation_size": int(len(val_split)),
            "labels": sorted(set(y_train) | set(y_val)),
            "models": {n: {"status": "failed", "error": err} for n, err in failures.items()},
        }
        write_json_atomic(METRICS, partial)
        return EXIT_TRAINING
    ctx.models = fitted
    ctx.model_failures = failures
    ctx.advance(PipelineState.MODELS_TRAINED)
    log_stage(log, "MODELS_TRAINED",
              f"trained={sorted(fitted)}, failed={sorted(failures)} (in-memory only)")

    # --- MODELS_EVALUATED ---
    label_universe = sorted(set(y_train) | set(y_val))
    metrics = evaluate_all(fitted, failures, X_val, y_val, label_universe, len(val_split))
    write_json_atomic(METRICS, metrics)
    ctx.metrics = metrics
    ctx.advance(PipelineState.MODELS_EVALUATED)
    log_stage(log, "MODELS_EVALUATED",
              f"evaluated {sum(1 for m in metrics['models'].values() if m.get('status') == 'ok')} model(s)")

    # --- WINNER_SELECTED ---
    try:
        selection = select_winner(metrics, cfg.selection_metric)
    except SelectionError as e:
        log_stage(log, "WINNER_SELECTED", f"FAILED: {e}")
        return EXIT_TRAINING
    write_json_atomic(MODEL_SELECTION_REPORT, selection)
    ctx.selection = selection
    ctx.winner_name = selection["winner"]
    ctx.advance(PipelineState.WINNER_SELECTED)
    log_stage(log, "WINNER_SELECTED",
              f"{selection['winner']} {selection['primary_metric']}={selection['primary_value']:.6f} "
              f"(tie_break={selection['tie_break_applied']})")

    # --- ARTIFACTS_SAVED (sole binary writer) ---
    save_artifacts(
        vectorizer=vectorizer,
        fitted_models=fitted,
        winner_name=ctx.winner_name,
        selection_metric=cfg.selection_metric,
        labels=label_universe,
    )
    _inference_mod._CACHE.clear()  # force reload from disk for downstream stages
    ctx.advance(PipelineState.ARTIFACTS_SAVED)
    log_stage(log, "ARTIFACTS_SAVED",
              f"vectorizer + {len(fitted)} models persisted, winner={ctx.winner_name}")

    # --- Error analysis (uses winning model on validation split) ---
    err_report = build_error_analysis(
        winner_name=ctx.winner_name,
        model=fitted[ctx.winner_name],
        vectorizer=vectorizer,
        val_df=val_split,
        val_texts_processed=val_texts,
        top_k=cfg.top_k_error_examples,
    )
    write_json_atomic(ERROR_ANALYSIS, err_report)
    log_stage(log, "ARTIFACTS_SAVED",
              f"error_analysis: {len(err_report.get('examples', []))} example(s) recorded")

    # --- TEST_PREDICTIONS_GENERATED ---
    # Use the *saved* artifacts via the shared inference helper, to guarantee the
    # CLI inference path is exercised at pipeline time too.
    summary = predict_test_set(test_df_pp, TEST_PREDICTIONS)
    if summary["row_count"] != len(test_df_pp):
        log_stage(log, "TEST_PREDICTIONS_GENERATED",
                  f"FAILED: row count mismatch {summary['row_count']} vs {len(test_df_pp)}")
        return EXIT_TRAINING
    ctx.advance(PipelineState.TEST_PREDICTIONS_GENERATED)
    log_stage(log, "TEST_PREDICTIONS_GENERATED",
              f"wrote {summary['row_count']} predictions to {relative_to_root(TEST_PREDICTIONS)}")

    # --- Optional cross-validation (stretch) ---
    if cfg.cross_validation.enabled:
        log_stage(log, "CROSS_VALIDATION", f"running {cfg.cross_validation.folds}-fold CV")
        cv_report = run_cross_validation(
            cfg,
            train_df_pp["text_processed"].astype(str).tolist(),
            train_df_pp["label"].astype(str).tolist(),
        )
        write_json_atomic(CROSS_VALIDATION_REPORT, cv_report)
        log_stage(log, "CROSS_VALIDATION",
                  f"completed {cv_report['effective_folds']} folds; "
                  f"adjustment={cv_report['adjustment_note']}")

    # --- REPORT_EXPORTED ---
    winner_metrics = metrics["models"][ctx.winner_name]
    key_metrics = {
        k: winner_metrics[k]
        for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1")
    }
    manifest = build_run_manifest(
        random_seed=cfg.random_seed,
        config_snapshot=cfg.raw,
        files_read={
            "train_csv": {"path": relative_to_root(TRAIN_CSV), "rows": int(len(train_df)),
                          "sha256": ctx.file_hashes["train_csv"]},
            "test_csv": {"path": relative_to_root(TEST_CSV), "rows": int(len(test_df)),
                         "sha256": ctx.file_hashes["test_csv"]},
            "config_json": {"path": relative_to_root(CONFIG_PATH),
                            "sha256": ctx.file_hashes["config_json"]},
        },
        stages_completed=ctx.stages_completed + [PipelineState.REPORT_EXPORTED.value],
        models_trained=sorted(fitted.keys()),
        models_failed=sorted(failures.keys()),
        winning_model=ctx.winner_name,
        selection_metric=cfg.selection_metric,
        key_metrics=key_metrics,
        cv_enabled=cfg.cross_validation.enabled,
        encoding_used=ctx.encoding_used,
    )
    write_json_atomic(RUN_MANIFEST, manifest)
    ctx.advance(PipelineState.REPORT_EXPORTED)
    log_stage(log, "REPORT_EXPORTED", f"wrote {relative_to_root(RUN_MANIFEST)}")

    log_stage(log, "PIPELINE", "OK")
    return EXIT_OK


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        print("UNEXPECTED ERROR:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(EXIT_GENERIC)
