# Implementation Plan — Replayable Text Classification Pipeline

## 0. Goals and Design Principles

This is a **replayable, stage-driven** local pipeline. The evaluator will:

1. Clone the repo from a clean state.
2. Possibly **swap the CSVs** with equivalent fixtures (same schema, different content).
3. Delete generated artifacts.
4. Re-run the pipeline from scratch.
5. Run a validation script and the inference CLI.

Therefore the implementation is guided by these principles:

- **Explicit stages over implicit flow.** Each stage is a function that consumes a typed state, validates inputs, produces outputs, and emits an artifact. The stage name appears in logs, artifacts, and code identifiers so the evaluator can grep for it.
- **No assumption about content.** No hard-coded label names, no assumed binary classification, no token-level rules tied to the sample text. Everything is derived from the data at hand.
- **Deterministic by construction.** All RNG paths funnel through a single seed loaded from `config.json`. No wall-clock dependent behaviour inside the pipeline (timestamps appear only in the manifest, never as features).
- **Single source of truth for preprocessing.** A `preprocess_text()` function lives in one module and is imported by training, evaluation, CLI inference, and `validate.py`. The validator checks this by calling it on a fixed string and comparing the result against what the saved artifacts imply.
- **Fail loudly and early.** If the data is malformed, the pipeline aborts at the validation stage with a non-zero exit code and a structured error report — it does not silently drop rows.
- **Simple > clever.** scikit-learn baselines, TF-IDF features, joblib persistence. No notebooks in the runtime path.

---

## 1. Repository Layout

```
classification/
├── README.md                      # how to run, in 60 seconds
├── implementation_plan.md         # this file
├── requirements.txt               # pinned versions (scikit-learn, pandas, numpy, joblib)
├── config.json                    # default config (matches the spec sample)
├── train.csv                      # sample fixture (replaceable)
├── test.csv                       # sample fixture (replaceable)
│
├── run_pipeline.py                # entry point: runs the full state machine
├── predict.py                     # single-text inference CLI
├── validate.py                    # repo-level validation command
│
├── src/
│   └── pipeline/
│       ├── __init__.py
│       ├── state.py               # PipelineState enum + StateContext dataclass
│       ├── config.py              # Config dataclass + loader with defaults
│       ├── logging_setup.py       # structured stage-prefixed logger
│       ├── io_paths.py            # ARTIFACTS_DIR, MODELS_DIR, atomic JSON writer
│       ├── data_loading.py        # Stage: DATA_LOADED
│       ├── data_validation.py     # Stage: DATA_VALIDATED
│       ├── preprocessing.py       # Stage: TEXT_PREPROCESSED (single source of truth)
│       ├── splitting.py           # Stage: SPLIT_CREATED
│       ├── features.py            # Stage: FEATURES_FIT
│       ├── models.py              # model registry + factory
│       ├── training.py            # Stage: MODELS_TRAINED
│       ├── evaluation.py          # Stage: MODELS_EVALUATED
│       ├── selection.py           # Stage: WINNER_SELECTED (deterministic tie-break)
│       ├── artifacts.py           # Stage: ARTIFACTS_SAVED (vectorizer, models, winner.meta, manifest)
│       ├── inference.py           # shared infer_one() — used by CLI and prediction.py
│       ├── error_analysis.py      # top-k misclassified validation rows
│       ├── prediction.py          # Stage: TEST_PREDICTIONS_GENERATED (calls inference.infer_one)
│       ├── reporting.py           # Stage: REPORT_EXPORTED + run_manifest
│       ├── safeguards.py          # imbalance / missing class / duplicate checks
│       └── cross_validation.py    # optional stretch
│
├── artifacts/                     # JSON reports, predictions (created at runtime)
│   ├── data_validation_report.json
│   ├── preprocessing_preview.json
│   ├── split_report.json
│   ├── metrics.json
│   ├── model_selection_report.json
│   ├── artifacts_manifest.json       # written by ARTIFACTS_SAVED stage
│   ├── error_analysis.json
│   ├── safeguards_report.json
│   ├── cross_validation_report.json  # only if enabled
│   ├── run_manifest.json
│   └── test_predictions.csv
│
├── models/                        # binary artifacts (created at runtime)
│   ├── vectorizer.joblib
│   ├── model_<name>.joblib        # one per trained model
│   ├── winner.joblib              # copy of winning model (stable path for CLI)
│   └── winner.meta.json           # winner name, paths, preprocessing probe
│
└── tests/
    ├── test_preprocessing.py      # deterministic, idempotent
    ├── test_selection.py          # tie-break logic
    ├── test_splitting.py          # stratification fallback paths
    └── test_end_to_end.py         # runs full pipeline on sample fixture
```

`artifacts/` and `models/` are listed in `.gitignore`. They are created on every run.

---

## 2. Pipeline State Machine

`src/pipeline/state.py` defines:

```python
class PipelineState(str, Enum):
    INIT = "INIT"
    DATA_LOADED = "DATA_LOADED"
    DATA_VALIDATED = "DATA_VALIDATED"
    TEXT_PREPROCESSED = "TEXT_PREPROCESSED"
    SPLIT_CREATED = "SPLIT_CREATED"
    FEATURES_FIT = "FEATURES_FIT"
    MODELS_TRAINED = "MODELS_TRAINED"
    MODELS_EVALUATED = "MODELS_EVALUATED"
    WINNER_SELECTED = "WINNER_SELECTED"
    ARTIFACTS_SAVED = "ARTIFACTS_SAVED"
    TEST_PREDICTIONS_GENERATED = "TEST_PREDICTIONS_GENERATED"
    REPORT_EXPORTED = "REPORT_EXPORTED"
```

A `StateContext` dataclass holds the artifacts produced so far (DataFrames, vectorizer, fitted models, metrics dict, winner name). `run_pipeline.py` invokes stage functions in order; each function asserts the required prior state and updates the context. The logger prefixes every line with the stage name:

```
[2026-05-22T10:14:32Z] [DATA_LOADED] train.csv rows=10, test.csv rows=3
[2026-05-22T10:14:32Z] [DATA_VALIDATED] 2 distinct labels, 0 empty rows
[2026-05-22T10:14:32Z] [WINNER_SELECTED] logistic_regression macro_f1=0.847
```

This satisfies the spec's "evaluator should be able to see these stages in code structure, logs, or output artifacts."

`run_pipeline.py` exits with code `0` on success, `2` on validation failure, `3` on unrecoverable training failure, `1` on unexpected exceptions.

---

## 3. Configuration Loading (`config.py`)

A dataclass `PipelineConfig` with **safe defaults** for every field so a partial config still works:

| Field | Default | Notes |
|---|---|---|
| `random_seed` | `42` | piped into numpy, splitter, every model that accepts `random_state` |
| `validation_split` | `0.2` | must satisfy `0.0 < x < 1.0`; values outside this range → exit 2. The pipeline uses the configured fraction exactly, with no silent rescaling. |
| `models` | `["logistic_regression", "linear_svm", "naive_bayes"]` | unknown names raise a clear `ConfigError` listing valid options |
| `vectorizer.type` | `"tfidf"` | also accepts `"count"`; anything else → `ConfigError` |
| `vectorizer.ngram_range` | `[1, 2]` | parsed to tuple, validated `low<=high>=1` |
| `vectorizer.max_features` | `5000` | `null` allowed → unlimited |
| `vectorizer.min_df` | `1` | int or float, validated |
| `selection_metric` | `"macro_f1"` | one of `accuracy / macro_precision / macro_recall / macro_f1` |
| `top_k_error_examples` | `10` | clamped to validation size |
| `cross_validation.enabled` | `false` | stretch goal |
| `cross_validation.folds` | `5` | |

**Missing `config.json` → hard failure.** The spec explicitly says the pipeline must read `config.json` from disk. We exit `2` with a clear message naming the expected path. The "safe defaults" above apply only to **individual missing fields inside a present `config.json`** (partial configs are tolerated; an absent file is not).

A malformed `config.json` (bad JSON, wrong types) → abort with clear message, exit `2`.

**Model-set enforcement.** After parsing `config.models`, validate at config-load time:
- At least 3 entries (after dedupe).
- At least one entry whose registry entry is tagged `family="linear"` (LogReg, LinearSVM, Ridge, SGD).
- At least one entry whose registry entry is tagged `family="probabilistic"` (LogReg, MultinomialNB, SGD with `loss="log_loss"`).
- All entries must exist in `MODEL_REGISTRY`.

If any condition fails, the pipeline exits `2` with a message listing what was missing and the valid options. We do **not** silently augment the config — the evaluator's intent should be respected and surfaced, not papered over. (LogisticRegression satisfies both linear and probabilistic, so the default config of 3 models passes trivially.)

---

## 4. Stage-by-Stage Specification

### 4.1 `DATA_LOADED` — `data_loading.py`

- Read CSVs with `pandas.read_csv(..., encoding="utf-8", dtype={"id": str, "text": str, "label": str})`.
- `id` is read as string to dodge integer-overflow / leading-zero issues, but we preserve the original representation for round-tripping into `test_predictions.csv`.
- On `UnicodeDecodeError`, retry with `encoding="utf-8-sig"` (BOM), then `latin-1` as last resort, recording the chosen encoding in the manifest.
- On `FileNotFoundError`, exit `2` with a message naming the missing file.

### 4.2 `DATA_VALIDATED` — `data_validation.py`

Performs every required check and **writes `data_validation_report.json` regardless of outcome** (so the evaluator sees what went wrong):

```json
{
  "status": "ok" | "failed",
  "checks": {
    "required_columns_train": {"passed": true, "missing": []},
    "required_columns_test": {"passed": true, "missing": []},
    "distinct_labels": {"passed": true, "count": 2, "labels": ["negative", "positive"]},
    "non_empty_text_train": {"passed": true, "empty_count": 0, "empty_ids": []},
    "non_empty_text_test":  {"passed": true, "empty_count": 0, "empty_ids": []},
    "unique_ids_train": {"passed": true, "duplicate_count": 0, "duplicates": []},
    "unique_ids_test":  {"passed": true, "duplicate_count": 0, "duplicates": []}
  },
  "errors": []
}
```

- "Non-empty after stripping whitespace" is checked on raw text *before* preprocessing.
- Duplicate IDs: list up to the first 20 to keep the artifact small.
- If any check fails, write the report, print the human-readable summary, exit `2`. **Do not proceed.**

### 4.3 `TEXT_PREPROCESSED` — `preprocessing.py`

Single function `preprocess_text(s: str) -> str`:

```python
def preprocess_text(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s
```

- This function is imported by training, prediction, and the CLI via a single shared helper. It is the only legal preprocessing path.
- `preprocessing_preview.json` stores `{"samples": [{"id": ..., "original": ..., "processed": ...}, ...]}` for the first 5 train rows and first 3 test rows.
- After preprocessing, if any row is now empty, the pipeline **aborts** (exit `2`) and appends a `post_preprocessing_empty_text` entry to `data_validation_report.json` listing the offending IDs. This is consistent with the spec's "non-empty text fields" rule and the "fail loudly" principle — no row is ever silently dropped.

### 4.4 `SPLIT_CREATED` — `splitting.py`

`train_test_split` with `random_state=seed`, `test_size=validation_split`, and `stratify=labels` **when possible**.

Stratification fallback ladder:
1. **Try stratified** with the full label column.
2. **If any class has < 2 samples**, stratification is impossible. Log a warning, fall back to non-stratified split, record `stratified=false` and the offending classes in `split_report.json`.
3. **If the validation slice would have 0 samples for some class**, record this in `safeguards_report.json` so the evaluator can see it.

`split_report.json`:

```json
{
  "random_seed": 42,
  "validation_fraction": 0.2,
  "stratified": true,
  "train_size": 8,
  "validation_size": 2,
  "labels": {
    "positive": {"train": 4, "validation": 1},
    "negative": {"train": 4, "validation": 1}
  },
  "warnings": []
}
```

Corner case — tiny dataset (e.g. 3 rows): if `int(n * validation_split) == 0`, force at least 1 validation row and warn.

### 4.5 `FEATURES_FIT` — `features.py`

- Build `TfidfVectorizer` (or `CountVectorizer`) **from config** with `ngram_range`, `max_features`, `min_df`.
- Fit on **train split only** to prevent validation leakage.
- **Held in the state context only — not written to disk here.** All binary persistence is centralized in `ARTIFACTS_SAVED` (§4.9) so the stage boundary is meaningful.
- Corner case: if every term gets filtered by `min_df` (vocabulary empty), abort with a clear message recommending `min_df=1`. Exit `3`.

### 4.6 `MODELS_TRAINED` — `training.py`, `models.py`

`models.py` is a registry. Each entry declares its `family` so config-time validation (see §3) can enforce "at least one linear" and "at least one probabilistic":

```python
@dataclass
class RegistryEntry:
    build: Callable[[int], BaseEstimator]
    families: frozenset[str]   # subset of {"linear", "probabilistic"}

MODEL_REGISTRY = {
    "logistic_regression": RegistryEntry(
        build=lambda seed: LogisticRegression(random_state=seed, max_iter=1000, n_jobs=1),
        families=frozenset({"linear", "probabilistic"}),
    ),
    "linear_svm": RegistryEntry(
        build=lambda seed: LinearSVC(random_state=seed),
        families=frozenset({"linear"}),
    ),
    "naive_bayes": RegistryEntry(
        build=lambda seed: MultinomialNB(),
        families=frozenset({"probabilistic"}),
    ),
    "ridge_classifier": RegistryEntry(
        build=lambda seed: RidgeClassifier(random_state=seed),
        families=frozenset({"linear"}),
    ),
    "sgd_classifier": RegistryEntry(
        build=lambda seed: SGDClassifier(random_state=seed, loss="log_loss"),
        families=frozenset({"linear", "probabilistic"}),
    ),
}
```

`training.py`:

- Iterates `config.models`, instantiates each, fits on the **same** vectorized train split.
- Wraps each `.fit()` in a `try/except`. If a single model fails, log the failure, record it in `metrics.json` with `"status": "failed", "error": "<message>"`, and continue with the others. If after training **fewer than 3 models succeed**, or the set of successful models no longer includes both a linear and a probabilistic family member, abort with exit `3` and a message listing the missing family. This protects the spec's "≥3 baselines, one linear, one probabilistic" requirement even when an individual training failure narrows the set below the threshold.
- **Fitted estimators are held in the state context only.** They are not written to disk in this stage — `ARTIFACTS_SAVED` (§4.9) is the sole binary writer. The failure list is also carried in the context and surfaces in `metrics.json` during `MODELS_EVALUATED`.

The composition of `config.models` itself is validated at config-load time (§3), so an evaluator-supplied config that violates the requirement is rejected up front — not silently augmented, not "warn and proceed".

### 4.7 `MODELS_EVALUATED` — `evaluation.py`

For each trained model, compute on the validation split:

- `accuracy`, `macro_precision`, `macro_recall`, `macro_f1` (via `sklearn.metrics`, `zero_division=0` to suppress warnings on absent classes)
- `per_class` breakdown: `{label: {precision, recall, f1, support}}`
- `confusion_matrix` as `{"labels": [...], "matrix": [[...], ...]}` (labels list ensures the matrix is interpretable even if a class is absent from predictions)

`metrics.json`:

```json
{
  "validation_size": 2,
  "labels": ["negative", "positive"],
  "models": {
    "logistic_regression": {
      "status": "ok",
      "accuracy": 1.0, "macro_precision": 1.0, "macro_recall": 1.0, "macro_f1": 1.0,
      "per_class": {...},
      "confusion_matrix": {"labels": ["negative","positive"], "matrix": [[1,0],[0,1]]}
    },
    "linear_svm": {...},
    "naive_bayes": {...}
  }
}
```

### 4.8 `WINNER_SELECTED` — `selection.py`

Pure function for testability:

```python
EPS = 1e-12  # treat metric values within EPS as equal — guards against float noise

def select_winner(metrics: dict, primary_metric: str) -> dict:
    candidates = [(name, m) for name, m in metrics["models"].items() if m["status"] == "ok"]
    if not candidates:
        raise SelectionError("no successful models to choose from")

    # Sort: 1) primary metric desc, 2) macro_precision desc, 3) name asc.
    candidates.sort(key=lambda kv: (-kv[1][primary_metric],
                                    -kv[1]["macro_precision"],
                                    kv[0]))
    winner_name, winner_metrics = candidates[0]

    # Determine which rule actually decided the winner by inspecting the
    # runners-up. tie_break_applied is one of: "none", "macro_precision",
    # "alphabetical".
    competitors = [m for _, m in candidates[1:]]
    tied_on_primary = [m for m in competitors
                       if abs(m[primary_metric] - winner_metrics[primary_metric]) < EPS]
    if not tied_on_primary:
        tie_break_applied = "none"
    else:
        tied_on_precision = [m for m in tied_on_primary
                             if abs(m["macro_precision"] - winner_metrics["macro_precision"]) < EPS]
        tie_break_applied = "alphabetical" if tied_on_precision else "macro_precision"

    reason = _format_reason(winner_name, winner_metrics, candidates, primary_metric,
                            tie_break_applied)

    return {
        "winner": winner_name,
        "primary_metric": primary_metric,
        "primary_value": winner_metrics[primary_metric],
        "ranking": [{"name": n, primary_metric: m[primary_metric],
                     "macro_precision": m["macro_precision"]} for n, m in candidates],
        "tie_break_applied": tie_break_applied,   # "none" | "macro_precision" | "alphabetical"
        "reason": reason,
    }
```

`tie_break_applied` is **computed**, not a placeholder. The three possible values map directly to the spec's tie-break ladder:

- `"none"` — the winner strictly beat all runners-up on the primary metric.
- `"macro_precision"` — at least one runner-up tied on the primary metric and lost on macro_precision.
- `"alphabetical"` — at least one runner-up tied on both primary metric and macro_precision; alphabetical name order decided it.

`_format_reason()` produces a human-readable sentence that mentions the closest competitor and, when a tie-break was applied, the specific value(s) that were equal. For example:

- `"none"` → `"Selected because it had the highest macro_f1 (0.847). Closest competitor logistic_regression at 0.823."`
- `"macro_precision"` → `"Tied with linear_svm on macro_f1 (0.847); selected on higher macro_precision (0.891 vs 0.864)."`
- `"alphabetical"` → `"Tied with naive_bayes on macro_f1 (0.847) and macro_precision (0.891); selected by alphabetical name order."`

This is **the** function unit-tested for tie-break correctness (one test per branch). `model_selection_report.json` is its return value, written to disk verbatim.

### 4.9 `ARTIFACTS_SAVED` — `artifacts.py`

The spec lists `ARTIFACTS_SAVED` as a discrete pipeline stage, so it gets its own module, log line, and verification step rather than being implicit in earlier stages. **All binary persistence happens here, and only here.** Earlier stages write **their own JSON reports** eagerly (validation report, split report, metrics, selection report — these are the artifacts that describe what happened during that stage), but the fitted vectorizer and every fitted model stay in the in-memory `StateContext` until this stage runs. That keeps the stage boundary meaningful and gives the evaluator a single grep-able log line confirming inference inputs are on disk.

What this stage does, in order, with each substep logged under `[ARTIFACTS_SAVED]`:

1. **Persist the vectorizer** to `models/vectorizer.joblib` via `joblib.dump`. The vectorizer object was fit in `FEATURES_FIT` and carried in the state context until now.
2. **Persist every successfully trained model** to `models/model_<name>.joblib`. The fitted estimators were carried in the state context from `MODELS_TRAINED`.
3. **Copy the winning model** to `models/winner.joblib` (copy, not symlink — Windows compatibility) so the CLI has a stable, version-agnostic filename.
4. **Write `models/winner.meta.json`** with:
   ```json
   {
     "winner_name": "logistic_regression",
     "selection_metric": "macro_f1",
     "labels": ["negative", "positive"],
     "vectorizer_path": "models/vectorizer.joblib",
     "winner_path": "models/winner.joblib",
     "preprocessing_module": "src.pipeline.preprocessing",
     "preprocessing_function": "preprocess_text",
     "preprocessing_probe": {"input": "  Hello   WORLD  ", "expected": "hello world"}
   }
   ```
   The CLI and `validate.py` both load this file to get a single declarative pointer to everything they need (so neither has to re-parse `model_selection_report.json` or guess at preprocessing). The `preprocessing_probe` is what allows `validate.py` to assert that the live `preprocess_text` matches what was used at training time.
5. **Write `artifacts/artifacts_manifest.json`** — the per-stage roll-up of every file persisted, with paths and SHA-256 hashes:
   ```json
   {
     "stage": "ARTIFACTS_SAVED",
     "vectorizer": {"path": "models/vectorizer.joblib", "sha256": "..."},
     "models": [
       {"name": "logistic_regression", "path": "models/model_logistic_regression.joblib", "sha256": "..."},
       {"name": "linear_svm",          "path": "models/model_linear_svm.joblib",          "sha256": "..."},
       {"name": "naive_bayes",         "path": "models/model_naive_bayes.joblib",         "sha256": "..."}
     ],
     "winner": {"name": "logistic_regression", "path": "models/winner.joblib", "sha256": "..."},
     "winner_meta": {"path": "models/winner.meta.json"},
     "reports": [
       "artifacts/data_validation_report.json",
       "artifacts/preprocessing_preview.json",
       "artifacts/split_report.json",
       "artifacts/metrics.json",
       "artifacts/model_selection_report.json",
       "artifacts/safeguards_report.json"
     ]
   }
   ```
6. **Existence assertion**: at the end of the stage, every path listed in `artifacts_manifest.json` is re-`stat`'d. If any is missing or zero bytes, the pipeline aborts with exit `3` before moving on to `TEST_PREDICTIONS_GENERATED`. This means inference downstream can assume its inputs exist.

The transition log line is: `[ARTIFACTS_SAVED] vectorizer + 3 models persisted, winner=logistic_regression`.

### 4.10 Error Analysis — `error_analysis.py`

For the winning model:

- Predict on the validation split.
- Find misclassified rows.
- For each, attempt to compute confidence via the shared helper `score_for(model, X)` (used by both error analysis and `inference.infer_one` so the CLI returns identical numbers):
  - `predict_proba` if available (LogReg, NB) → `confidence = max(proba)`; also store the per-class proba dict.
  - else `decision_function` if available (LinearSVC, Ridge) — **with an explicit binary special case**:
    - **Binary classifier** (`len(model.classes_) == 2`): `decision_function` returns a 1D array of shape `(n_samples,)`, the signed margin against the positive class (`model.classes_[1]`). Convert to a per-class score dict `{classes_[0]: -margin, classes_[1]: +margin}`; the reported `confidence_or_score` is the score corresponding to the *predicted* label (i.e. `+margin` if positive class was predicted, `-margin` otherwise — so larger = more confident in the prediction). The raw signed margin is also stored under `raw_margin` so the sign is recoverable.
    - **Multiclass** (`len(model.classes_) > 2`): `decision_function` returns shape `(n_samples, n_classes)`; the reported score is the value at the predicted class index, and the full per-class score dict is stored.
  - else `null`.
- Note that decision-function scores are unbounded — they are *not* probabilities. The artifact field is named `confidence_or_score` and `error_analysis.json` records `score_type: "proba" | "margin" | "decision" | "none"` so downstream readers don't confuse the two scales.
- Sort by ascending confidence (= "model was least sure" = most useful to inspect). Take `top_k_error_examples`, clamped to the number of available misclassifications.
- `reason` is templated, not free-form: `"Low confidence prediction (0.51); model nearly chose true label"` or `"High confidence wrong prediction (0.94); potential labeling issue or hard sample"`. The template is chosen by simple rules on the score.

```json
{
  "winning_model": "logistic_regression",
  "examples": [
    {"id": "4", "text": "the latest update made navigation confusing",
     "true_label": "negative", "predicted_label": "positive",
     "confidence_or_score": 0.62,
     "reason": "Low confidence prediction (0.62); review for ambiguous wording"}
  ]
}
```

Edge case — zero misclassifications: write `"examples": []` and `"note": "no misclassifications on validation split"`. Do not error.

### 4.11 `TEST_PREDICTIONS_GENERATED` — `prediction.py`

- Load `test.csv` (already loaded in context), preprocess with the **same** `preprocess_text`, transform with the **saved vectorizer**, predict with the **winning model**.
- Output `test_predictions.csv` with columns exactly `id,predicted_label`. IDs preserved as original strings.
- Row count must equal `len(test.csv)`; an internal assertion guards this.

### 4.12 `REPORT_EXPORTED` — `reporting.py`

Writes `run_manifest.json`:

```json
{
  "timestamp_utc": "2026-05-22T10:14:32Z",
  "random_seed": 42,
  "config_snapshot": {...},
  "files_read": {
    "train_csv":  {"path": "train.csv", "rows": 10, "sha256": "..."},
    "test_csv":   {"path": "test.csv",  "rows": 3,  "sha256": "..."},
    "config_json":{"path": "config.json", "sha256": "..."}
  },
  "stages_completed": ["INIT", "...", "REPORT_EXPORTED"],
  "models_trained":  ["logistic_regression", "linear_svm", "naive_bayes"],
  "models_failed":   [],
  "winning_model":   "logistic_regression",
  "selection_metric":"macro_f1",
  "key_metrics":     {"accuracy": 1.0, "macro_f1": 1.0},
  "artifacts": {
    "data_validation_report":   "artifacts/data_validation_report.json",
    "preprocessing_preview":    "artifacts/preprocessing_preview.json",
    "split_report":             "artifacts/split_report.json",
    "metrics":                  "artifacts/metrics.json",
    "model_selection_report":   "artifacts/model_selection_report.json",
    "artifacts_manifest":       "artifacts/artifacts_manifest.json",
    "error_analysis":           "artifacts/error_analysis.json",
    "safeguards_report":        "artifacts/safeguards_report.json",
    "cross_validation_report":  "artifacts/cross_validation_report.json",
    "test_predictions":         "artifacts/test_predictions.csv",
    "vectorizer":               "models/vectorizer.joblib",
    "trained_models": {
      "logistic_regression":    "models/model_logistic_regression.joblib",
      "linear_svm":             "models/model_linear_svm.joblib",
      "naive_bayes":            "models/model_naive_bayes.joblib"
    },
    "winning_model":            "models/winner.joblib",
    "winner_meta":              "models/winner.meta.json"
  },
  "environment": {
    "python": "3.11.7",
    "sklearn": "1.4.2",
    "pandas": "2.2.0",
    "numpy": "1.26.4",
    "platform": "Windows-11"
  }
}
```

Notes:
- The `artifacts` block lists **every** produced path, mirroring `artifacts_manifest.json` plus the JSON reports written by earlier stages. `cross_validation_report` is `null` when the CV flag is disabled (the key is always present so consumers can read it without a `KeyError`). Failed-model joblib paths are omitted from `trained_models` since they were not persisted.
- The SHA-256 of input files lets the evaluator confirm that fixtures actually flowed through training (no static cache).

### 4.13 Safeguards — `safeguards.py`

Runs after split, before training; results saved to `safeguards_report.json`:

- **Class imbalance**: ratio `max_class / min_class` in training set. Warn if `> 5.0`.
- **Class missing from validation**: any label in train but not in validation → warning, and the affected label.
- **Train/validation duplicate texts** (after preprocessing): if any post-preprocessed text appears in both splits → warning, with up to 10 example IDs. This catches leakage when the source data has near-duplicates.
- **Empty validation set after split** (tiny data): hard warning.

```json
{
  "warnings": [
    {"code": "CLASS_IMBALANCE", "severity": "warn",
     "details": {"ratio": 6.2, "max_class": "positive", "min_class": "negative"}},
    {"code": "DUPLICATE_TEXTS_ACROSS_SPLITS", "severity": "warn",
     "details": {"count": 2, "example_ids": ["5", "9"]}}
  ],
  "status": "ok_with_warnings"
}
```

Warnings never stop the pipeline. Only validation failures do.

### 4.14 Cross-Validation (Stretch) — `cross_validation.py`

If `config.cross_validation.enabled == true`:

- `StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)` on the full **preprocessed training data** (before the holdout split — CV is its own evaluation).
- Fallback to plain `KFold` if stratification impossible.
- For each fold, fit a fresh vectorizer + each candidate model on the fold's train, evaluate on the fold's val.
- Store per-fold and aggregate (mean ± std) metrics in `cross_validation_report.json`.
- This report is **informational**. Winner selection still uses the holdout validation `metrics.json` to keep behaviour consistent whether CV is on or off, but the manifest links to the CV report.

Corner case: requested `folds` greater than smallest class size → reduce folds to the largest viable value, record the adjustment.

---

## 5. The Inference CLI — `predict.py`

```bash
python predict.py --text "The app is easy to use"
python predict.py --text "..." --json     # machine-readable
```

`predict.py` is a thin argparse wrapper. All real work lives in `src/pipeline/inference.py::infer_one(text)`, the **shared inference helper** used both by the CLI and by `prediction.py` (test-set scoring). This is what makes "same preprocessing as training" enforceable rather than aspirational.

`infer_one(text)` behaviour:

1. Load `models/winner.meta.json` to discover paths and the recorded preprocessing probe.
2. Load `models/winner.joblib` and `models/vectorizer.joblib` via `joblib.load`.
3. Apply `preprocess_text` (imported from `src/pipeline/preprocessing.py` — same module the pipeline used; never re-implemented here).
4. Transform with the saved vectorizer.
5. `predict()` → label.
6. Compute confidence/score the same way error analysis does: `predict_proba` → `decision_function` → `null`.
7. Return an `InferenceResult` dataclass with `label`, `confidence`, `score`, `model_name`.

The CLI loads `infer_one`, calls it, and formats output. Default human output:
```
label: positive
confidence: 0.87
model: logistic_regression
```
`--json` flag prints `{"label": "...", "confidence": 0.87, "score": null, "model": "..."}`.

Corner cases:
- Any required artifact missing → exit `2` with `"Run run_pipeline.py first."`.
- Empty `--text` after preprocessing → exit `2` with `"Text is empty after preprocessing."`.
- `--text` not provided → argparse usage error.
- Optional `--stdin` flag reads from stdin for piping.
- sklearn version mismatch breaks `joblib.load` → catch and exit `2` with `"Model artifact incompatible with installed sklearn version — retrain."` (`requirements.txt` pins versions to make this rare).

---

## 6. The Validation Script — `validate.py`

`python validate.py` runs offline checks against the current state of the repo. Exit code `0` on full pass, `1` on any failure; prints a per-check pass/fail summary.

**Shared inference helper.** Before listing the validator's checks, note one structural requirement that makes the consistency check actually meaningful: both `predict.py` and the pipeline's `prediction.py` (test set scoring) call a **single shared function** `infer_one(text: str) -> InferenceResult` defined in `src/pipeline/inference.py`. This function is the only path that loads `winner.joblib` + `vectorizer.joblib`, applies `preprocess_text`, vectorizes, and predicts. `predict.py` is a thin argparse wrapper around it; `prediction.py` calls it in a loop (or via batched vectorizer transform built on the same preprocess). The CLI does not reimplement preprocessing. The validator can therefore prove consistency by exercising this one helper.

Checks (each is a separate function, each prints `[PASS]` or `[FAIL] <reason>`):

1. **Required artifacts exist**: list every required path (including `models/winner.meta.json` and `artifacts/artifacts_manifest.json`); for each missing, fail.
2. **JSON artifacts are valid**: load each `.json` artifact, ensure top-level structure has expected keys.
3. **Required dataset columns enforced**: read `train.csv`/`test.csv`, assert column presence (this verifies the *current* CSVs, in case the evaluator swapped them).
4. **Preprocessing function unchanged since training**: load `winner.meta.json`, read the recorded `preprocessing_probe = {input, expected}`, import `preprocess_text` from `src.pipeline.preprocessing`, and assert `preprocess_text(input) == expected`. If this fails, the function was modified after training and the saved vectorizer's vocabulary is no longer trustworthy. Also assert idempotence: `preprocess_text(preprocess_text(s)) == preprocess_text(s)`.
5. **CLI uses the same preprocessing as training** (the strong form of the consistency check): construct two probe strings whose raw form differs but whose post-preprocessing form is identical — e.g. `"  Hello   WORLD  "` and `"hello world"`. Invoke the CLI on each via subprocess with `--json`, parse the responses, and assert **the predicted label and score are byte-identical** between the two. Any divergence proves the CLI is skipping or duplicating preprocessing. Then invoke the CLI on a third probe whose raw form would tokenize differently from its preprocessed form (e.g. `"  THE\tApp\nIs   Easy"`); assert the response is identical to invoking the CLI on the already-preprocessed `"the app is easy"`. This three-way equality test catches the realistic failure modes: missing preprocessing, double preprocessing, and silently divergent preprocessing logic in the CLI.
6. **Shared-helper import check**: parse `predict.py`'s source text and assert it imports `infer_one` from `src.pipeline.inference` (and does not import `TfidfVectorizer`, does not redefine `preprocess_text`, and does not call `re.sub` on the input). This is a cheap static guard that complements the behavioural check above.
7. **≥3 models were trained** and the family requirement holds: read `metrics.json`, count entries with `status == "ok"`, assert ≥3, and assert the set of successful models covers both `family="linear"` and `family="probabilistic"` (using the registry). Confirm matching `models/model_<name>.joblib` files exist.
8. **Winner derived deterministically from saved metrics**: re-run the pure `select_winner()` function on `metrics.json` with the recorded `selection_metric`, assert result equals the winner recorded in `model_selection_report.json` and `winner.meta.json`.
9. **Test predictions cover all rows**: assert `len(test_predictions.csv) == len(test.csv)` and `set(ids)` matches exactly (no extras, no drops).
10. **CLI smoke test**: spawn `python predict.py --text "hello world" --json`, parse the JSON, assert it has `label` whose value appears in `metrics.json["labels"]`.

A final line prints `OK (N checks passed)` or `FAILED (k of N checks failed)`.

---

## 7. Corner Cases and Fallbacks — Master Table

| # | Scenario | Stage | Behaviour |
|---|---|---|---|
| 1 | Missing input file | DATA_LOADED | exit 2, named error |
| 2 | Bad encoding | DATA_LOADED | retry utf-8-sig → latin-1, record choice |
| 3 | Missing column in CSV | DATA_VALIDATED | report + exit 2 |
| 4 | <2 distinct labels | DATA_VALIDATED | report + exit 2 |
| 5 | Empty/whitespace text rows | DATA_VALIDATED | report + exit 2 |
| 6 | Duplicate IDs | DATA_VALIDATED | report + exit 2 |
| 7 | Malformed `config.json` | INIT | exit 2 |
| 8 | Missing `config.json` | INIT | exit 2 — the file is required by the spec, no silent default-fallback |
| 9 | Unknown model name in config | INIT | exit 2 with list of valid names |
| 9a | Config has <3 models / no linear / no probabilistic | INIT | exit 2 with the specific requirement that failed |
| 10 | Class with <2 samples for stratification | SPLIT_CREATED | fallback to non-stratified split, warn |
| 11 | Validation split would be empty | SPLIT_CREATED | force ≥1, warn |
| 12 | Class missing from validation | SPLIT_CREATED | safeguard warning |
| 13 | Train/val duplicate texts | SAFEGUARDS | warning with example IDs |
| 14 | Empty vocabulary after `min_df` | FEATURES_FIT | exit 3 with hint |
| 15 | A single model fails to train | MODELS_TRAINED | record failure, continue; abort only if <3 successes |
| 16 | Model lacks `predict_proba` | ERROR_ANALYSIS / CLI | use `decision_function`, else `null` |
| 17 | Zero misclassifications | ERROR_ANALYSIS | empty list + note, not an error |
| 18 | Tied primary metric | WINNER_SELECTED | macro_precision → alphabetical name |
| 19 | All models failed | WINNER_SELECTED | exit 3 with explanation |
| 20 | `top_k_error_examples` > #errors | ERROR_ANALYSIS | clamp silently |
| 21 | Test row text empty after preprocessing | DATA_VALIDATED / TEXT_PREPROCESSED | exit 2 — same rule as training data, no synthetic labels. The empty-after-strip check at `DATA_VALIDATED` catches this on raw text; the post-preprocessing re-check (§4.3) is a second gate. |
| 22 | CLI invoked before pipeline ran | CLI | exit 2 with "run run_pipeline.py first" |
| 23 | sklearn version mismatch breaks unpickling | CLI / Eval | catch, exit with "model artifact incompatible — retrain"; `requirements.txt` pins versions to make this rare |
| 24 | Class imbalance >5x | SAFEGUARDS | warning |
| 25 | CV folds > smallest class size | CROSS_VAL | reduce folds, record adjustment |
| 26 | Same model name twice in config | INIT | dedupe, warn |
| 27 | `validation_split` outside `(0.0, 1.0)` | INIT | exit 2 with named error — no silent rescaling, the configured fraction is used as-is |
| 28 | Non-UTF8 text inside CSV cells | preprocessing | preprocess_text handles via the encoding chosen at load; no decoding inside preprocess |
| 29 | IDs that look numeric vs string | DATA_LOADED | always read as string; preserve original for output |
| 30 | Windows path separators in manifest | reporting | use `pathlib.PurePosixPath` strings for portability of the JSON |

---

## 8. Reproducibility

- One seed in `config.json` is the only randomness source.
- `random.seed(seed)`, `np.random.seed(seed)` set at the top of `run_pipeline.py`.
- Every estimator that accepts `random_state` receives it.
- `train_test_split` receives `random_state=seed`.
- KFold receives `random_state=seed` with `shuffle=True`.
- Vectorizer is fit only on train split — never on full corpus — to keep evaluation honest.
- No timestamps used as features anywhere. The only timestamp is in `run_manifest.json` for human reference.
- Two consecutive runs on the same data + config produce byte-identical `metrics.json` and `test_predictions.csv`. A test in `tests/test_end_to_end.py` enforces this.

---

## 9. Testing Strategy

- `test_preprocessing.py` — idempotence, whitespace collapse, unicode safety, empty/`None` input.
- `test_selection.py` — tie on primary metric falls through to precision then alphabetical; raises when no successful models; respects `selection_metric` from config.
- `test_splitting.py` — stratification fallback when a class has 1 sample; warning on empty val split for a class.
- `test_end_to_end.py` — runs the full pipeline on the sample fixture in a tmp directory, asserts every artifact exists, asserts determinism by running twice and diffing JSON outputs.

Run with `pytest -q`. Not required for the evaluator, but `validate.py` runs the equivalent end-to-end check.

---

## 10. How the Evaluator Will Run It

```bash
# clean checkout
pip install -r requirements.txt

# (optional) replace train.csv / test.csv with fixtures
# (optional) edit config.json

python run_pipeline.py      # produces all artifacts
python validate.py          # confirms repo integrity
python predict.py --text "the app crashes on launch"
```

`README.md` documents this in 60 seconds at the top, followed by a brief description of each artifact.

---

## 11. Implementation Order (build order, not run order)

1. Skeleton: `state.py`, `config.py`, `logging_setup.py`, `io_paths.py`.
2. `data_loading.py` + `data_validation.py`. Wire into `run_pipeline.py`, smoke-test.
3. `preprocessing.py` with unit tests (single source of truth — get this right first).
4. `splitting.py` with stratification fallback tests.
5. `features.py` + `models.py` registry.
6. `training.py` with per-model failure isolation.
7. `evaluation.py` with full metrics + confusion matrix.
8. `selection.py` (pure, tested).
9. `artifacts.py` — ARTIFACTS_SAVED stage (vectorizer, models, winner copy, `winner.meta.json`, `artifacts_manifest.json`, existence assertion).
10. `inference.py` — shared `infer_one()` used by both CLI and prediction.
11. `error_analysis.py` with proba/decision/null score fallback.
12. `prediction.py` for test set — calls `inference.infer_one` (or its batched cousin).
13. `safeguards.py`.
14. `reporting.py` (manifest).
15. `predict.py` CLI — thin wrapper over `inference.infer_one`.
16. `validate.py` — last, because it depends on everything; includes the three-probe preprocessing-consistency test and the static import guard.
17. (Stretch) `cross_validation.py` + config flag.
18. README + final pass on log messages so stage names are visible.

---

## 12. Out-of-Scope (intentionally)

- No web service.
- No deep learning model. The spec explicitly says a large model is not needed.
- No hyperparameter search. Baselines stay baseline.
- No external API calls anywhere — training and inference are fully local.
- No persistent state beyond files in `artifacts/` and `models/`.
- No notebook in the main pipeline path (a notebook may be added under `notebooks/` for exploration, but it is not required and not run by `run_pipeline.py`).
