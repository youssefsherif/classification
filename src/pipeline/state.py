from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


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


_ORDER = list(PipelineState)


def assert_state(ctx: "StateContext", required: PipelineState) -> None:
    if _ORDER.index(ctx.state) < _ORDER.index(required):
        raise RuntimeError(
            f"stage precondition not met: need >= {required.value}, "
            f"current = {ctx.state.value}"
        )


@dataclass
class StateContext:
    state: PipelineState = PipelineState.INIT
    config: Any = None
    train_df: Any = None
    test_df: Any = None
    train_split_df: Any = None
    val_split_df: Any = None
    vectorizer: Any = None
    X_train: Any = None
    X_val: Any = None
    y_train: Any = None
    y_val: Any = None
    models: dict[str, Any] = field(default_factory=dict)
    model_failures: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    selection: dict[str, Any] = field(default_factory=dict)
    winner_name: str = ""
    stages_completed: list[str] = field(default_factory=list)
    encoding_used: str = "utf-8"
    file_hashes: dict[str, str] = field(default_factory=dict)
    cv_report: Any = None

    def advance(self, new_state: PipelineState) -> None:
        self.state = new_state
        if new_state.value not in self.stages_completed:
            self.stages_completed.append(new_state.value)
