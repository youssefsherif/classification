from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone


class StageFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        stage = getattr(record, "stage", "PIPELINE")
        return f"[{ts}] [{stage}] {record.getMessage()}"


def get_logger(name: str = "pipeline") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StageFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_stage(logger: logging.Logger, stage: str, message: str) -> None:
    logger.info(message, extra={"stage": stage})
