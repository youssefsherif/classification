from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_TRAIN_COLS = ("id", "text", "label")
REQUIRED_TEST_COLS = ("id", "text")


class DataLoadError(Exception):
    pass


def _read_with_encoding_fallback(path: Path) -> tuple[pd.DataFrame, str]:
    """Try utf-8, then utf-8-sig (BOM), then latin-1. Records the chosen encoding."""
    encodings = ("utf-8", "utf-8-sig", "latin-1")
    last_err: Exception | None = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, dtype=str, keep_default_na=False)
            return df, enc
        except UnicodeDecodeError as e:
            last_err = e
            continue
    raise DataLoadError(f"could not decode {path} with any of {encodings}: {last_err}")


def load_csvs(train_path: Path, test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if not train_path.exists():
        raise DataLoadError(f"train.csv not found at '{train_path}'")
    if not test_path.exists():
        raise DataLoadError(f"test.csv not found at '{test_path}'")

    train_df, enc_train = _read_with_encoding_fallback(train_path)
    test_df, enc_test = _read_with_encoding_fallback(test_path)

    # Encoding stored is the "least common denominator" — if both files succeeded
    # with the same encoding, that's what we record; otherwise the train encoding wins.
    encoding_used = enc_train if enc_train == enc_test else f"train={enc_train};test={enc_test}"

    return train_df, test_df, encoding_used
