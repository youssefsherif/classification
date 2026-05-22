from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .inference import batch_predict


def predict_test_set(test_df: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    """Run inference on every row of test.csv via the shared inference helper,
    write `id,predicted_label` to disk, return summary stats.

    Assumes test_df has already been validated (no empty texts after preprocessing).
    """
    texts = test_df["text"].astype(str).tolist()
    labels, _scores = batch_predict(texts)
    if len(labels) != len(test_df):
        raise RuntimeError(
            f"prediction count {len(labels)} does not match test rows {len(test_df)}"
        )
    out = pd.DataFrame({
        "id": test_df["id"].astype(str).values,
        "predicted_label": labels,
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8")
    return {"row_count": len(out)}
