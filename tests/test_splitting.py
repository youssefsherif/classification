import pandas as pd

from src.pipeline.splitting import split_train_validation


def _df(labels):
    return pd.DataFrame({
        "id": [str(i) for i in range(len(labels))],
        "text": [f"text {i}" for i in range(len(labels))],
        "label": labels,
    })


def test_stratified_split_balanced():
    df = _df(["a"] * 10 + ["b"] * 10)
    tr, va, rep = split_train_validation(df, 0.2, seed=42)
    assert rep["stratified"] is True
    assert rep["train_size"] + rep["validation_size"] == 20
    assert set(tr["label"]) == {"a", "b"}


def test_singleton_class_falls_back_to_unstratified():
    df = _df(["a"] * 9 + ["b"])  # b has 1 sample only
    _tr, _va, rep = split_train_validation(df, 0.2, seed=42)
    assert rep["stratified"] is False
    assert any("stratification disabled" in w for w in rep["warnings"])


def test_tiny_dataset_forces_at_least_one_val():
    df = _df(["a", "b", "a"])  # 3 rows, 0.1 -> round(0.3) = 0
    _tr, va, rep = split_train_validation(df, 0.1, seed=42)
    assert len(va) >= 1
    assert any("forced validation_size" in w for w in rep["warnings"])


def test_label_breakdown_present():
    df = _df(["a"] * 5 + ["b"] * 5)
    _tr, _va, rep = split_train_validation(df, 0.4, seed=42)
    assert "a" in rep["labels"] and "b" in rep["labels"]
    assert rep["labels"]["a"]["train"] + rep["labels"]["a"]["validation"] == 5
