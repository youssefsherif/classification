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


def test_multiclass_small_val_size_enlarged():
    # 12 rows, 3 balanced classes, 0.2 -> val_size=2 which is < n_classes=3.
    # Must enlarge val_size to 3 and keep stratification.
    df = _df(["a"] * 4 + ["b"] * 4 + ["c"] * 4)
    _tr, va, rep = split_train_validation(df, 0.2, seed=42)
    assert rep["stratified"] is True
    assert len(va) >= 3
    assert any("enlarged" in w for w in rep["warnings"])


def test_multiclass_dataset_too_small_falls_back():
    # 4 rows, 3 classes. n_classes=3 but only n-1=3 rows available for val. Boundary.
    df = _df(["a", "b", "c", "a"])
    _tr, va, rep = split_train_validation(df, 0.25, seed=42)
    # Either stratified with enlarged val_size, or fallback to non-stratified —
    # both are valid; the key requirement is that it must not crash.
    assert len(va) >= 1


def test_label_breakdown_present():
    df = _df(["a"] * 5 + ["b"] * 5)
    _tr, _va, rep = split_train_validation(df, 0.4, seed=42)
    assert "a" in rep["labels"] and "b" in rep["labels"]
    assert rep["labels"]["a"]["train"] + rep["labels"]["a"]["validation"] == 5
