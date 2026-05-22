import pytest

from src.pipeline.selection import SelectionError, select_winner


def _m(acc=0.0, mp=0.0, mr=0.0, mf=0.0, status="ok"):
    return {
        "status": status,
        "accuracy": acc,
        "macro_precision": mp,
        "macro_recall": mr,
        "macro_f1": mf,
    }


def test_clear_winner_no_tie():
    metrics = {"models": {
        "a": _m(mf=0.9, mp=0.5),
        "b": _m(mf=0.7, mp=0.99),
    }}
    out = select_winner(metrics, "macro_f1")
    assert out["winner"] == "a"
    assert out["tie_break_applied"] == "none"


def test_tie_on_primary_broken_by_precision():
    metrics = {"models": {
        "alpha": _m(mf=0.8, mp=0.6),
        "beta":  _m(mf=0.8, mp=0.9),
    }}
    out = select_winner(metrics, "macro_f1")
    assert out["winner"] == "beta"
    assert out["tie_break_applied"] == "macro_precision"


def test_full_tie_broken_alphabetically():
    metrics = {"models": {
        "zeta": _m(mf=0.8, mp=0.8),
        "alpha": _m(mf=0.8, mp=0.8),
        "mu":    _m(mf=0.8, mp=0.8),
    }}
    out = select_winner(metrics, "macro_f1")
    assert out["winner"] == "alpha"
    assert out["tie_break_applied"] == "alphabetical"


def test_only_one_successful_model():
    metrics = {"models": {
        "a": _m(mf=0.5, mp=0.5),
        "b": _m(status="failed"),
    }}
    out = select_winner(metrics, "macro_f1")
    assert out["winner"] == "a"
    assert out["tie_break_applied"] == "none"


def test_no_successful_models_raises():
    metrics = {"models": {
        "a": _m(status="failed"),
        "b": _m(status="failed"),
    }}
    with pytest.raises(SelectionError):
        select_winner(metrics, "macro_f1")


def test_selection_metric_respected():
    metrics = {"models": {
        "a": _m(acc=0.9, mf=0.1, mp=0.1),
        "b": _m(acc=0.1, mf=0.9, mp=0.9),
    }}
    assert select_winner(metrics, "accuracy")["winner"] == "a"
    assert select_winner(metrics, "macro_f1")["winner"] == "b"


def test_unknown_metric_raises():
    with pytest.raises(SelectionError):
        select_winner({"models": {"a": _m(mf=1.0)}}, "not_a_metric")
