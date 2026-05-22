from src.pipeline.preprocessing import (
    PROBE_EXPECTED,
    PROBE_INPUT,
    preprocess_text,
)


def test_probe_constants():
    assert preprocess_text(PROBE_INPUT) == PROBE_EXPECTED


def test_lowercase_and_strip():
    assert preprocess_text("  Hello WORLD  ") == "hello world"


def test_collapse_internal_whitespace():
    assert preprocess_text("a\tb\nc   d") == "a b c d"


def test_idempotence():
    samples = [
        "  Hello   WORLD\tfoo\nbar  ",
        "already normalized",
        "",
        "   ",
        "Mixed Whitespace",  # NBSP is treated like other whitespace by \s in Python regex
    ]
    for s in samples:
        once = preprocess_text(s)
        twice = preprocess_text(once)
        assert once == twice, f"not idempotent on {s!r}: {once!r} -> {twice!r}"


def test_none_and_non_string():
    assert preprocess_text(None) == ""
    assert preprocess_text(123) == "123"
