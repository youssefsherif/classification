from __future__ import annotations

import re

# Single source of truth. Imported by training, evaluation, prediction, and the CLI.
# Do not re-implement preprocessing anywhere else.

_WHITESPACE_RE = re.compile(r"\s+")

# Probe used by validate.py to confirm that the function in use at inference time
# matches the function used at training time. Persisted in winner.meta.json.
PROBE_INPUT = "  Hello   WORLD  "
PROBE_EXPECTED = "hello world"


def preprocess_text(s: str | None) -> str:
    """Deterministic text normalization.

    - None or non-string input -> empty string
    - strip leading/trailing whitespace
    - lowercase
    - collapse any run of internal whitespace (incl. tabs, newlines) to a single space
    """
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip().lower()
    s = _WHITESPACE_RE.sub(" ", s)
    return s
