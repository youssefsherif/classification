#!/usr/bin/env python
"""Single-text inference CLI. Thin wrapper around src.pipeline.inference.infer_one."""

from __future__ import annotations

import argparse
import json
import sys

from src.pipeline.inference import InferenceError, infer_one


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Predict the class label for a single piece of text using the trained pipeline."
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", type=str, help="text to classify")
    src.add_argument("--stdin", action="store_true", help="read text from stdin")
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON output")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.stdin:
        text = sys.stdin.read()
    else:
        text = args.text
    try:
        result = infer_one(text)
    except InferenceError as e:
        msg = str(e)
        if args.json:
            print(json.dumps({"error": msg}))
        else:
            print(f"ERROR: {msg}", file=sys.stderr)
        return 2

    if args.json:
        out = {
            "label": result.label,
            "confidence": result.confidence,
            "score": result.score,
            "score_type": result.score_type,
            "model": result.model_name,
            "per_class_scores": result.per_class_scores,
            "raw_margin": result.raw_margin,
        }
        print(json.dumps(out))
    else:
        print(f"label: {result.label}")
        if result.confidence is not None:
            print(f"confidence: {result.confidence:.6f}")
        elif result.score is not None:
            print(f"score: {result.score:.6f} ({result.score_type})")
        else:
            print("confidence: null")
        print(f"model: {result.model_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
