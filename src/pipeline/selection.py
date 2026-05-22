from __future__ import annotations

from typing import Any


EPS = 1e-12


class SelectionError(Exception):
    pass


def _format_reason(
    winner_name: str,
    winner_metrics: dict[str, Any],
    ranking: list[tuple[str, dict[str, Any]]],
    primary_metric: str,
    tie_break_applied: str,
) -> str:
    pm = primary_metric
    wv = winner_metrics[pm]
    if len(ranking) < 2:
        return (
            f"Selected {winner_name} as the only successful candidate "
            f"(no competitors); {pm}={wv:.6f}."
        )
    runner_name, runner_metrics = ranking[1]
    rv = runner_metrics[pm]
    if tie_break_applied == "none":
        return (
            f"Selected {winner_name} because it had the highest {pm} ({wv:.6f}). "
            f"Closest competitor {runner_name} at {rv:.6f}."
        )
    if tie_break_applied == "macro_precision":
        return (
            f"Tied with {runner_name} on {pm} ({wv:.6f}); selected on higher macro_precision "
            f"({winner_metrics['macro_precision']:.6f} vs {runner_metrics['macro_precision']:.6f})."
        )
    # alphabetical
    return (
        f"Tied with {runner_name} on {pm} ({wv:.6f}) and macro_precision "
        f"({winner_metrics['macro_precision']:.6f}); selected by alphabetical name order."
    )


def select_winner(metrics: dict[str, Any], primary_metric: str) -> dict[str, Any]:
    if primary_metric not in {"accuracy", "macro_precision", "macro_recall", "macro_f1"}:
        raise SelectionError(f"unknown selection_metric: {primary_metric}")

    candidates: list[tuple[str, dict[str, Any]]] = [
        (name, m) for name, m in metrics["models"].items() if m.get("status") == "ok"
    ]
    if not candidates:
        raise SelectionError("no successful models to choose from")

    candidates.sort(key=lambda kv: (
        -kv[1][primary_metric],
        -kv[1]["macro_precision"],
        kv[0],
    ))
    winner_name, winner_metrics = candidates[0]

    competitors = [m for _, m in candidates[1:]]
    tied_on_primary = [
        m for m in competitors
        if abs(m[primary_metric] - winner_metrics[primary_metric]) < EPS
    ]
    if not tied_on_primary:
        tie_break_applied = "none"
    else:
        tied_on_precision = [
            m for m in tied_on_primary
            if abs(m["macro_precision"] - winner_metrics["macro_precision"]) < EPS
        ]
        tie_break_applied = "alphabetical" if tied_on_precision else "macro_precision"

    reason = _format_reason(
        winner_name, winner_metrics, candidates, primary_metric, tie_break_applied
    )

    return {
        "winner": winner_name,
        "primary_metric": primary_metric,
        "primary_value": float(winner_metrics[primary_metric]),
        "ranking": [
            {
                "name": n,
                primary_metric: float(m[primary_metric]),
                "macro_precision": float(m["macro_precision"]),
            }
            for n, m in candidates
        ],
        "tie_break_applied": tie_break_applied,
        "reason": reason,
    }
