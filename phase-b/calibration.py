"""Compute judge calibration against human labels."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import HUMAN_LABELS_PATH, PAIRWISE_OUTPUT_PATH


def _read_csv(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _label_to_int(label: str) -> int:
    mapping = {"A": 1, "B": -1, "tie": 0}
    return mapping[label]


def compute_kappa(human_rows: list[dict], judge_rows: list[dict]) -> float:
    from sklearn.metrics import cohen_kappa_score

    judge_map = {row["pair_id"]: row["stable_winner"] for row in judge_rows}
    shared = [row for row in human_rows if row["pair_id"] in judge_map]
    if not shared:
        raise ValueError("No overlapping pair_id values between human and judge rows.")

    human = [_label_to_int(row["human_winner"]) for row in shared]
    judge = [_label_to_int(judge_map[row["pair_id"]]) for row in shared]
    return float(cohen_kappa_score(human, judge))


def interpret_kappa(kappa: float) -> str:
    if kappa < 0:
        return "Worse than random."
    if kappa < 0.2:
        return "Very low agreement."
    if kappa < 0.4:
        return "Low agreement."
    if kappa < 0.6:
        return "Moderate agreement."
    return "Production-leaning agreement."


def write_report(kappa: float, judge_rows: list[dict], path: str = "phase-b/judge_bias_report.md") -> None:
    forward_position_a_wins = sum(1 for row in judge_rows if row["forward_winner"] == "A")
    reverse_position_a_wins = sum(1 for row in judge_rows if row["reverse_winner"] == "B")
    position_bias_rate = abs(forward_position_a_wins - reverse_position_a_wins) / max(len(judge_rows), 1)
    methods = sorted(
        {
            row.get("judge_method_forward", "unknown")
            for row in judge_rows
        }
        | {
            row.get("judge_method_reverse", "unknown")
            for row in judge_rows
        }
    )

    lines = [
        "# Judge Bias Report",
        "",
        f"- Pair count: {len(judge_rows)}",
        f"- Judge methods observed: {', '.join(methods)}",
        f"- Position-bias rate: {position_bias_rate:.2f}",
        f"- Cohen's kappa: {kappa:.3f}",
        f"- Interpretation: {interpret_kappa(kappa)}",
        "",
        "Swap-and-average was used by judging `(A,B)` and `(B,A)` before mapping scores back to stable labels.",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    human_rows = _read_csv(HUMAN_LABELS_PATH)
    judge_rows = _read_csv(PAIRWISE_OUTPUT_PATH)
    kappa = compute_kappa(human_rows, judge_rows)
    write_report(kappa, judge_rows)
    print(f"Cohen's kappa: {kappa:.3f}")
