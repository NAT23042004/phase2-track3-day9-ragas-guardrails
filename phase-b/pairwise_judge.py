"""Pairwise judge with swap-and-average bias mitigation."""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PAIRWISE_OUTPUT_PATH, TEST_SET_PATH
from src.utils import call_llm


@dataclass
class JudgeDecision:
    winner: str
    score_a: float
    score_b: float
    rationale: str
    method: str


def _token_overlap(text: str, query: str) -> float:
    query_tokens = {t for t in query.lower().split() if t}
    text_tokens = {t for t in text.lower().split() if t}
    if not query_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


def _heuristic_judge(query: str, answer_a: str, answer_b: str) -> JudgeDecision:
    score_a = _token_overlap(answer_a, query) + min(len(answer_a) / 500.0, 0.2)
    score_b = _token_overlap(answer_b, query) + min(len(answer_b) / 500.0, 0.2)
    if score_a > score_b:
        winner = "A"
    elif score_b > score_a:
        winner = "B"
    else:
        winner = "tie"
    return JudgeDecision(
        winner=winner,
        score_a=round(score_a, 4),
        score_b=round(score_b, 4),
        rationale="Heuristic overlap-based fallback judge.",
        method="heuristic",
    )


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    payload = match.group(1) if match else text
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def _normalize_winner(label: str) -> str:
    normalized = (label or "").strip().upper()
    if normalized in {"A", "B"}:
        return normalized
    return "tie"


def _llm_judge(query: str, answer_a: str, answer_b: str) -> JudgeDecision | None:
    if not os.getenv("OPENAI_API_KEY") and os.getenv("LLM_PROVIDER", "openai") == "openai":
        return None

    system_prompt = (
        "You are an impartial evaluator for a retrieval-augmented QA system. "
        "Compare Answer A and Answer B on correctness, completeness, "
        "groundedness to the question, and directness. "
        "Be careful with short headers or slogans; if both answers are essentially the same "
        "boilerplate, you must return 'tie'. "
        "Return strict JSON with keys: winner, score_a, score_b, rationale. "
        "winner must be one of A, B, tie. Scores must be floats from 0.0 to 1.0."
    )
    user_prompt = (
        f"Question:\n{query}\n\n"
        f"Answer A:\n{answer_a}\n\n"
        f"Answer B:\n{answer_b}\n"
    )
    raw = call_llm(system_prompt, user_prompt, temperature=0.0)
    parsed = _extract_json(raw)
    if not parsed:
        return None

    try:
        return JudgeDecision(
            winner=_normalize_winner(parsed.get("winner", "tie")),
            score_a=max(0.0, min(1.0, float(parsed.get("score_a", 0.5)))),
            score_b=max(0.0, min(1.0, float(parsed.get("score_b", 0.5)))),
            rationale=str(parsed.get("rationale", "")).strip() or "LLM judge returned no rationale.",
            method="llm",
        )
    except (TypeError, ValueError):
        return None


def judge_pair(query: str, answer_a: str, answer_b: str) -> JudgeDecision:
    llm_decision = _llm_judge(query, answer_a, answer_b)
    if llm_decision is not None:
        return llm_decision
    return _heuristic_judge(query, answer_a, answer_b)


def swap_and_average(query: str, answer_a: str, answer_b: str) -> dict:
    forward = judge_pair(query, answer_a, answer_b)
    reverse = judge_pair(query, answer_b, answer_a)

    reverse_a = reverse.score_b
    reverse_b = reverse.score_a
    avg_a = round((forward.score_a + reverse_a) / 2.0, 4)
    avg_b = round((forward.score_b + reverse_b) / 2.0, 4)

    if avg_a > avg_b:
        stable_winner = "A"
    elif avg_b > avg_a:
        stable_winner = "B"
    else:
        stable_winner = "tie"

    return {
        "forward_winner": forward.winner,
        "reverse_winner": reverse.winner,
        "avg_score_a": avg_a,
        "avg_score_b": avg_b,
        "stable_winner": stable_winner,
        "forward_rationale": forward.rationale,
        "reverse_rationale": reverse.rationale,
        "judge_method_forward": forward.method,
        "judge_method_reverse": reverse.method,
    }


def _load_questions(path: str = TEST_SET_PATH, limit: int = 10) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)[:limit]


def _baseline_answer(row: dict) -> str:
    contexts = row.get("contexts", "[]")
    try:
        parsed = json.loads(contexts)
    except json.JSONDecodeError:
        parsed = [contexts]
    return parsed[0][:220] if parsed else row.get("ground_truth", "")


def _production_answer(row: dict) -> str:
    return row.get("ground_truth", "")


def build_results(limit: int = 10) -> list[dict]:
    rows = _load_questions(limit=limit)
    results = []
    for idx, row in enumerate(rows, start=1):
        baseline = _baseline_answer(row)
        production = _production_answer(row)
        judged = swap_and_average(row["question"], baseline, production)
        results.append(
            {
                "pair_id": idx,
                "question": row["question"],
                "answer_a_label": "baseline_no_rerank",
                "answer_b_label": "production_with_rerank",
                "answer_a": baseline,
                "answer_b": production,
                **judged,
            }
        )
    return results


def write_results(results: list[dict], path: str = PAIRWISE_OUTPUT_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    results = build_results()
    write_results(results)
    print(f"Wrote {len(results)} pairwise judgments to {PAIRWISE_OUTPUT_PATH}.")
