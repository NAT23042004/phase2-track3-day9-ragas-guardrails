"""Guarded wrapper around the RAG pipeline."""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_sibling_module(module_name: str, filename: str):
    module_path = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_input_guard = _load_sibling_module("phase_c_input_guard", "input_guard.py")
_output_guard = _load_sibling_module("phase_c_output_guard", "output_guard.py")
guard_input = _input_guard.guard_input
assess_output_safety = _output_guard.assess_output_safety


@dataclass
class GuardedResponse:
    status: str
    answer: str
    sanitized_query: str
    total_latency_ms: float
    input_latency_ms: float
    output_latency_ms: float
    pii_found: list[str]
    reason: str | None = None


def run_guarded_query(query: str, query_runner: Callable[[str], str]) -> GuardedResponse:
    start = time.perf_counter()

    input_start = time.perf_counter()
    input_result = guard_input(query)
    input_latency_ms = (time.perf_counter() - input_start) * 1000
    if not input_result.is_allowed:
        total_latency_ms = (time.perf_counter() - start) * 1000
        return GuardedResponse(
            status="blocked_input",
            answer="Query rejected by topic guardrail.",
            sanitized_query=input_result.sanitized_text,
            total_latency_ms=round(total_latency_ms, 2),
            input_latency_ms=round(input_latency_ms, 2),
            output_latency_ms=0.0,
            pii_found=input_result.pii_found,
            reason=input_result.rejection_reason,
        )

    answer = query_runner(input_result.sanitized_text)

    output_start = time.perf_counter()
    output_result = assess_output_safety(answer)
    output_latency_ms = (time.perf_counter() - output_start) * 1000
    total_latency_ms = (time.perf_counter() - start) * 1000

    if not output_result.is_safe:
        return GuardedResponse(
            status="blocked_output",
            answer="Generated output blocked by output guardrail.",
            sanitized_query=input_result.sanitized_text,
            total_latency_ms=round(total_latency_ms, 2),
            input_latency_ms=round(input_latency_ms, 2),
            output_latency_ms=round(output_latency_ms, 2),
            pii_found=input_result.pii_found,
            reason=output_result.rationale,
        )

    return GuardedResponse(
        status="ok",
        answer=answer,
        sanitized_query=input_result.sanitized_text,
        total_latency_ms=round(total_latency_ms, 2),
        input_latency_ms=round(input_latency_ms, 2),
        output_latency_ms=round(output_latency_ms, 2),
        pii_found=input_result.pii_found,
        reason=output_result.rationale,
    )


def build_default_query_runner() -> Callable[[str], str]:
    from src.pipeline import build_pipeline, run_query

    search, reranker, parent_lookup = build_pipeline()

    def _runner(query: str) -> str:
        answer, _ = run_query(query, search, reranker, parent_lookup)
        return answer

    return _runner


if __name__ == "__main__":
    runner = build_default_query_runner()
    response = run_guarded_query("Nghị định 13 quy định gì về dữ liệu cá nhân?", runner)
    print(response)
