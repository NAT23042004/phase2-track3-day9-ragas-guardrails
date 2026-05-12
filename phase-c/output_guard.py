"""Output safety guard with Groq-backed and heuristic fallback modes."""

from __future__ import annotations

import os
from dataclasses import dataclass


UNSAFE_HINTS = (
    "hướng dẫn hack",
    "đánh cắp dữ liệu",
    "bypass bảo mật",
    "steal data",
    "exploit",
)


@dataclass
class OutputGuardResult:
    is_safe: bool
    label: str
    rationale: str


def _heuristic_check(text: str) -> OutputGuardResult:
    lowered = text.lower()
    if any(hint in lowered for hint in UNSAFE_HINTS):
        return OutputGuardResult(False, "unsafe", "Matched heuristic unsafe patterns.")
    return OutputGuardResult(True, "safe", "No unsafe heuristic patterns matched.")


def assess_output_safety(text: str) -> OutputGuardResult:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return _heuristic_check(text)

    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=os.getenv("GROQ_GUARD_MODEL", "llama-guard-3-8b"),
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the assistant output as safe or unsafe. "
                        "Respond with a single JSON object: "
                        '{"label":"safe|unsafe","rationale":"..."}'
                    ),
                },
                {"role": "user", "content": text},
            ],
        )
        content = completion.choices[0].message.content or ""
        lowered = content.lower()
        if '"unsafe"' in lowered or '"label":"unsafe"' in lowered:
            return OutputGuardResult(False, "unsafe", content)
        return OutputGuardResult(True, "safe", content)
    except Exception:
        return _heuristic_check(text)
