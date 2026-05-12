"""Input guardrails: PII redaction and domain/topic filtering."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

VN_PHONE_RE = re.compile(r"(?<!\d)(?:\+84|84|0)(?:\d[\s.-]?){8,10}\d(?!\d)")
CCCD_RE = re.compile(r"(?<!\d)\d{12}(?!\d)")

ALLOWED_TOPIC_HINTS = (
    "thuế",
    "gtgt",
    "tài chính",
    "dữ liệu",
    "cá nhân",
    "bảo vệ",
    "nghị định",
    "chủ thể",
    "privacy",
    "tax",
    "finance",
    "rag",
    "lab",
)


@dataclass
class InputGuardResult:
    is_allowed: bool
    sanitized_text: str
    pii_found: list[str]
    rejection_reason: str | None = None


def redact_pii(text: str) -> tuple[str, list[str]]:
    pii_found: list[str] = []
    redacted = text

    if CCCD_RE.search(redacted):
        pii_found.append("cccd")
        redacted = CCCD_RE.sub("[REDACTED_CCCD]", redacted)

    if VN_PHONE_RE.search(redacted):
        pii_found.append("vn_phone")
        redacted = VN_PHONE_RE.sub("[REDACTED_PHONE]", redacted)

    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine

        analyzer = AnalyzerEngine()
        anonymizer = AnonymizerEngine()
        results = analyzer.analyze(text=redacted, language="en")
        if results:
            redacted = anonymizer.anonymize(text=redacted, analyzer_results=results).text
            pii_found.append("presidio")
    except Exception:
        pass

    return redacted, pii_found


def is_allowed_topic(text: str) -> bool:
    lowered = text.lower()
    return any(hint in lowered for hint in ALLOWED_TOPIC_HINTS)


def guard_input(text: str) -> InputGuardResult:
    sanitized_text, pii_found = redact_pii(text)
    allowed = is_allowed_topic(sanitized_text)
    return InputGuardResult(
        is_allowed=allowed,
        sanitized_text=sanitized_text,
        pii_found=pii_found,
        rejection_reason=None if allowed else "Out-of-scope topic for this lab pipeline.",
    )
