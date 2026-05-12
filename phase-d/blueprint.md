# Phase D Blueprint

## Scope

This lab blueprint covers a Vietnamese RAG system over tax and personal-data documents, with evaluation in Phase A, pairwise judging in Phase B, and guardrails in Phase C.

## SLOs

1. Faithfulness SLO: target `>= 0.85`, alert if `< 0.80` for 30 minutes.
2. Answer relevancy SLO: target `>= 0.80`, alert if `< 0.75` for 30 minutes.
3. Context precision SLO: target `>= 0.75`, alert if `< 0.70` for 30 minutes.
4. P95 end-to-end latency SLO: target `< 2.5s`, alert if `> 3.0s` for 5 minutes.
5. Guardrail false-block rate SLO: target `< 3%`, alert if `> 5%` for 1 hour.
6. Input redaction coverage SLO: target `>= 95%` on known PII patterns, alert if `< 90%`.

## Monitoring

- **Redaction Quality:** Log percentage of input queries containing PII-like patterns that are successfully redacted.
- **Judge Calibration:** Monitor Cohen's kappa between LLM judge and human review on a weekly sample of 20 pairs. If kappa drops below 0.4, trigger a prompt audit.
- **Bias Tracking:** Log position-bias rates (A vs B preference) for the pairwise judge. Alert if position preference exceeds 60/40 over 100 queries.
- **RAGAS Trends:** Persist aggregate Faithfulness and Relevancy scores in Prometheus/Grafana to detect performance drift after document updates.
- **Query Latency:** Track per-stage latency (Input Guard -> Retrieval -> Rerank -> Generation -> Output Guard).

## Incident Response

- **Low Faithfulness:** If faithfulness breaches 0.80, check for "context stuffing" or model hallucinations. Revert to a more restrictive prompt template or increase retrieval top-k.
- **High Latency:** If P95 latency > 3s, temporarily bypass the Rerank module or the Output Guard (if safety risk is low) while scaling inference capacity.
- **Judge Misalignment:** If Cohen's kappa < 0.2, stop using the LLM judge for automated release gating. Re-calibrate with a new few-shot prompt or transition to a stronger judge model (e.g., GPT-4o).
- **Guardrail Over-blocking:** If block rate > 5%, audit the "Topic Allowlist". If valid queries are blocked, refine the zero-shot classifier or add problematic keywords to a "Safe" exception list.
- **PII Leakage:** If raw PII is detected in logs, immediately update Presidio regex patterns and scrub the affected log entries.

## Release Gate

- Phase A aggregate metrics must remain above threshold.
- Guardrail tests must pass in CI.
- Pairwise judge calibration should remain at least moderate (`kappa >= 0.4`) before relying on it for monitoring.
