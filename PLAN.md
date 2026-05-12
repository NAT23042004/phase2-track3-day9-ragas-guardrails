# Lab 24 Completion Plan

**Summary**
- Complete Lab 24 by turning the existing Day 18 RAG code into a
submission repo with `phase-a/`, `phase-b/`, `phase-c/`, `phase-d/
`, `README.md`, and demo notes.
- Current repo has `data/` and Day 18-style `src/`, but is missing
`config.py`, dependency files, tests, phase folders, and all
required submission artifacts.
- Use `python3`, not `python`. Do not touch port `8082`; use Qdrant
on `6333`.

**Implementation Changes**
- Add project foundation:
- Create `config.py` from the Day 18 D6-C401 config, adjusted for
`lab24` paths and collection name.
- Add `requirements.txt` with Day 18 deps plus `langchain-
openai`, `langchain-community`, `presidio-analyzer`, `presidio-
anonymizer`, `groq`, `scikit-learn`, `pandas`, `datasets`.
- Create `README.md` with 200-300 word project overview and run
commands.

- Phase A, RAGAS:
- Create `phase-a/generate_testset.py` using RAGAS
`TestsetGenerator`, `simple/reasoning/multi_context`, and `data/
*.md`.
- Output `phase-a/testset_v1.csv` with at least `question`,
`ground_truth`, `contexts`, `evolution_type`.
- Add `phase-a/run_ragas_eval.py` that reuses
`src.pipeline.run_query()` and `src.m4_eval.evaluate_ragas()`.
- Write `phase-a/failure_analysis.md` from the bottom 10 scoring
questions.
- Write `phase-a/testset_review_notes.md` with manual review
notes for 10 questions.

- Phase B, judge and calibration:
- Create `phase-b/pairwise_judge.py`.
- Compare two answer variants: baseline retrieval answer without
reranker vs production answer with reranker.
- Implement swap-and-average by judging `(A,B)` and `(B,A)`, then
mapping both results back to stable labels.
- Output `phase-b/pairwise_results.csv`.
- Add `phase-b/human_labels.csv` for 10 manually labeled pairs.
- Add `phase-b/calibration.py` using
`sklearn.metrics.cohen_kappa_score`.
- Write `phase-b/judge_bias_report.md` with position-bias rate,
kappa score, and interpretation.

- Phase C, guardrails:
- Create `phase-c/input_guard.py` with Presidio `AnalyzerEngine`,
`AnonymizerEngine`, and custom regex recognizers for Vietnamese
phone numbers and CCCD.
- Add topic guardrail that allows finance/tax/privacy/RAG-lab
questions and rejects unrelated prompts.
- Create `phase-c/output_guard.py` using Groq Python SDK with
`GROQ_API_KEY` and a Llama Guard-compatible model/configured guard
prompt.
- Create `phase-c/guarded_pipeline.py` wrapping input redaction/
topic check, `run_query()`, output safety check, and latency
    - CI/CD gate: fail if RAGAS aggregate or guardrail tests drop
below thresholds.
- Add `demo/demo_script.md` for the 5-minute video.

- `python3 -m compileall -q src phase-a phase-b phase-c`
- `python3 -m pytest -q` after adding tests.
- Add focused tests:
- `tests/test_input_guard.py`: VN phone, CCCD, and normal text
redaction.
columns.
- `phase-b/pairwise_results.csv` exists and includes swapped
judgments.
- `phase-c` scripts can classify safe/unsafe examples.
- `phase-d/blueprint.md` contains at least 5 SLOs with concrete
thresholds.

allows it when no GPU is available.
- The target is the excellent path, not just 60/100, so all
required artifacts plus reports/tests are included.