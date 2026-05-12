# Lab 24: Evaluation and Guardrails for a Vietnamese RAG Pipeline

This repository turns the earlier Day 18 Vietnamese RAG code into a Lab 24 submission focused on evaluation, judging, guardrails, and production planning. The source pipeline in `src/` still provides hierarchical chunking, hybrid retrieval, reranking, enrichment, and RAGAS hooks. On top of that, this repo now adds the four required lab phases: synthetic test-set generation and RAGAS evaluation in `phase-a/`, pairwise LLM-as-judge and calibration in `phase-b/`, multi-layer safety checks in `phase-c/`, and an operational blueprint in `phase-d/`.

The implementation is designed to run in two modes. When APIs and heavier dependencies are available, the scripts use RAGAS, OpenAI-compatible models, Presidio, and Groq-backed safety checks. When those services are missing, the code falls back to deterministic local heuristics so the repository still imports, tests, and demonstrates the intended workflow. That tradeoff is intentional because the original codebase was missing `config.py` and other submission scaffolding, which would otherwise make the lab repo structurally complete but operationally fragile.

Typical flow:

```bash
python3 phase-a/generate_testset.py
python3 phase-a/run_ragas_eval.py
python3 phase-b/pairwise_judge.py
python3 phase-b/calibration.py
python3 -m pytest -q
python3 -m compileall -q src phase-a phase-b phase-c tests
```

If you want full online behavior, set the relevant environment variables such as `OPENAI_API_KEY`, `GROQ_API_KEY`, and local `QDRANT_HOST`/`QDRANT_PORT`. The repository defaults to Qdrant on port `6333`, per the lab plan.
