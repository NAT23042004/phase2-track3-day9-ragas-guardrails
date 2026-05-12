# Demo Script

1. Show repository structure and explain the four lab phases.
2. Run `python3 phase-a/generate_testset.py` and open `phase-a/testset_v1.csv`.
3. Run `python3 phase-b/pairwise_judge.py` and highlight swap-and-average columns.
4. Run `python3 phase-c/adversarial_tests.py` to showcase guardrail robustness and latency profiling.
5. Show `phase-c/input_guard.py` redacting a Vietnamese phone number and CCCD.
6. Show `phase-c/guarded_pipeline.py` blocking an out-of-scope or unsafe example.
7. Close with `phase-d/blueprint.md` and the defined SLOs.
