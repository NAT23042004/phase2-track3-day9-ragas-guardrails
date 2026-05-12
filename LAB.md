# Lab 24 & Closing

**Hands-on: build full eval + guardrail blueprint cho RAG của Day 18, ready for production**

---

## Mục tiêu
Build complete eval + guardrail stack cho RAG pipeline từ Day 18, với latency budget và CI/CD-ready blueprint.

### Phase A: RAGAS (30')
1. Test set 50 q (3 distributions)
2. Run 4 metrics
3. Bottom 10 questions
4. Failure cluster analysis

### Phase B: Judge (30')
5. Pairwise pipeline
6. Swap-and-average
7. Cohen κ vs 10 human
8. Document biases

### Phase C: Guard (30')
9. Presidio PII + topic
10. Test 20 adversarial
11. Llama Guard 3 output
12. Measure P95 latency
13. Blueprint 1-pager

---

## Hướng dẫn xử lý sự cố (Troubleshooting)

| Triệu chứng | Cách xử lý |
| :--- | :--- |
| RAGAS scores rất thấp (<0.5) tất cả metrics | Check judge model, có thể sai API key, hoặc context format không đúng (list of strings) |
| Faithfulness cao nhưng AR thấp | Answer đúng context nhưng off-topic. Improve prompt instruction về relevance |
| CP thấp (<0.5) nhưng CR cao | Retrieval lấy đủ chunks nhưng rank sai. Add re-ranker (Cohere Rerank, RankGPT) |
| Llama Guard 3 quá restrictive | Default threshold strict. Custom categories trong system prompt, hoặc swap to Perspective API |
| Cohen κ < 0.4 với judge | Judge bias mạnh. Try swap-and-average, hoặc cross-judge protocol |
| Presidio không bắt PII tiếng Việt | Default model en-only. Add custom regex VN (CCCD, phone_vn) hoặc spaCy VN model |

> **Mẹo:** Bật DEBUG cho RAGAS và Llama Guard logger → thấy raw judge output, tìm ra root cause trong 5 phút.

---

## Tiêu chí đánh giá (Grading Breakdown)

### 1. RAGAS Evaluation (30%)
* Test set 50+ q (10)
* All 4 metrics computed (10)
* Failure cluster analysis (5)
* CI/CD integration plan (5)

### 2. Guardrails (25%)
* Presidio PII (5)
* Topic validator (5)
* Llama Guard 3 (10)
* Latency P95 measured (5)

### 3. LLM-Judge (25%)
* Pairwise + absolute (10)
* Bias mitigation (swap) (5)
* Cohen κ vs human (10)

### 4. Blueprint (20%)
* SLO definition (5)
* Architecture diagram (5)
* Alert playbook (5)
* Cost analysis (5)
