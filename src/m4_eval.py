"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import csv
import math
import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON or CSV."""
    with open(path, encoding="utf-8") as f:
        if path.endswith(".csv"):
            rows = []
            reader = csv.DictReader(f)
            for row in reader:
                contexts = row.get("contexts", "[]")
                try:
                    row["contexts"] = json.loads(contexts)
                except json.JSONDecodeError:
                    row["contexts"] = [contexts]
                rows.append(row)
            return rows
        return json.load(f)


from src.utils import call_llm, get_embeddings
from config import DEFAULT_LLM, FALLBACK_LLM, JUDGE_LLM

try:
    from ragas.llms import BaseRagasLLM
    from langchain_core.outputs import LLMResult, Generation
    from ragas.embeddings import BaseRagasEmbeddings
    RAGAS_IMPORT_ERROR = None
except Exception as exc:
    BaseRagasLLM = object
    BaseRagasEmbeddings = object
    LLMResult = None
    Generation = None
    RAGAS_IMPORT_ERROR = exc

class VertexRagasLLM(BaseRagasLLM):
    def generate_text(self, prompt, n=1, temperature=1e-8, stop=None, callbacks=None):
        if RAGAS_IMPORT_ERROR is not None:
            raise RuntimeError(f"RAGAS dependencies unavailable: {RAGAS_IMPORT_ERROR}")
        # prompt is a PromptValue or string
        prompt_str = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
        res = call_llm("Bạn là quan tòa chấm điểm RAG.", prompt_str, temperature=temperature, model_name=JUDGE_LLM)
        
        # Làm sạch JSON bằng regex (mạnh mẽ hơn)
        import re
        match = re.search(r"(\{.*\})", res, re.DOTALL)
        clean_res = match.group(1) if match else res.strip()
        
        # DEBUG LOG
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "ragas_debug.txt"), "a", encoding="utf-8") as f:
            f.write(f"\n--- PROMPT ---\n{prompt_str}\n")
            f.write(f"--- RESPONSE ---\n{res}\n")
            f.write(f"--- CLEANED ---\n{clean_res}\n")
            f.write("-" * 50 + "\n")
            
        return LLMResult(generations=[[Generation(text=clean_res) for _ in range(n)]])
    
    async def agenerate_text(self, prompt, n=1, temperature=1e-8, stop=None, callbacks=None):
        return self.generate_text(prompt, n, temperature, stop, callbacks)
    
    def is_finished(self, response: LLMResult) -> bool:
        return True

class VertexRagasEmbeddings(BaseRagasEmbeddings):
    def embed_texts(self, texts):
        if RAGAS_IMPORT_ERROR is not None:
            raise RuntimeError(f"RAGAS dependencies unavailable: {RAGAS_IMPORT_ERROR}")
        return get_embeddings(texts)
    
    def embed_documents(self, texts):
        return self.embed_texts(texts)
    
    def embed_query(self, text):
        return self.embed_texts([text])[0]
        
    async def aembed_texts(self, texts):
        return self.embed_texts(texts)

    async def aembed_documents(self, texts):
        return self.embed_texts(texts)
        
    async def aembed_query(self, text):
        return self.embed_texts([text])[0]


def _safe_float(value, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric) or math.isinf(numeric):
        return default
    return numeric


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from datasets import Dataset
    import pandas as pd

    # 1. Prepare dataset
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
    dataset = Dataset.from_dict(data)

    # 2. Run evaluation
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    
    # Khởi tạo wrappers
    ragas_llm = VertexRagasLLM()
    ragas_emb = VertexRagasEmbeddings()

    from ragas.run_config import RunConfig
    run_config = RunConfig(max_workers=4, timeout=120)

    result = evaluate(dataset, metrics=metrics, llm=ragas_llm, embeddings=ragas_emb, run_config=run_config, raise_exceptions=False)

    # 3. Convert to EvalResult list
    df = result.to_pandas()
    per_question = []
    for _, row in df.iterrows():
        per_question.append(EvalResult(
            question=row.get("user_input", row.get("question", "")),
            answer=row.get("response", row.get("answer", "")),
            contexts=row.get("retrieved_contexts", row.get("contexts", [])),
            ground_truth=row.get("reference", row.get("ground_truth", "")),
            faithfulness=_safe_float(row.get("faithfulness", 0.0)),
            answer_relevancy=_safe_float(row.get("answer_relevancy", 0.0)),
            context_precision=_safe_float(row.get("context_precision", 0.0)),
            context_recall=_safe_float(row.get("context_recall", 0.0))
        ))

    # 4. Return results dictionary
    # Ragas result indexing returns a list of scores per row. We need the mean.
    def get_metric_mean(res, metric_name):
        scores = res[metric_name]
        if isinstance(scores, (list, pd.Series)):
            s = pd.Series(scores).apply(lambda x: _safe_float(x, 0.0))
            return float(s.mean())
        return _safe_float(scores, 0.0)

    return {
        "faithfulness": get_metric_mean(result, "faithfulness"),
        "answer_relevancy": get_metric_mean(result, "answer_relevancy"),
        "context_precision": get_metric_mean(result, "context_precision"),
        "context_recall": get_metric_mean(result, "context_recall"),
        "per_question": per_question
    }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if not eval_results:
        return []

    processed = []
    for res in eval_results:
        # Calculate avg score for sorting
        scores = [
            _safe_float(res.faithfulness),
            _safe_float(res.answer_relevancy),
            _safe_float(res.context_precision),
            _safe_float(res.context_recall),
        ]
        avg_score = sum(scores) / 4.0
        faithfulness = scores[0]
        answer_relevancy = scores[1]
        context_precision = scores[2]
        context_recall = scores[3]
        
        # Default values
        diagnosis = "Passing thresholds"
        fix = "None needed"
        worst_metric = ""
        score = 0.0
        
        # Map to diagnosis using Diagnostic Tree thresholds
        if faithfulness < 0.85:
            diagnosis = "LLM hallucinating"
            fix = "Tighten prompt, lower temperature"
            worst_metric = "faithfulness"
            score = faithfulness
        elif context_recall < 0.75:
            diagnosis = "Missing relevant chunks"
            fix = "Improve chunking or add BM25"
            worst_metric = "context_recall"
            score = context_recall
        elif context_precision < 0.75:
            diagnosis = "Too many irrelevant chunks"
            fix = "Add reranking or metadata filter"
            worst_metric = "context_precision"
            score = context_precision
        elif answer_relevancy < 0.80:
            diagnosis = "Answer doesn't match question"
            fix = "Improve prompt template"
            worst_metric = "answer_relevancy"
            score = answer_relevancy
        else:
            # If all metrics pass thresholds, still identify the lowest one for reporting
            metrics = {
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "context_precision": context_precision,
                "context_recall": context_recall,
            }
            worst_metric = min(metrics, key=metrics.get)
            score = metrics[worst_metric]

        processed.append({
            "question": res.question,
            "avg_score": avg_score,
            "worst_metric": worst_metric,
            "score": score,
            "diagnosis": diagnosis,
            "suggested_fix": fix
        })
    
    # Sort by avg_score ascending and take bottom_n
    processed.sort(key=lambda x: x["avg_score"])
    return processed[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    aggregate = {}
    for k, v in results.items():
        if k == "per_question":
            continue
        aggregate[k] = _safe_float(v, 0.0)

    sanitized_failures = []
    for failure in failures:
        sanitized_failures.append({
            **failure,
            "avg_score": _safe_float(failure.get("avg_score"), 0.0),
            "score": _safe_float(failure.get("score"), 0.0),
        })

    report = {
        "aggregate": aggregate,
        "num_questions": len(results.get("per_question", [])),
        "failures": sanitized_failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
