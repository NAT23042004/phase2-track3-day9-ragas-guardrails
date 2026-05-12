"""Production RAG Pipeline — Bài tập NHÓM: ghép M1+M2+M3+M4."""

import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import (RERANK_TOP_K, LLM_PROVIDER, DEFAULT_LLM, FALLBACK_LLM, 
                    GCP_PROJECT_ID, GCP_LOCATION)


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60)

    # Step 1: Load & Chunk (M1)
    print("\n[1/3] Chunking documents...")
    docs = load_documents()
    all_parents = []
    all_children = []
    
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        all_parents.extend(parents)
        all_children.extend(children)
    
    print(f"  {len(all_children)} child chunks from {len(docs)} documents")
    print(f"  {len(all_parents)} parent chunks for context")

    # Step 2: Optimized Enrichment (M5)
    # CHỈ làm giàu Parent chunks để tiết kiệm call LLM (Giảm từ ~400 xuống ~35 calls)
    print(f"\n[2/4] Enriching {len(all_parents)} PARENT chunks only (Optimized)...")
    parent_dicts = [{"text": p.text, "metadata": p.metadata} for p in all_parents]
    enriched_parents = enrich_chunks(parent_dicts)
    
    # Tạo lookup table cho parent (chứa text đã được làm giàu)
    parent_lookup = {}
    parent_context_map = {} # parent_id -> context_prepend
    
    for e in enriched_parents:
        pid = e.auto_metadata.get("parent_id")
        parent_lookup[pid] = e.enriched_text
        # Trích xuất context từ enriched_text (phần trong ngoặc [])
        import re
        match = re.search(r"\[Ngữ cảnh: (.*?)\]", e.enriched_text)
        if match:
            parent_context_map[pid] = match.group(1)

    # Cập nhật child chunks: thừa hưởng context từ parent
    final_chunks = []
    for child in all_children:
        pid = child.parent_id
        text = child.text
        if pid in parent_context_map:
            text = f"[Ngữ cảnh: {parent_context_map[pid]}]\n\n{text}"
        
        final_chunks.append({
            "text": text,
            "metadata": {**child.metadata, "parent_id": pid}
        })
    print(f"  Enrichment done (Selective + Combined). Processed {len(all_parents)} LLM calls.")

    # Step 3: Index (M2)
    print("\n[3/4] Indexing (BM25 + Dense)...")
    search = HybridSearch()
    search.index(final_chunks)

    # Step 4: Reranker (M3)
    print("\n[4/4] Loading reranker...")
    reranker = CrossEncoderReranker()

    return search, reranker, parent_lookup


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker,
              parent_lookup: dict[str, str], use_reranker: bool = True) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    results = search.search(query)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K) if use_reranker else []
    source_results = reranked if reranked else results[:3]
    contexts = []
    seen_parent_ids = set()
    
    for result in source_results:
        parent_id = result.metadata.get("parent_id")
        if parent_id and parent_id in parent_lookup:
            if parent_id in seen_parent_ids:
                continue
            contexts.append(parent_lookup[parent_id])
            seen_parent_ids.add(parent_id)
        else:
            contexts.append(result.text)

    # TODO (nhóm): Replace with LLM generation for better scores
    # --- LOGGING DIAGNOSTICS ---
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    trace_file = os.path.join(log_dir, "pipeline_trace.log")
    
    with open(trace_file, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"QUERY: {query}\n")
        f.write(f"{'-'*50}\n")
        f.write(f"STEP 1: HYBRID SEARCH (TOP {len(results)})\n")
        for i, r in enumerate(results[:5]):
            f.write(f"  [{i}] Score: {r.score:.4f} | Method: {r.method} | Source: {r.metadata.get('source')}\n")
            f.write(f"      Text: {r.text[:200]}...\n")
        
        f.write(f"\nSTEP 2: RERANKING (TOP {len(source_results)})\n")
        for i, r in enumerate(source_results):
            # Lấy điểm score (của SearchResult) hoặc rerank_score (của RerankResult)
            s = getattr(r, "rerank_score", getattr(r, "score", 0.0))
            f.write(f"  [{i}] Score: {s:.4f} | Source: {r.metadata.get('source')}\n")
            f.write(f"      Text: {r.text[:200]}...\n")

    # --- LLM Generation ---
    from src.utils import call_llm
    
    context_str = "\n\n".join(contexts)
    sys_prompt = (
        "Bạn là trợ lý RAG trả lời câu hỏi tiếng Việt dựa trên tài liệu được cung cấp. "
        "Chỉ sử dụng thông tin xuất hiện trực tiếp trong phần Context. "
        "Không sử dụng kiến thức bên ngoài, không suy luận ngoài context, không bịa thêm chi tiết. "
        "Trả lời trực tiếp, ngắn gọn, đúng trọng tâm câu hỏi. "
        "Nếu context không đủ thông tin để trả lời chắc chắn, chỉ trả lời đúng câu sau: "
        "'Không tìm thấy thông tin trong tài liệu.'"
    )
    user_prompt = f"Context:\n{context_str}\n\nCâu hỏi: {query}"
    
    answer = call_llm(sys_prompt, user_prompt)
    
    if not answer:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin trong tài liệu."

    with open(trace_file, "a", encoding="utf-8") as f:
        f.write(f"\nSTEP 3: GENERATED ANSWER\n")
        f.write(f"Answer: {answer}\n")
        f.write(f"{'='*50}\n")

    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker,
                      parent_lookup: dict[str, str]):
    """Run evaluation on test set."""
    print("\n[Eval] Running queries (Multi-threaded)...")
    test_set = load_test_set()
    questions, answers, all_contexts, ground_truths = [], [], [], []

    from concurrent.futures import ThreadPoolExecutor

    def process_item(idx_item):
        i, item = idx_item
        ans, ctx = run_query(item["question"], search, reranker, parent_lookup)
        print(f"  [{i+1}/{len(test_set)}] Xử lý xong: {item['question'][:50]}...")
        return item["question"], ans, ctx, item["ground_truth"]

    with ThreadPoolExecutor(max_workers=10) as executor:
        results_parallel = list(executor.map(process_item, enumerate(test_set)))

    for q, a, c, gt in results_parallel:
        questions.append(q)
        answers.append(a)
        all_contexts.append(c)
        ground_truths.append(gt)

    print("\n[Eval] Running RAGAS...")
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker, parent_lookup = build_pipeline()
    evaluate_pipeline(search, reranker, parent_lookup)
    print(f"\nTotal: {time.time() - start:.1f}s")
