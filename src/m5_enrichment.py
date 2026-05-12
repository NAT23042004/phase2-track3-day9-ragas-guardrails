"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys, json
from dataclasses import dataclass, field


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import call_llm

@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────

def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    """
    if not text.strip():
        return ""
        
    sys_prompt = "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt."
    return call_llm(sys_prompt, text)


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────

def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    """
    if not text.strip():
        return []
        
    sys_prompt = f"Dựa trên đoạn văn sau, hãy tạo chính xác {n_questions} câu hỏi ngắn gọn mà đoạn văn này có thể trả lời. Mỗi câu hỏi nằm trên một dòng riêng biệt. Không đánh số thứ tự."
    raw_output = call_llm(sys_prompt, text)
    
    if not raw_output:
        return []
        
    questions = [q.strip("- ").strip("123456789. ") for q in raw_output.split("\n") if q.strip()]
    return questions[:n_questions]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──

def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    """
    if not text.strip():
        return text
        
    sys_prompt = "Viết 1 câu cực ngắn (dưới 20 từ) mô tả đoạn văn này nói về chủ đề gì trong tài liệu gốc. Chỉ trả về 1 câu duy nhất."
    user_prompt = f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}"
    
    context = call_llm(sys_prompt, user_prompt)
    if not context:
        return text
        
    return f"[Ngữ cảnh: {context}]\n\n{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────

def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, category.
    """
    if not text.strip():
        return {}
        
    sys_prompt = "Extract metadata từ đoạn văn dưới dạng JSON. Các field: topic (chủ đề), entities (danh sách tên người/tổ chức), category (loại tài liệu: policy|hr|it|finance). Trả về JSON nguyên bản, không dùng markdown block."
    
    raw_json = call_llm(sys_prompt, text)
    if not raw_json:
        return {}
        
    try:
        # Làm sạch chuỗi JSON nếu LLM trả về markdown code block
        clean_json = raw_json.strip().replace("```json", "").replace("```", "")
        return json.loads(clean_json)
    except Exception as e:
        print(f"Error parsing metadata JSON: {e}")
        return {}


def enrich_chunk_combined(text: str, document_title: str = "") -> dict:
    """
    Gộp Contextual, Summary, HyQA và Metadata vào 1 lần gọi LLM.
    Giảm số lần call từ 4 xuống 1.
    """
    if not text.strip():
        return {}

    sys_prompt = """
    Phân tích đoạn văn sau và trả về JSON với các trường:
    1. "context": 1 câu cực ngắn (< 20 từ) mô tả đoạn này nói về cái gì trong tài liệu.
    2. "summary": tóm tắt ngắn gọn 2 câu.
    3. "questions": list 3 câu hỏi mà đoạn văn trả lời.
    4. "metadata": dict chứa {topic, entities: [], category}.
    
    Trả về JSON nguyên bản, không dùng markdown block.
    """.strip()
    
    user_prompt = f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}"
    raw_json = call_llm(sys_prompt, user_prompt)
    
    try:
        clean_json = raw_json.strip().replace("```json", "").replace("```", "")
        return json.loads(clean_json)
    except Exception:
        return {}


def enrich_chunks(chunks: list[dict],
                  methods: list[str] | None = None) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline. Đã tối ưu hóa gộp call LLM.
    """
    if methods is None:
        methods = ["contextual", "hyqa", "metadata"]
    
    enriched = []
    print(f"Enriching {len(chunks)} chunks using Optimized Combined Strategy...")
    
    for i, c in enumerate(chunks):
        raw_text = c.get("text", "")
        old_metadata = c.get("metadata", {})
        doc_title = old_metadata.get("source", "")
        
        # Gộp tất cả vào 1 call
        res = enrich_chunk_combined(raw_text, document_title=doc_title)
        
        context = res.get("context", "")
        summary = res.get("summary", "")
        questions = res.get("questions", [])
        auto_meta = res.get("metadata", {})

        # Merge text
        enriched_text = raw_text
        if context:
            enriched_text = f"[Ngữ cảnh: {context}]\n\n{raw_text}"
        
        # Nếu có HyQA, ta có thể chọn prepend vào text để tăng khả năng search
        if questions:
            enriched_text += "\n\n[Câu hỏi giả định]: " + " ".join(questions)

        enriched.append(EnrichedChunk(
            original_text=raw_text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**old_metadata, **auto_meta},
            method="combined_optimized"
        ))
        
    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
