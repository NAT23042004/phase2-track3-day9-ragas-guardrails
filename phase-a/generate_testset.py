"""Generate Phase A test set using RAGAS when available, else a local fallback."""

from __future__ import annotations

import csv
import json
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PHASE_A_MIN_ROWS, TEST_SET_PATH
from src.m1_chunking import load_documents


EVOLUTION_SEQUENCE = ["simple", "reasoning", "multi_context"]


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def _fallback_rows(test_size: int = PHASE_A_MIN_ROWS) -> list[dict]:
    docs = load_documents()
    paragraphs: list[dict] = []

    for doc in docs:
        for para in [p.strip() for p in doc["text"].split("\n\n") if p.strip()]:
            if len(para) < 60:
                continue
            paragraphs.append({"source": doc["metadata"]["source"], "text": para})

    if not paragraphs:
        raise RuntimeError("No markdown paragraphs found under data/.")

    rows: list[dict] = []
    for idx in range(test_size):
        evolution = EVOLUTION_SEQUENCE[idx % len(EVOLUTION_SEQUENCE)]
        primary = paragraphs[idx % len(paragraphs)]
        secondary = paragraphs[(idx + 1) % len(paragraphs)]
        primary_sentences = _split_sentences(primary["text"])
        secondary_sentences = _split_sentences(secondary["text"])
        lead = primary_sentences[0] if primary_sentences else primary["text"]

        if evolution == "simple":
            question = f"Theo tài liệu {primary['source']}, nội dung chính của đoạn này là gì?"
            ground_truth = lead
            contexts = [primary["text"]]
        elif evolution == "reasoning":
            question = (
                f"Từ đoạn trong {primary['source']}, có thể rút ra thông tin hoặc nghĩa vụ nào quan trọng nhất?"
            )
            ground_truth = " ".join(primary_sentences[:2]) if primary_sentences else primary["text"]
            contexts = [primary["text"]]
        else:
            question = (
                f"Kết hợp hai đoạn trong {primary['source']} và {secondary['source']}, "
                "những điểm liên quan nhất cần chú ý là gì?"
            )
            ground_truth = " ".join((primary_sentences[:1] + secondary_sentences[:1])).strip()
            contexts = [primary["text"], secondary["text"]]

        rows.append(
            {
                "question": question,
                "ground_truth": ground_truth,
                "contexts": json.dumps(contexts, ensure_ascii=False),
                "evolution_type": evolution,
            }
        )
    return rows


def _ragas_rows(test_size: int = PHASE_A_MIN_ROWS) -> list[dict]:
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.testset import TestsetGenerator
    from ragas.testset.synthesizers import (
        MultiHopAbstractQuerySynthesizer,
        MultiHopSpecificQuerySynthesizer,
        SingleHopSpecificQuerySynthesizer,
    )

    loader = DirectoryLoader("./data", glob="**/*.md", loader_cls=TextLoader)
    documents = loader.load()
    llm = ChatOpenAI(model="gpt-4o-mini")
    embeddings = OpenAIEmbeddings()
    generator = TestsetGenerator.from_langchain(
        llm=llm,
        embedding_model=embeddings,
    )
    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=generator.llm), 0.5),
        (MultiHopAbstractQuerySynthesizer(llm=generator.llm), 0.25),
        (MultiHopSpecificQuerySynthesizer(llm=generator.llm), 0.25),
    ]
    testset = generator.generate_with_langchain_docs(
        documents=documents,
        testset_size=test_size,
        query_distribution=query_distribution,
        raise_exceptions=True,
    )
    records = testset.to_pandas().to_dict(orient="records")
    rows = []
    for record in records:
        synth_name = str(
            record.get("synthesizer_name")
            or record.get("query_type")
            or record.get("evolution_type")
            or ""
        ).lower()
        if "single" in synth_name:
            evolution_type = "simple"
        elif "abstract" in synth_name:
            evolution_type = "reasoning"
        else:
            evolution_type = "multi_context"
        rows.append(
            {
                "question": record.get("user_input", record.get("question", "")),
                "ground_truth": record.get("reference", record.get("ground_truth", "")),
                "contexts": json.dumps(
                    record.get("reference_contexts", record.get("contexts", [])),
                    ensure_ascii=False,
                ),
                "evolution_type": evolution_type,
            }
        )
    return rows


def generate_rows(test_size: int = PHASE_A_MIN_ROWS) -> tuple[list[dict], str]:
    try:
        if os.getenv("OPENAI_API_KEY"):
            return _ragas_rows(test_size=test_size), "ragas"
    except Exception as exc:
        print(f"Falling back from RAGAS generation: {exc}")
    return _fallback_rows(test_size=test_size), "fallback"


def write_rows(rows: list[dict], path: str = TEST_SET_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question", "ground_truth", "contexts", "evolution_type"],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    rows, mode = generate_rows()
    random.seed(24)
    write_rows(rows)
    print(f"Wrote {len(rows)} rows to {TEST_SET_PATH} using {mode} mode.")
