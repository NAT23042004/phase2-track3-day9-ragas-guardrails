"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os
import sys
import time
from dataclasses import dataclass
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    """Represents a single reranked document result.

    Attributes:
        text: The document text content.
        original_score: The original retrieval score before reranking.
        rerank_score: The new score assigned by the cross-encoder reranker.
        metadata: Additional metadata associated with the document.
        rank: The final rank position (0-based) after reranking.
    """
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    """Cross-encoder reranker using BAAI/bge-reranker-v2-m3.

    This model is multilingual and supports Vietnamese, making it suitable
    for the Vietnamese-specific handling requirement (A3). It uses a
    cross-encoder architecture that takes both query and document as input
    and outputs a relevance score, which is more accurate than bi-encoder
    similarity for reranking tasks.

    Attributes:
        model_name: The HuggingFace model identifier.
        _model: Lazily-loaded reranker model instance.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        """Initialize the cross-encoder reranker.

        Args:
            model_name: The HuggingFace model name for the reranker.
        """
        self.model_name = model_name
        self._model = None
        self.device = os.getenv("RERANK_DEVICE", "cpu")

    def _load_model(self):
        """Lazy-load the CrossEncoder model for cross-encoding.

        Uses sentence_transformers' CrossEncoder which wraps HuggingFace
        cross-encoder models. The model is loaded only once and cached
        in self._model. Uses fp16 for memory efficiency on GPU, falls
        back to CPU if no GPU available.

        Returns:
            The loaded CrossEncoder model instance.
        """
        if self._model is None:
            from sentence_transformers import CrossEncoder
            target_device = self.device
            if target_device == "auto":
                target_device = "cuda" if self._has_gpu() else "cpu"
            self._model = CrossEncoder(
                self.model_name,
                device=target_device,
            )
        return self._model

    @staticmethod
    def _has_gpu() -> bool:
        """Check if CUDA GPU is available for acceleration.

        Returns:
            True if CUDA is available, False otherwise.
        """
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = RERANK_TOP_K
    ) -> list[RerankResult]:
        """Rerank documents using the cross-encoder model.

        Takes a list of retrieved documents and reranks them by computing
        a relevance score for each (query, document) pair using the
        cross-encoder model. Results are sorted by score descending and
        truncated to top_k.

        Args:
            query: The user's search query string.
            documents: List of dicts with keys 'text', 'score', 'metadata'.
            top_k: Number of top results to return after reranking.

        Returns:
            A list of RerankResult objects sorted by rerank_score descending,
            limited to top_k items.
        """
        if not documents:
            return []

        model = self._load_model()
        # Build (query, document_text) pairs for the cross-encoder
        pairs = [(query, doc["text"]) for doc in documents]
        # predict returns a numpy array of logit scores
        scores = model.predict(pairs)

        # Convert to list of floats (handle numpy array and scalar)
        import numpy as np
        if isinstance(scores, np.ndarray):
            scores = scores.flatten().tolist()
        elif isinstance(scores, (float, int)):
            scores = [float(scores)]
        else:
            scores = [float(s) for s in scores]

        # Combine scores with documents, sort descending, take top_k
        scored = [(score, doc) for score, doc in zip(scores, documents)]
        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for rank, (score, doc) in enumerate(scored[:top_k]):
            results.append(RerankResult(
                text=doc["text"],
                original_score=doc.get("score", 0.0),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=rank,
            ))
        return results


class FlashrankReranker:
    """Lightweight reranker using FlashRank for fast inference.

    FlashRank is designed for speed (<5ms per query) and is useful as a
    lightweight alternative when latency is critical and a slight drop
    in reranking quality is acceptable. Falls back gracefully if the
    flashrank package is not installed.

    Attributes:
        _model: Lazily-loaded FlashRank Ranker instance, or None if unavailable.
    """

    def __init__(self):
        """Initialize the FlashRank reranker."""
        self._model = None

    def _load_model(self):
        """Lazy-load the FlashRank Ranker model.

        Returns:
            The Ranker instance, or None if flashrank is not installed.
        """
        if self._model is None:
            try:
                from flashrank import Ranker
                self._model = Ranker()
            except ImportError:
                # flashrank not installed — fallback gracefully
                return None
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = RERANK_TOP_K
    ) -> list[RerankResult]:
        """Rerank documents using FlashRank.

        Args:
            query: The user's search query string.
            documents: List of dicts with keys 'text', 'score', 'metadata'.
            top_k: Number of top results to return.

        Returns:
            A list of RerankResult objects sorted by rerank_score descending,
            limited to top_k items. Returns empty list if FlashRank is unavailable.
        """
        if not documents:
            return []

        model = self._load_model()
        if model is None:
            return []

        from flashrank import RerankRequest
        passages = [{"text": d["text"]} for d in documents]
        result = model.rerank(RerankRequest(query=query, passages=passages))

        results = []
        for rank, item in enumerate(result[:top_k]):
            original_doc = documents[rank] if rank < len(documents) else {}
            results.append(RerankResult(
                text=item.get("text", ""),
                original_score=original_doc.get("score", 0.0),
                rerank_score=float(item.get("score", 0.0)),
                metadata=original_doc.get("metadata", {}),
                rank=rank,
            ))
        return results


def benchmark_reranker(
    reranker,
    query: str,
    documents: list[dict],
    n_runs: int = 5
) -> dict:
    """Benchmark reranker latency over multiple runs.

    Measures the wall-clock time of reranker.rerank() using
    time.perf_counter() for high-resolution timing. Runs the reranker
    n_runs times and computes average, minimum, and maximum latency
    in milliseconds.

    Args:
        reranker: A reranker instance with a rerank(query, documents) method.
        query: The query string to benchmark with.
        documents: The document list to rerank.
        n_runs: Number of benchmark runs (default 5).

    Returns:
        A dict with keys 'avg_ms', 'min_ms', 'max_ms' containing the
        average, minimum, and maximum reranking latency in milliseconds.
    """
    times: list[float] = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed_ms = (time.perf_counter() - start) * 1000  # convert to ms
        times.append(elapsed_ms)

    return {
        "avg_ms": round(mean(times), 2),
        "min_ms": round(min(times), 2),
        "max_ms": round(max(times), 2),
    }


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
