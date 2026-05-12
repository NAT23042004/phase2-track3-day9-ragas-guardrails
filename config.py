"""Project configuration for Lab 24."""

from __future__ import annotations

import os

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")

TEST_SET_PATH = os.path.join(BASE_DIR, "phase-a", "testset_v1.csv")

HIERARCHICAL_PARENT_SIZE = int(os.getenv("HIERARCHICAL_PARENT_SIZE", "1800"))
HIERARCHICAL_CHILD_SIZE = int(os.getenv("HIERARCHICAL_CHILD_SIZE", "450"))
SEMANTIC_THRESHOLD = float(os.getenv("SEMANTIC_THRESHOLD", "0.55"))

QDRANT_HOST = os.getenv("QDRANT_HOST", "127.0.0.1")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "lab24_rag")

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GCP_EMBEDDING_MODEL = os.getenv("GCP_EMBEDDING_MODEL", "textembedding-gecko@003")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
DEFAULT_LLM = os.getenv("DEFAULT_LLM", "gpt-4o-mini")
FALLBACK_LLM = os.getenv("FALLBACK_LLM", "gpt-4.1-mini")
JUDGE_LLM = os.getenv("JUDGE_LLM", DEFAULT_LLM)
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")

BM25_TOP_K = int(os.getenv("BM25_TOP_K", "10"))
DENSE_TOP_K = int(os.getenv("DENSE_TOP_K", "10"))
HYBRID_TOP_K = int(os.getenv("HYBRID_TOP_K", "8"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))

ALLOWED_TOPICS = {
    "finance",
    "tax",
    "privacy",
    "personal_data",
    "rag",
    "lab24",
}

PHASE_A_MIN_ROWS = int(os.getenv("PHASE_A_MIN_ROWS", "50"))
PAIRWISE_OUTPUT_PATH = os.path.join(BASE_DIR, "phase-b", "pairwise_results.csv")
HUMAN_LABELS_PATH = os.path.join(BASE_DIR, "phase-b", "human_labels.csv")
