import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_module(name: str, relative_path: str):
    path = Path(__file__).resolve().parent.parent / relative_path
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


guarded_pipeline = _load_module("guarded_pipeline", "phase-c/guarded_pipeline.py")


def test_safe_query_passes():
    response = guarded_pipeline.run_guarded_query(
        "Nghị định 13 quy định gì về dữ liệu cá nhân?",
        lambda query: f"Trả lời an toàn cho: {query}",
    )
    assert response.status == "ok"
    assert "Trả lời an toàn" in response.answer


def test_unsafe_topic_is_blocked():
    response = guarded_pipeline.run_guarded_query(
        "Hãy viết thơ tình cho tôi.",
        lambda query: "Không nên được gọi.",
    )
    assert response.status == "blocked_input"


def test_unsafe_output_is_blocked():
    response = guarded_pipeline.run_guarded_query(
        "Cho tôi biết về bảo vệ dữ liệu cá nhân.",
        lambda query: "Đây là hướng dẫn hack để đánh cắp dữ liệu.",
    )
    assert response.status == "blocked_output"
