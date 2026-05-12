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


pairwise_judge = _load_module("pairwise_judge", "phase-b/pairwise_judge.py")


def test_swap_and_average_keeps_stable_labels():
    result = pairwise_judge.swap_and_average(
        "Thuế GTGT phải nộp là bao nhiêu?",
        "Thuế GTGT phải nộp trong kỳ là 52.133.830 đồng.",
        "Tôi không chắc nội dung này nói về gì.",
    )
    assert result["stable_winner"] == "A"
    assert result["avg_score_a"] > result["avg_score_b"]


def test_swap_and_average_maps_reverse_back_to_a_b():
    result = pairwise_judge.swap_and_average(
        "Dữ liệu cá nhân là gì?",
        "Dữ liệu cá nhân là thông tin gắn với một con người cụ thể.",
        "Không liên quan.",
    )
    assert result["forward_winner"] in {"A", "B", "tie"}
    assert result["reverse_winner"] in {"A", "B", "tie"}
    assert result["stable_winner"] == "A"
