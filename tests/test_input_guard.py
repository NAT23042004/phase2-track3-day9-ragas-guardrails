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


input_guard = _load_module("input_guard", "phase-c/input_guard.py")


def test_redacts_vn_phone():
    redacted, pii = input_guard.redact_pii("Liên hệ 0912 345 678 để hỏi về thuế GTGT.")
    assert "[REDACTED_PHONE]" in redacted
    assert "vn_phone" in pii


def test_redacts_cccd():
    redacted, pii = input_guard.redact_pii("CCCD của tôi là 001203123456 khi hỏi về dữ liệu cá nhân.")
    assert "[REDACTED_CCCD]" in redacted
    assert "cccd" in pii


def test_normal_text_unchanged():
    text = "Nghị định 13 quy định về bảo vệ dữ liệu cá nhân."
    redacted, pii = input_guard.redact_pii(text)
    assert redacted == text
    assert pii == []
