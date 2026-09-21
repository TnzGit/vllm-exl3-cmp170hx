import importlib.util
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "kvmem_persistent_topk_diagnose.py"


def _load():
    spec = importlib.util.spec_from_file_location("topk_diag", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def test_register_prefers_stable_libtorch_extension(monkeypatch):
    mod = _load()
    calls = []

    def fake_import(name):
        calls.append(name)
        if name != "vllm._C_stable_libtorch":
            raise AssertionError("legacy fallback should not be reached")
        return object()

    monkeypatch.setattr(mod.importlib, "import_module", fake_import)
    monkeypatch.setattr(
        mod.torch,
        "ops",
        SimpleNamespace(_C=SimpleNamespace(persistent_topk=object())),
    )

    loaded = mod._register_persistent_topk()
    assert loaded == "vllm._C_stable_libtorch"
    assert calls == ["vllm._C_stable_libtorch"]


def test_register_falls_back_to_legacy_extension(monkeypatch):
    mod = _load()
    calls = []

    def fake_import(name):
        calls.append(name)
        if name == "vllm._C_stable_libtorch":
            raise ModuleNotFoundError(name)
        return object()

    monkeypatch.setattr(mod.importlib, "import_module", fake_import)
    monkeypatch.setattr(
        mod.torch,
        "ops",
        SimpleNamespace(_C=SimpleNamespace(persistent_topk=object())),
    )

    loaded = mod._register_persistent_topk()
    assert loaded == "vllm._C"
    assert calls == ["vllm._C_stable_libtorch", "vllm._C"]


def test_register_fails_closed_if_no_extension_registers_op(monkeypatch):
    mod = _load()

    def fake_import(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(mod.importlib, "import_module", fake_import)
    monkeypatch.setattr(
        mod.torch,
        "ops",
        SimpleNamespace(_C=SimpleNamespace()),
    )

    try:
        mod._register_persistent_topk()
    except RuntimeError as exc:
        assert "_C_stable_libtorch" in str(exc)
        assert "vllm._C" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
