import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "kvmem_qsa_shadow_probe.py"


def _load():
    spec = importlib.util.spec_from_file_location("shadow_probe", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def test_probe_normalizes_k0_case_schema():
    mod = _load()
    out = mod.normalize_case_metadata({
        "name": "k0",
        "target_tokens": 160000,
        "query_span": [159900, 160000],
        "needles": [{"marker": "A"}],
    })
    assert out["case"] == "k0"
    assert out["target_tokens"] == 160000
    assert out["needles"] == [{"marker": "A"}]


def test_probe_normalizes_k1a_turn_schema():
    mod = _load()
    out = mod.normalize_case_metadata({
        "name": "turn",
        "prompt_tokens": 159600,
        "query_span": [159488, 159600],
        "target_facts": [{"marker": "B"}],
    })
    assert out["case"] == "turn"
    assert out["target_tokens"] == 159600
    assert out["needles"] == [{"marker": "B"}]


def test_probe_rejects_missing_token_count():
    mod = _load()
    try:
        mod.normalize_case_metadata({
            "name": "bad",
            "query_span": [1, 2],
        })
    except ValueError as exc:
        assert "target_tokens" in str(exc)
    else:
        raise AssertionError("expected ValueError")
