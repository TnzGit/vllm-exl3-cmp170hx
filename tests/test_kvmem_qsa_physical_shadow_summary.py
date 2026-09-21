import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "kvmem_qsa_physical_shadow_summarize.py"


def _load():
    spec = importlib.util.spec_from_file_location("q2_summary", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def _plan():
    return {
        "budget_tokens": 65536,
        "resident_region_count": 256,
        "resident_page_count": 4096,
        "active_reserve_pages": 64,
        "physical_page_count": 4160,
        "expected_qsa_layers": 2,
    }


def _stats():
    return [
        {
            "layer_name": "layer.3",
            "attention_exact": True,
            "attention_max_abs": 0.0,
            "bootstrap_pages": 4096,
            "resident_physical_pages": 4160,
            "resident_page_tokens": 16,
            "full_block_tokens": 1568,
            "resident_table_width": 10080,
            "historical_selected": 100,
            "historical_resident_kept": 94,
            "historical_selected_dropped": 6,
        },
        {
            "layer_name": "layer.7",
            "attention_exact": True,
            "attention_max_abs": 0.0,
            "bootstrap_pages": 4096,
            "resident_physical_pages": 4160,
            "resident_page_tokens": 16,
            "full_block_tokens": 1568,
            "resident_table_width": 10080,
            "historical_selected": 100,
            "historical_resident_kept": 93,
            "historical_selected_dropped": 7,
        },
    ]


def test_q2a_go_requires_exact_attention_and_target():
    mod = _load()
    response = {
        "target_codes_in_order": True,
        "finish_reason": "stop",
        "usage": {"completion_tokens": 109},
        "text": "violet-harbor-31, granite-comet-72",
    }
    out = mod.summarize(response, _stats(), _plan())
    assert out["physical_shadow_go"] is True
    assert out["classification"] == "Q2A_PHYSICAL_EXACT_GO"
    assert out["evidence"]["attention_max_abs"] == 0.0
    assert out["evidence"]["bootstrap_ok"] is True
    assert out["evidence"]["accounting_ok"] is True
    assert out["evidence"]["physical_geometry_ok"] is True
    assert out["evidence"]["resident_page_tokens"] == [16]
    assert out["evidence"]["full_block_tokens"] == [1568]
    assert out["evidence"]["cross_granularity_ratio"] == 98.0


def test_q2a_rejects_any_attention_difference():
    mod = _load()
    rows = _stats()
    rows[1]["attention_exact"] = False
    rows[1]["attention_max_abs"] = 0.001
    response = {
        "target_codes_in_order": True,
        "finish_reason": "stop",
        "usage": {"completion_tokens": 109},
        "text": "ok",
    }
    out = mod.summarize(response, rows, _plan())
    assert out["physical_shadow_go"] is False
    assert out["classification"] == "Q2A_PHYSICAL_NO_GO"


def test_q2a_rejects_incomplete_layer_bootstrap():
    mod = _load()
    response = {
        "target_codes_in_order": True,
        "finish_reason": "stop",
        "usage": {"completion_tokens": 109},
        "text": "ok",
    }
    out = mod.summarize(response, _stats()[:1], _plan())
    assert out["physical_shadow_go"] is False
    assert out["evidence"]["layer_coverage_ok"] is False
