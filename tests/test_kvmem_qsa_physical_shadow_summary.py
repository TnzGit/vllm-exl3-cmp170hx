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
        "page_tokens": 16,
        "active_page0": 9968,
        "expected_qsa_layers": 12,
    }


def _stats():
    layer_ids = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47)
    return [
        {
            "layer_name": f"layer.{layer_id}",
            "attention_exact": True,
            "attention_max_abs": 0.0,
            "attention_mean_abs": 0.0,
            "attention_mismatch_elements": 0,
            "attention_elements": 1000,
            "input_mapping_exact": True,
            "input_tokens_compared": 100,
            "first_bad_input_token": None,
            "bootstrap_pages": 4096,
            "resident_physical_pages": 4160,
            "resident_page_tokens": 16,
            "full_block_tokens": 1568,
            "resident_table_width": 10080,
            "resident_cache_bytes": 136314880,
            "historical_selected": 100,
            "historical_resident_kept": 94,
            "historical_selected_dropped": 6,
        }
        for layer_id in layer_ids
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
    assert out["evidence"]["input_mapping_exact_all_records"] is True
    assert out["evidence"]["bootstrap_ok"] is True
    assert out["evidence"]["accounting_ok"] is True
    assert out["evidence"]["physical_geometry_ok"] is True
    assert out["evidence"]["resident_page_tokens"] == [16]
    assert out["evidence"]["full_block_tokens"] == [1568]
    assert out["evidence"]["cross_granularity_ratio"] == 98.0
    assert out["evidence"]["resident_cache_mib_per_layer"] == 130.0
    assert out["evidence"]["resident_cache_gib_all_layers"] == 1.5234375


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
    assert out["classification"] == "Q2A_MAPPING_EXACT_SEMANTIC_GO_NONEXACT"
    assert out["physical_shadow_go"] is True


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



def test_q2a_rejects_byte_inexact_selected_kv_mapping():
    mod = _load()
    rows = _stats()
    rows[0]["input_mapping_exact"] = False
    rows[0]["first_bad_input_token"] = 12345
    response = {
        "target_codes_in_order": True,
        "finish_reason": "stop",
        "usage": {"completion_tokens": 109},
        "text": "ok",
    }
    out = mod.summarize(response, rows, _plan())
    assert out["physical_shadow_go"] is False
    assert out["classification"] == "Q2A_INPUT_MAPPING_NO_GO"
    assert out["evidence"]["first_bad_input_tokens"] == [12345]


def test_q2a_numeric_nonexact_is_separate_when_inputs_are_byte_exact():
    mod = _load()
    rows = _stats()
    rows[0]["attention_exact"] = False
    rows[0]["attention_max_abs"] = 0.015625
    rows[0]["attention_mean_abs"] = 0.0002
    rows[0]["attention_mismatch_elements"] = 25
    response = {
        "target_codes_in_order": True,
        "finish_reason": "stop",
        "usage": {"completion_tokens": 109},
        "text": "violet-harbor-31, granite-comet-72",
    }
    out = mod.summarize(response, rows, _plan())
    assert out["mapping_gate"] is True
    assert out["semantic_gate"] is True
    assert out["physical_shadow_go"] is True
    assert out["classification"] == "Q2A_MAPPING_EXACT_SEMANTIC_GO_NONEXACT"
    assert out["evidence"]["attention_max_abs"] == 0.015625
    assert out["evidence"]["input_mapping_exact_all_records"] is True
