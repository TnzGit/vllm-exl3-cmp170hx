import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "kvmem_qsa_cpu_backed_summarize.py"


def _load():
    spec = importlib.util.spec_from_file_location("q2b_summary", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def _plan():
    return {
        "budget_tokens": 65536,
        "resident_page_count": 4096,
        "active_reserve_pages": 64,
        "physical_page_count": 4160,
        "publication_staging_pages": 128,
        "transfer_tensor_page_count": 4288,
        "page_tokens": 16,
        "active_page0": 9968,
        "expected_qsa_layers": 12,
    }


def _stats():
    layer_ids = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47)
    rows = []
    for layer_id in layer_ids:
        rows.append({
            "layer_name": f"layers.{layer_id}.self_attn.attn",
            "rows_applied": 1,
            "historical_selected": 100,
            "historical_resident_kept": 94,
            "historical_selected_dropped": 6,
            "bootstrap_pages": 4096,
            "resident_physical_pages": 4160,
            "resident_page_tokens": 16,
            "full_block_tokens": 1568,
            "resident_table_width": 10192,
            "resident_cache_bytes": 136314880,
            "cpu_backing_all_present": True,
            "bootstrap_all_pages_exact": True,
            "bootstrap_pages_compared": 4096,
            "first_bad_bootstrap_page": None,
            "d2h_publish_bytes": 134217728,
            "d2h_publish_jobs": 32,
            "d2h_event_seconds": 0.020,
            "d2h_wall_seconds": 0.021,
            "h2d_stage_in_bytes": 134217728,
            "h2d_event_seconds": 0.020,
            "h2d_wall_seconds": 0.021,
            "transfer_page_size_bytes": 32768,
            "transfer_tensor_bytes": 140509184,
            "bootstrap_source": "vllm_generic_cpu_offload",
            "input_mapping_exact": True,
            "input_tokens_compared": 100,
            "first_bad_input_token": None,
            "attention_exact": True,
            "attention_max_abs": 0.0,
            "attention_mean_abs": 0.0,
            "attention_mismatch_elements": 0,
            "attention_elements": 1000,
        })
    return rows


def _response():
    return {
        "target_codes_in_order": True,
        "finish_reason": "stop",
        "usage": {"completion_tokens": 109},
        "text": "violet-harbor-31, granite-comet-72",
    }


def test_q2b_exact_go_requires_real_cpu_round_trip_and_mapping():
    mod = _load()
    out = mod.summarize(_response(), _stats(), _plan())
    assert out["classification"] == "Q2B_CPU_BACKED_EXACT_GO"
    assert out["cpu_backed_go"] is True
    assert out["transfer_gate"] is True
    assert out["physical_mapping_go"] is True
    assert out["semantic_go"] is True
    assert out["transfer"]["d2h_total_gib"] == 1.5
    assert out["transfer"]["h2d_total_gib"] == 1.5
    assert out["transfer"]["transfer_tensor_mib_per_layer"] == 134.0
    assert out["evidence"]["input_mapping_exact_all_records"] is True


def test_q2b_allows_page_specialization_numeric_nonexact_after_hard_gates():
    mod = _load()
    rows = _stats()
    rows[0]["attention_exact"] = False
    rows[0]["attention_max_abs"] = 0.03125
    rows[0]["attention_mean_abs"] = 8e-5
    rows[0]["attention_mismatch_elements"] = 20
    out = mod.summarize(_response(), rows, _plan())
    assert out["classification"] == (
        "Q2B_CPU_BACKED_MAPPING_EXACT_SEMANTIC_GO_NONEXACT"
    )
    assert out["cpu_backed_go"] is True
    assert out["transfer_gate"] is True


def test_q2b_rejects_any_cpu_bootstrap_byte_failure():
    mod = _load()
    rows = _stats()
    rows[0]["bootstrap_all_pages_exact"] = False
    rows[0]["first_bad_bootstrap_page"] = 123
    out = mod.summarize(_response(), rows, _plan())
    assert out["classification"] == "Q2B_CPU_TRANSFER_NO_GO"
    assert out["cpu_backed_go"] is False
    assert out["transfer"]["first_bad_bootstrap_pages"] == [123]


def test_q2b_rejects_transfer_byte_count_mismatch():
    mod = _load()
    rows = _stats()
    rows[0]["h2d_stage_in_bytes"] -= 32768
    out = mod.summarize(_response(), rows, _plan())
    assert out["classification"] == "Q2B_CPU_TRANSFER_NO_GO"
    assert out["transfer_gate"] is False


def test_q2b_rejects_selected_input_mapping_failure_after_transfer():
    mod = _load()
    rows = _stats()
    rows[0]["input_mapping_exact"] = False
    rows[0]["first_bad_input_token"] = 87718
    out = mod.summarize(_response(), rows, _plan())
    assert out["classification"] == "Q2B_INPUT_MAPPING_NO_GO"
    assert out["cpu_backed_go"] is False



def test_q2b_rejects_semantic_failure_after_transfer_and_mapping_pass():
    mod = _load()
    response = _response()
    response["target_codes_in_order"] = False
    response["text"] = "wrong"
    out = mod.summarize(response, _stats(), _plan())
    assert out["transfer_gate"] is True
    assert out["physical_mapping_go"] is True
    assert out["classification"] == "Q2B_SEMANTIC_NO_GO"
    assert out["cpu_backed_go"] is False
