import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "kvmem_q2d_reload_summarize.py"


def _load():
    spec = importlib.util.spec_from_file_location("q2d_summary", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _plan():
    return {
        "query_span": [0, 17],
        "scheduler_chunk_tokens": 8,
        "expected_qsa_layers": 2,
        "physical_page_count": 4160,
        "page_tokens": 16,
    }


def _rows():
    rows = []
    for layer in ("l0", "l1"):
        published = 0
        for first, last in ((0, 7), (8, 15), (16, 16)):
            published = (last + 1) // 16
            rows.append({
                "event": "q2d_reload_shadow",
                "layer": layer,
                "first_pos": first,
                "last_pos": last,
                "query_rows": last - first + 1,
                "max_working_pages": 4033,
                "physical_page_cap": 4160,
                "published_pages_total": published,
                "resident_slots": 4000,
                "peak_resident_slots": 4100,
                "stage_compare_exact": True,
                "attention_allclose": True,
                "attention_max_abs": 0.015625,
                "attention_mismatch_elements": 10,
                "reload_vs_stock_split_exact": True,
                "reload_vs_stock_split_allclose": True,
                "reload_vs_stock_split_max_abs": 0.0,
                "reload_vs_stock_split_mismatch_elements": 0,
                "reload_vs_stock_split_mean_abs": 0.0,
                "split_calls": 1,
                "reload_miss_pages": 2,
                "selected_history_pages": 3,
                "d2h_bytes_total": published * 32768,
                "h2d_bytes_total": 65536,
                "d2h_jobs_total": published,
                "h2d_jobs_total": 2,
            })
    return rows


def _response():
    return {
        "target_codes_in_order": True,
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 17},
    }


def test_reload_summary_requires_exact_coverage_and_passes_valid_evidence():
    result = _load().summarize(_response(), _rows(), _plan())
    assert result["reload_shadow_go"] is True
    assert result["classification"] == "Q2D_CPU_RELOAD_SHADOW_SEMANTIC_GO_NONEXACT"
    assert result["records"] == 6
    assert result["min_final_published_pages_per_layer"] == 1


def test_reload_summary_rejects_missing_span():
    result = _load().summarize(_response(), _rows()[:-1], _plan())
    assert result["reload_shadow_go"] is False
    assert result["coverage_gate"] is False


def test_reload_summary_rejects_capacity_or_stage_corruption():
    rows = _rows()
    rows[0]["max_working_pages"] = 4161
    result = _load().summarize(_response(), rows, _plan())
    assert result["capacity_gate"] is False
    rows = _rows()
    rows[-1]["stage_compare_exact"] = False
    result = _load().summarize(_response(), rows, _plan())
    assert result["stage_gate"] is False
