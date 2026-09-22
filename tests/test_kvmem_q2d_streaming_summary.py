import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "kvmem_q2d_streaming_summarize.py"


def _load():
    spec = importlib.util.spec_from_file_location("q2d_stream_summary", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _plan():
    return {
        "query_span": [0, 17],
        "scheduler_chunk_tokens": 8,
        "expected_qsa_layers": 2,
        "page_tokens": 16,
        "physical_page_count": 4160,
        "write_page_count": 128,
        "read_cache_page_count": 4032,
    }


def _worker():
    rows = []
    for layer in ("l0", "l1"):
        for first, last in ((0, 7), (8, 15), (16, 16)):
            published = (last + 1) // 16
            rows.append({
                "event": "q2d_streaming_runtime",
                "layer": layer,
                "first_pos": first,
                "last_pos": last,
                "query_rows": last - first + 1,
                "current_write_pages": 1,
                "resident_read_pages": published,
                "combined_real_pages": published + 1,
                "max_working_pages": published + 1,
                "physical_page_cap": 4160,
                "published_pages_total": published,
                "cpu_roundtrip_pages_total": published,
                "cpu_roundtrip_exact": True,
                "d2h_bytes_total": published * 32768,
                "h2d_bytes_total": published * 32768,
                "d2h_jobs_total": published,
                "h2d_jobs_total": published,
                "reload_miss_pages": 0,
                "selected_history_pages": 0,
                "read_table_mode": "dynamic_cpu_history_plus_scheduler_writes",
                "write_partition": [0, 127],
                "read_partition": [128, 4159],
            })
    return rows


def _scheduler():
    return [
        {
            "event": "q2d_scheduler_assign",
            "logical_pages": [0, 1],
            "write_ids": [0, 1],
            "real_write_pages": 2,
            "peak_real_write_pages": 2,
        },
        {
            "event": "q2d_scheduler_reclaim",
            "freed_pages": 1,
            "freed_write_ids": [0],
            "real_write_pages": 1,
            "peak_real_write_pages": 2,
        },
    ]


def _response():
    return {
        "target_codes_in_order": True,
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 17},
    }


def test_streaming_summary_passes_complete_evidence():
    result = _load().summarize(_response(), _worker(), _scheduler(), _plan())
    assert result["streaming_qualification_go"] is True
    assert result["classification"] == "Q2D_CPU_AUTHORITATIVE_STREAMING_SEMANTIC_GO"


def test_streaming_summary_rejects_missing_coverage_and_capacity():
    result = _load().summarize(_response(), _worker()[:-1], _scheduler(), _plan())
    assert result["coverage_gate"] is False
    rows = _worker()
    rows[0]["combined_real_pages"] = 4161
    result = _load().summarize(_response(), rows, _scheduler(), _plan())
    assert result["capacity_gate"] is False


def test_streaming_summary_rejects_cpu_or_scheduler_failure():
    rows = _worker()
    rows[-1]["cpu_roundtrip_exact"] = False
    result = _load().summarize(_response(), rows, _scheduler(), _plan())
    assert result["cpu_authority_gate"] is False
    result = _load().summarize(_response(), _worker(), [], _plan())
    assert result["scheduler_gate"] is False
