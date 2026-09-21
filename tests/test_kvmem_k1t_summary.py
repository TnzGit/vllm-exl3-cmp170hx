import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "kvmem_k1t_summarize.py"


def _load():
    spec = importlib.util.spec_from_file_location("k1t_summary", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def _probe(wall_ms=12.0, event_ms=11.5):
    return {
        "geometry": {"pages_per_region": 16},
        "transition": {
            "stage_in_gib": 72 / 1024,
            "stage_in_bytes": 72 * 1024 * 1024,
            "query_replacements_regions": 12,
            "stage_in_pages": 192,
        },
        "h2d_stage_in": {
            "median_event_ms": event_ms,
            "median_wall_ms": wall_ms,
        },
        "byte_verification": {"all_repeats_exact": True},
        "qsa_page_table": {
            "all_stage_in_mappings_exact": True,
            "all_evicted_pages_are_negative": True,
        },
        "cpu_backing": {"all_keys_hit_after_store": True},
    }


def test_summary_correctness_go_and_close_to_floor():
    mod = _load()
    out = mod.summarize(_probe(), 6.3494)
    assert out["transfer_correctness_go"] is True
    assert out["performance_class"] == "CLOSE_TO_RAW_FLOOR"
    assert out["expected_stage_in_pages"] == 192
    assert 10.0 < out["raw_copy_floor_ms"] < 12.5


def test_summary_perf_is_not_hard_correctness_gate():
    mod = _load()
    out = mod.summarize(_probe(wall_ms=80.0, event_ms=60.0), 6.3494)
    assert out["transfer_correctness_go"] is True
    assert out["performance_class"] == "HIGH_TRANSFER_OVERHEAD"


def test_summary_rejects_byte_mismatch():
    mod = _load()
    p = _probe()
    p["byte_verification"]["all_repeats_exact"] = False
    out = mod.summarize(p, 6.3494)
    assert out["transfer_correctness_go"] is False
