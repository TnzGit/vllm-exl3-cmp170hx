import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "kvmem_q2c_runtime_summarize.py"


def _load():
    spec = importlib.util.spec_from_file_location("q2c_summary", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _plan():
    return {
        "expected_qsa_layers": 2,
        "page_tokens": 16,
        "resident_page_count": 4,
        "active_reserve_pages": 2,
        "physical_page_count": 6,
        "apply_min_pos": 160,
        "active_page0": 10,
        "publication_staging_pages": 2,
    }


def _boot():
    return {
        "classification": "Q2C_DUAL_POOL_BOOT_GO",
        "dual_pool_boot_gate": True,
    }


def _worker(layer):
    return {
        "layer": layer, "phase": "shrunk", "logical_pages": 12,
        "scheduler_real_pages": 6, "physical_page_cap": 6,
        "resident_history_pages": 4, "active_real_pages": 2,
        "hole_pages": 6, "hole_unique_ids": 1, "null_block_id": 0,
        "historical_selected": 10, "historical_resident_kept": 4,
        "historical_selected_dropped": 6,
        "prefill_historical_selected": 100,
        "prefill_historical_selected_dropped": 20,
        "cpu_published": True,
        "cpu_restored_after_shrink": True, "cpu_restore_exact": True,
        "d2h_bytes": 4 * 32768, "d2h_jobs": 2,
        "h2d_bytes": 4 * 32768, "h2d_jobs": 2,
        "staging_pages": 2, "staging_bytes": 2 * 32768,
    }


def _scheduler():
    return [
        {
            "event": "q2c_scheduler_reclaim",
            "logical_row_pages": 8, "real_pages_before": 6,
            "real_pages_after": 5, "freed_pages": 1,
            "physical_page_cap": 6, "peak_real_pages": 6,
        },
        {
            "event": "q2c_scheduler_boundary",
            "logical_row_pages": 10, "real_pages_at_boundary": 4,
            "physical_page_cap": 6, "resident_history_pages": 4,
            "active_reserve_pages": 2, "peak_real_pages": 6,
        },
    ]


def test_q2c_runtime_summary_go():
    mod = _load()
    response = {
        "target_codes_in_order": True, "finish_reason": "stop",
        "usage": {"completion_tokens": 32}, "text": "ok",
    }
    out = mod.summarize(
        response, _scheduler(), [_worker("a"), _worker("b")], _plan(), _boot()
    )
    assert out["classification"] == "Q2C_FROZEN_PLAN_OWNERSHIP_SEMANTIC_GO"
    assert out["q2c_runtime_go"] is True
    assert out["scheduler_shrink_gate"] is True
    assert out["scheduler"]["peak_real_pages"] == 6
    assert out["scheduler"]["peak_within_cap"] is True
    assert out["scheduler"]["reclaimed_pages_total"] == 1
    assert out["worker_block_table_gate"] is True
    assert out["cpu_authority_gate"] is True
    assert out["visibility_gate"] is True
    assert out["semantic_gate"] is True


def test_q2c_runtime_summary_rejects_peak_above_cap():
    mod = _load()
    response = {"target_codes_in_order": True, "finish_reason": "stop"}
    sched = _scheduler()
    sched[0]["peak_real_pages"] = 7
    out = mod.summarize(
        response, sched, [_worker("a"), _worker("b")], _plan(), _boot()
    )
    assert out["classification"] == "Q2C_SCHEDULER_SHRINK_NO_GO"
    assert out["q2c_runtime_go"] is False


def test_q2c_runtime_summary_rejects_no_boundary():
    mod = _load()
    response = {"target_codes_in_order": True, "finish_reason": "stop"}
    out = mod.summarize(
        response, _scheduler()[:1], [_worker("a"), _worker("b")], _plan(), _boot()
    )
    assert out["classification"] == "Q2C_SCHEDULER_SHRINK_NO_GO"
    assert out["q2c_runtime_go"] is False


def test_q2c_runtime_summary_rejects_missing_dual_pool_boot_evidence():
    mod = _load()
    response = {
        "target_codes_in_order": True,
        "finish_reason": "stop",
        "usage": {"completion_tokens": 32},
    }
    bad_boot = {
        "classification": "Q2C_DUAL_POOL_BOOT_NO_GO",
        "dual_pool_boot_gate": False,
    }
    out = mod.summarize(
        response, _scheduler(), [_worker("a"), _worker("b")], _plan(), bad_boot
    )
    assert out["classification"] == "Q2C_DUAL_POOL_BOOT_EVIDENCE_NO_GO"
    assert out["q2c_runtime_go"] is False
    assert out["dual_pool_boot_gate"] is False
