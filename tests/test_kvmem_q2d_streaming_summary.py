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
        "staging_pages": 128,
        "cpu_page_count": 10063,
        "query_row_batch": 64,
    }


def _worker():
    rows = []
    for layer in ("l0", "l1"):
        for first, last in ((0, 7), (8, 15), (16, 16)):
            published = (last + 1) // 16
            published_this = int(first == 8)
            rows.append({
                "event": "q2d_streaming_runtime",
                "layer": layer,
                "io_mode": "staged_copy",
                "first_pos": first,
                "last_pos": last,
                "query_rows": last - first + 1,
                "current_write_pages": 1,
                "resident_read_pages": published,
                "combined_real_pages": published + 1,
                "max_working_pages": published + 1,
                "physical_page_cap": 4160,
                "published_pages_total": published,
                "published_pages_this_forward": published_this,
                "cpu_roundtrip_pages_total": published,
                "cpu_roundtrip_exact": True,
                "direct_load_verified_pages_total": 0,
                "direct_consumer_syncs_total": 0,
                "d2h_bytes_total": published * 32768,
                "h2d_bytes_total": published * 32768,
                "d2h_jobs_total": published,
                "h2d_jobs_total": published,
                "reload_miss_pages": 0,
                "selected_history_pages": 0,
                "split_calls": 1,
                "forward_exposed_wall_seconds": 0.01,
                "write_mapping_wall_seconds": 0.001,
                "kv_update_submit_wall_seconds": 0.001,
                "publish_wall_seconds": 0.001,
                "selection_plan_wall_seconds": 0.001,
                "stage_history_wall_seconds": 0.001,
                "table_build_wall_seconds": 0.001,
                "attention_submit_wall_seconds": 0.001,
                "d2h_event_seconds": 0.001,
                "d2h_wall_seconds": 0.001,
                "d2h_prepare_seconds": 0.001,
                "d2h_submit_seconds": 0.001,
                "d2h_wait_seconds": 0.001,
                "d2h_finish_seconds": 0.001,
                "h2d_event_seconds": 0.001,
                "h2d_wall_seconds": 0.001,
                "h2d_prepare_seconds": 0.001,
                "h2d_submit_seconds": 0.001,
                "h2d_wait_seconds": 0.001,
                "h2d_finish_seconds": 0.001,
                "d2d_copy_submit_seconds": 0.001,
                "direct_consumer_sync_seconds": 0.001,
                "publication_oracle_gather_submit_seconds": 0.001,
                "stage_residency_lookup_seconds": 0.001,
                "stage_slot_assignment_seconds": 0.001,
                "stage_transfer_call_seconds": 0.001,
                "stage_table_delta_seconds": 0.001,
                "stage_touch_seconds": 0.001,
                "trace_wall_seconds": 0.001,
                "trace_records_total": published + 1,
                "trace_bytes_total": 100,
                "trace_truncated": False,
                "read_table_mode": "dynamic_cpu_history_plus_scheduler_writes",
                "write_partition": [0, 127],
                "read_partition": [128, 4159],
            })
    return rows


def _memory():
    rows = []
    for layer_id, layer in enumerate(("l0", "l1")):
        rows.append({
            "event": "q2e_memory_census",
            "layer": layer,
            "layer_id": layer_id,
            "io_mode": "staged_copy",
            "backing_tensor": "staging",
            "staging_role": "transfer_bounce",
            "page_bytes": 32768,
            "addressable_pages": 4160,
            "write_pages": 128,
            "read_pages": 4032,
            "null_pages": 1,
            "dedicated_tensor_bytes": 4161 * 32768,
            "staging_pages": 128,
            "staging_tensor_bytes": 128 * 32768,
            "cpu_backing_pages": 10063,
            "cpu_backing_logical_bytes": 10063 * 32768,
            "dynamic_table_bytes": 10063 * 4,
        })
    return rows


def _rows():
    return _memory() + _worker()


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
    result = _load().summarize(_response(), _rows(), _scheduler(), _plan())
    assert result["streaming_qualification_go"] is True
    assert result["classification"] == "Q2D_CPU_AUTHORITATIVE_STREAMING_SEMANTIC_GO"


def test_streaming_summary_rejects_missing_coverage_and_capacity():
    result = _load().summarize(_response(), _memory() + _worker()[:-1], _scheduler(), _plan())
    assert result["coverage_gate"] is False
    rows = _worker()
    rows[0]["combined_real_pages"] = 4161
    result = _load().summarize(_response(), _memory() + rows, _scheduler(), _plan())
    assert result["capacity_gate"] is False


def test_streaming_summary_rejects_cpu_or_scheduler_failure():
    rows = _worker()
    rows[-1]["cpu_roundtrip_exact"] = False
    result = _load().summarize(_response(), _memory() + rows, _scheduler(), _plan())
    assert result["cpu_authority_gate"] is False
    result = _load().summarize(_response(), _rows(), [], _plan())
    assert result["scheduler_gate"] is False


def test_streaming_summary_requires_exact_memory_census():
    rows = _rows()
    rows[0]["dedicated_tensor_bytes"] -= 1
    result = _load().summarize(_response(), rows, _scheduler(), _plan())
    assert result["memory_census_gate"] is False
    assert result["classification"] == "Q2D_STREAMING_EVIDENCE_INCOMPLETE"


def test_streaming_summary_validates_complete_trace_contract():
    trace = {
        "complete": True,
        "truncated": False,
        "records": 8,
        "history_pages": 0,
        "missing_pages": 2,
        "assigned_slots": 2,
        "max_query_rows": 8,
        "max_history_pages": 0,
        "per_layer_records": {"0": 4, "1": 4},
        "sha256": "a" * 64,
    }
    result = _load().summarize(
        _response(), _rows(), _scheduler(), _plan(), trace
    )
    assert result["trace_gate"] is True
    bad = dict(trace, records=7)
    result = _load().summarize(
        _response(), _rows(), _scheduler(), _plan(), bad
    )
    assert result["trace_gate"] is False


def test_streaming_summary_gates_paired_policy_digests():
    trace = {
        "complete": True,
        "truncated": False,
        "records": 8,
        "history_pages": 0,
        "missing_pages": 2,
        "assigned_slots": 2,
        "max_query_rows": 8,
        "max_history_pages": 0,
        "per_layer_records": {"0": 4, "1": 4},
        "sha256": "b" * 64,
    }
    baseline = _load().summarize(
        _response(), _rows(), _scheduler(), _plan(), trace
    )
    digest = baseline["scheduler_policy_digest"]
    result = _load().summarize(
        _response(), _rows(), _scheduler(), _plan(), trace,
        "staged_copy", "b" * 64, digest,
    )
    assert result["paired_trace_exact"] is True
    assert result["paired_policy_gate"] is True
    result = _load().summarize(
        _response(), _rows(), _scheduler(), _plan(), trace,
        "staged_copy", "c" * 64, digest,
    )
    assert result["paired_trace_exact"] is False
    assert result["paired_policy_gate"] is True
    result = _load().summarize(
        _response(), _rows(), _scheduler(), _plan(), trace,
        "staged_copy", "b" * 64, "d" * 64,
    )
    assert result["paired_policy_gate"] is False


def test_scheduler_policy_digest_excludes_completion_length():
    baseline = _load().summarize(
        _response(), _rows(), _scheduler(), _plan()
    )
    longer = _scheduler() + [
        {
            "event": "q2d_scheduler_assign",
            "logical_pages": [2],
            "write_ids": [2],
            "real_write_pages": 2,
            "peak_real_write_pages": 2,
        },
        {
            "event": "q2d_scheduler_reclaim",
            "processed_computed_tokens": 32,
            "freed_logical_pages": [1],
            "freed_write_ids": [1],
            "freed_pages": 1,
            "real_write_pages": 1,
            "peak_real_write_pages": 2,
        },
    ]
    result = _load().summarize(_response(), _rows(), longer, _plan())
    assert result["scheduler_policy_digest"] == baseline["scheduler_policy_digest"]
    assert (
        result["scheduler_full_request_digest"]
        != baseline["scheduler_full_request_digest"]
    )
    assert result["scheduler_prefill_boundary_gate"] is True


def test_streaming_summary_can_require_dynamic_table_oracle():
    result = _load().summarize(
        _response(), _rows(), _scheduler(), _plan(),
        require_dynamic_table_oracle=True,
    )
    assert result["dynamic_table_oracle_gate"] is False
    rows = _rows()
    checks = {"l0": 0, "l1": 0}
    for row in rows:
        if row.get("event") == "q2d_streaming_runtime":
            checks[row["layer"]] += int(row["split_calls"])
            row["dynamic_table_oracle_enabled"] = True
            row["dynamic_table_oracle_checks_total"] = checks[row["layer"]]
            row["dynamic_table_oracle_pages_total"] = 1
    result = _load().summarize(
        _response(), rows, _scheduler(), _plan(),
        require_dynamic_table_oracle=True,
    )
    assert result["dynamic_table_oracle_gate"] is True
    assert result["dynamic_table_oracle_check_mismatches"] == 0

    rows[-1]["dynamic_table_oracle_checks_total"] -= 1
    result = _load().summarize(
        _response(), rows, _scheduler(), _plan(),
        require_dynamic_table_oracle=True,
    )
    assert result["dynamic_table_oracle_gate"] is False
    assert result["dynamic_table_oracle_check_mismatches"] == 1


def test_streaming_summary_gates_expected_direct_io_mode():
    rows = _rows()
    for row in rows:
        row["io_mode"] = "direct_dedicated_slots"
        if row["event"] == "q2e_memory_census":
            row["backing_tensor"] = "dedicated_qsa_cache"
            row["staging_role"] = "allocated_unused"
        if row["event"] == "q2d_streaming_runtime":
            row["d2d_copy_submit_seconds"] = 0.0
            row["direct_load_verified_pages_total"] = 1
            row["direct_consumer_syncs_total"] = 1
    result = _load().summarize(
        _response(), rows, _scheduler(), _plan(), None,
        "direct_dedicated_slots",
    )
    assert result["io_mode_gate"] is True
    assert result["streaming_qualification_go"] is True
    result = _load().summarize(
        _response(), rows, _scheduler(), _plan(), None, "staged_copy"
    )
    assert result["io_mode_gate"] is False
