from tools.kvmem_q2c_attribution_summarize import (
    OUTPUT_FIELDS,
    SELECTION_FIELDS,
    summarize,
    validate_virtual_lifecycle,
)


def _response(ok: bool):
    return {
        "query_span": [32, 48],
        "target_codes_in_order": ok,
        "finish_reason": "stop",
    }


def _row(mode: str):
    row = {
        "event": "q2c_selection",
        "mode": mode,
        "layer": "layers.0.qsa",
        "first_pos": 0,
        "last_pos": 31,
    }
    for index, field in enumerate(SELECTION_FIELDS + OUTPUT_FIELDS, start=1):
        row[field] = index
    return row


def _scheduler():
    return [
        {
            "event": "q2c_scheduler_assign",
            "logical_pages": [0, 1],
            "virtual_ids": [0, 1],
            "generations": [1, 1],
        },
        {
            "event": "q2c_scheduler_reclaim",
            "freed_pages": 1,
            "freed_logical_pages": [1],
            "freed_virtual_ids": [1],
        },
        {
            "event": "q2c_scheduler_assign",
            "logical_pages": [2],
            "virtual_ids": [1],
            "generations": [2],
        },
    ]


def _q2c_summary():
    return {
        "scheduler_shrink_gate": True,
        "worker_block_table_gate": True,
        "cpu_authority_gate": True,
        "visibility_gate": True,
        "worker": {"write_mapping_exact_all": True},
    }


def _summarize(a=True, b=False, c=False, *, mutate_c=None):
    rows = {
        "a": [_row("a_full_original")],
        "b": [_row("b_full_masked")],
        "c": [_row("c_bounded")],
    }
    if mutate_c:
        rows["c"][0][mutate_c] += 1
    return summarize(
        {"a": _response(a), "b": _response(b), "c": _response(c)},
        rows,
        _scheduler(),
        _q2c_summary(),
        {"physical_page_count": 2},
    )


def test_matching_b_c_proves_progressive_policy_causal():
    out = _summarize()
    assert out["classification"] == "Q2C_ATTRIBUTION_PROGRESSIVE_POLICY_CAUSAL"
    assert out["conclusive"] is True
    assert out["b_c_selection"]["exact"] is True
    assert out["b_c_attention_output"]["exact"] is True


def test_b_pass_c_fail_points_to_bounded_implementation():
    out = _summarize(b=True, c=False)
    assert out["classification"] == "Q2C_ATTRIBUTION_BOUNDED_IMPLEMENTATION_NO_GO"
    assert out["conclusive"] is True


def test_cross_run_output_divergence_does_not_override_a_b_causal_control():
    out = _summarize(mutate_c="output_bit_sum")
    assert out["classification"] == "Q2C_ATTRIBUTION_PROGRESSIVE_POLICY_CAUSAL"
    assert out["conclusive"] is True
    assert out["b_c_selection"]["exact"] is True
    assert out["b_c_attention_output"]["exact"] is False


def test_virtual_lifecycle_rejects_reuse_before_free():
    rows = _scheduler()
    rows.insert(
        1,
        {
            "event": "q2c_scheduler_assign",
            "logical_pages": [3],
            "virtual_ids": [1],
            "generations": [2],
        },
    )
    out = validate_virtual_lifecycle(rows, 2)
    assert out["exact"] is False
    assert "live owner" in out["error"]
