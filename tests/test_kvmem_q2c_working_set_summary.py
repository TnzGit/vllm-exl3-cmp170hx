from tools.kvmem_q2c_working_set_summarize import summarize


def _plan():
    return {
        "query_span": [48, 64],
        "expected_qsa_layers": 2,
        "physical_page_count": 6,
        "resident_page_count": 4,
        "active_reserve_pages": 2,
        "page_tokens": 16,
        "scheduler_chunk_tokens": 32,
    }


def _row(
    layer, first, last, working, misses,
    batch512=5, batch256=3, batch128=3, batch64=2,
):
    row = {
        "mode": "a_full_original",
        "layer": layer,
        "first_pos": first,
        "last_pos": last,
        "query_rows": last - first + 1,
        "unique_pages_before": working - 1,
        "unique_historical_pages_before": working - 2,
        "unique_nonresident_historical_pages_before": misses,
        "unique_pages_with_current_writes_before": working,
        "row_batch_512_max_working_pages": batch512,
        "row_batch_256_max_working_pages": batch256,
        "row_batch_128_max_working_pages": batch128,
        "row_batch_64_max_working_pages": batch64,
    }
    for size, value in (
        (512, batch512), (256, batch256), (128, batch128), (64, batch64)
    ):
        row[f"row_batch_{size}_max_selected_pages"] = max(0, value - 1)
        row[f"row_batch_{size}_cold_h2d_pages_no_reuse"] = value
        row[f"row_batch_{size}_subbatches"] = 1
    return row


def test_working_set_requires_split_when_any_layer_chunk_exceeds_cap():
    rows = [
        _row("a", 0, 31, 5, 1),
        _row("a", 32, 63, 7, 3),
        _row("b", 0, 31, 6, 2),
        _row("b", 32, 63, 4, 1),
    ]
    out = summarize(rows, _plan())
    assert out["classification"] == "Q2C_RELOAD_ROW_BATCH_512_FEASIBLE"
    assert out["evidence_gate"] is True
    assert out["max_working_pages_with_current_writes"] == 7
    assert out["chunks_over_cap"] == 1
    assert out["frozen_contract_spare_pages"] == 0
    assert out["full_history_cpu_backing_pages_per_layer"] == 4
    assert out["largest_measured_feasible_query_row_batch"] == 512
    assert out["coverage_gate"] is True
    assert out["records"] == out["expected_records"] == 4
    assert out["row_batch_evidence"]["512"]["cold_h2d_pages_no_reuse"] == 20


def test_working_set_accepts_whole_chunk_at_or_below_cap():
    rows = [
        _row(layer, first, last, 6, 1)
        for layer in ("a", "b")
        for first, last in ((0, 31), (32, 63))
    ]
    out = summarize(rows, _plan())
    assert out["classification"] == "Q2C_RELOAD_WHOLE_CHUNK_FEASIBLE"
    assert out["whole_1024_query_chunk_feasible"] is True


def test_working_set_rejects_incomplete_chunk_coverage():
    rows = [_row("a", 0, 31, 5, 1), _row("b", 0, 31, 5, 1)]
    out = summarize(rows, _plan())
    assert out["classification"] == "Q2C_RELOAD_EVIDENCE_INCOMPLETE"
    assert out["evidence_gate"] is False
    assert out["missing_records"] == 2


def test_working_set_rejects_unexpected_prefill_records():
    rows = [
        _row(layer, first, last, 5, 1)
        for layer in ("a", "b")
        for first, last in ((0, 31), (32, 63))
    ]
    rows.append(_row("a", 0, 1, 2, 0))
    out = summarize(rows, _plan())
    assert out["classification"] == "Q2C_RELOAD_EVIDENCE_INCOMPLETE"
    assert out["unexpected_prefill_records"] == 1
