from tools.kvmem_q2c_split_summarize import summarize


def _plan():
    return {
        "query_span": [48, 64],
        "scheduler_chunk_tokens": 32,
        "expected_qsa_layers": 2,
    }


def _response(ok=True):
    return {
        "target_codes_in_order": ok,
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 64},
    }


def _row(layer, first, last, *, exact=True):
    n = last - first + 1
    return {
        "mode": "d_full_split",
        "layer": layer,
        "first_pos": first,
        "last_pos": last,
        "query_rows": n,
        "split_rows": 16,
        "split_calls": (n + 15) // 16,
        "split_exact": exact,
        "split_elements": n * 8,
        "split_mismatch_elements": 0 if exact else 1,
        "split_max_abs": 0.0 if exact else 0.125,
    }


def _complete():
    return [
        _row(layer, first, last)
        for layer in ("a", "b")
        for first, last in ((0, 31), (32, 63))
    ]


def test_split_summary_requires_complete_exact_semantic_evidence():
    out = summarize(_response(), _complete(), _plan(), 16)
    assert out["classification"] == "Q2C_FULL_KV_ROW_SPLIT_EXACT_SEMANTIC_GO"
    assert out["split_qualification_go"] is True
    assert out["records"] == out["expected_records"] == 4


def test_split_summary_rejects_missing_chunk():
    out = summarize(_response(), _complete()[:-1], _plan(), 16)
    assert out["classification"] == "Q2C_FULL_KV_ROW_SPLIT_EVIDENCE_INCOMPLETE"
    assert out["split_qualification_go"] is False


def test_split_summary_rejects_same_forward_difference():
    rows = _complete()
    rows[-1] = _row("b", 32, 63, exact=False)
    out = summarize(_response(), rows, _plan(), 16)
    assert out["classification"] == "Q2C_FULL_KV_ROW_SPLIT_ATTENTION_NO_GO"
    assert out["split_qualification_go"] is False
