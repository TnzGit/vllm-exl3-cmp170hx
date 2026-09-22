import json

import torch

from vllm_exl3.kvmem_q2c_attribution import (
    FINGERPRINT_FIELDS,
    apply_progressive_visibility,
    tensor_bit_fingerprint,
    write_attribution_event,
)


def _plan():
    return {
        "page_tokens": 16,
        "resident_pages": [0, 2],
        "active_from_pos": 64,
        "apply_min_pos": 64,
    }


def test_observe_and_apply_share_exact_policy_and_fingerprint():
    original = torch.tensor([[0, 17, 33, 49, 65, -1], [1, 18, 34, 50, 66, -1]])
    positions = torch.tensor([64, 65])
    observed = original.clone()
    masked = original.clone()

    a = apply_progressive_visibility(
        _plan(), observed, positions, apply_mask=False
    )
    b = apply_progressive_visibility(
        _plan(), masked, positions, apply_mask=True
    )

    assert torch.equal(observed, original)
    assert b["would_drop"] == 4
    assert b["actual_dropped"] == 4
    assert a["actual_dropped"] == 0
    assert masked.tolist() == [[0, -1, 33, -1, 65, -1], [1, -1, 34, -1, 66, -1]]

    expected = original.clone()
    expected_obs = apply_progressive_visibility(
        _plan(), expected, positions, apply_mask=True
    )
    selection_fields = [
        field for field in FINGERPRINT_FIELDS if not field.startswith("output_")
    ]
    assert all(b[field] == expected_obs[field] for field in selection_fields)


def test_attribution_writer_is_default_off_and_jsonl(monkeypatch, tmp_path):
    monkeypatch.delenv("VLLM_QWEN_KVMEM_Q2C_ATTRIB_STATS_PATH", raising=False)
    write_attribution_event({"event": "selection", "layer": "x"})
    path = tmp_path / "attrib.jsonl"
    assert not path.exists()

    monkeypatch.setenv("VLLM_QWEN_KVMEM_Q2C_ATTRIB_STATS_PATH", str(path))
    write_attribution_event({"event": "selection", "layer": "x"})
    assert json.loads(path.read_text()) == {"event": "selection", "layer": "x"}


def test_shape_mismatch_fails_closed():
    selected = torch.zeros((2, 3), dtype=torch.int64)
    positions = torch.zeros((1,), dtype=torch.int64)
    try:
        apply_progressive_visibility(_plan(), selected, positions, apply_mask=True)
    except RuntimeError as exc:
        assert "row count mismatch" in str(exc)
    else:
        raise AssertionError("shape mismatch did not fail")


def test_output_bit_fingerprint_is_exact_and_sensitive():
    left = torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16)
    right = left.clone()
    assert tensor_bit_fingerprint(left) == tensor_bit_fingerprint(right)
    right[0, 1] = -3.0
    assert tensor_bit_fingerprint(left) != tensor_bit_fingerprint(right)
