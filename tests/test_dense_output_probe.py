from types import SimpleNamespace

import torch

import vllm_exl3.exl3 as exl3


def _x(dtype):
    return SimpleNamespace(dtype=dtype)


def test_dense_output_probe_default_off(monkeypatch):
    monkeypatch.setattr(exl3, "_EXL3_DENSE_FP16_OUT_PROBE", False)
    layer = SimpleNamespace(_exl3_prefix="model.layers.0.attn.q_proj")
    assert exl3._dense_output_probe_dtype(layer, _x(torch.bfloat16), []) is torch.float32


def test_dense_output_probe_enables_only_bf16_pure_exl3_non_lm_head(monkeypatch):
    monkeypatch.setattr(exl3, "_EXL3_DENSE_FP16_OUT_PROBE", True)

    dense = SimpleNamespace(_exl3_prefix="model.layers.0.attn.q_proj")
    assert exl3._dense_output_probe_dtype(dense, _x(torch.bfloat16), []) is torch.float16

    # Other input dtypes stay on the historical FP32 epilogue.
    assert exl3._dense_output_probe_dtype(dense, _x(torch.float16), []) is torch.float32
    assert exl3._dense_output_probe_dtype(dense, _x(torch.float32), []) is torch.float32

    # Mixed BF16/EXL3 shards would introduce cat/promotion confounds.
    assert exl3._dense_output_probe_dtype(dense, _x(torch.bfloat16), [1]) is torch.float32

    # Draft/main lm_head is intentionally excluded from this probe.
    lm_head = SimpleNamespace(_exl3_prefix="model.mtp.lm_head")
    assert exl3._dense_output_probe_dtype(lm_head, _x(torch.bfloat16), []) is torch.float32
