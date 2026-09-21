from types import SimpleNamespace

import torch

import vllm_exl3.exl3 as exl3


class FakeExt:
    def __init__(self):
        self.calls = []

    def exl3_moe_coop(self, *args):
        self.calls.append(args)


def _fixture():
    x = torch.randn(4, 2560, dtype=torch.bfloat16)
    weights = torch.full((4, 10), 0.1, dtype=torch.bfloat16)
    local = torch.arange(40, dtype=torch.long) % 8
    layer = SimpleNamespace(
        _exl3_codebook_flags=(True, False, True, False, True, False),
        _exl3_k=3,
    )
    ptrs = {
        name: torch.zeros(8, dtype=torch.int64)
        for name in (
            "gate_trellis", "gate_suh", "gate_svh",
            "up_trellis", "up_suh", "up_svh",
            "down_trellis", "down_suh", "down_svh",
        )
    }
    temps = (
        torch.empty(1, 1, 1),
        torch.empty(1, 1, 1),
        torch.empty(1, 1, 640),
        torch.empty(1, 1, 640),
    )
    return x, weights, local, layer, ptrs, temps


def test_early_coop_default_off(monkeypatch):
    monkeypatch.setattr(exl3, "_COOP", True)
    monkeypatch.setattr(exl3, "_COOP_EARLY_PRELUDE", False)
    ext = FakeExt()
    x, weights, local, layer, ptrs, temps = _fixture()
    out = exl3._try_exl3_coop_early(
        x, weights, local, layer, ptrs, temps, 8, 10, None, ext
    )
    assert out is None
    assert ext.calls == []


def test_early_coop_calls_same_kernel_for_decode_shape(monkeypatch):
    monkeypatch.setattr(exl3, "_COOP", True)
    monkeypatch.setattr(exl3, "_COOP_EARLY_PRELUDE", True)
    monkeypatch.setattr(exl3, "_COOP_OUT_EMPTY", False)
    monkeypatch.setattr(exl3, "FAT_EXPERT_THRESHOLD", 256)

    ext = FakeExt()
    x, weights, local, layer, ptrs, temps = _fixture()
    out = exl3._try_exl3_coop_early(
        x, weights, local, layer, ptrs, temps, 8, 10, None, ext
    )

    assert out is not None
    assert out.shape == (4, 2560)
    assert out.dtype is torch.float32
    assert len(ext.calls) == 1

    args = ext.calls[0]
    assert args[0].dtype is torch.float16  # xh
    assert args[1].dtype is torch.int64    # sel
    assert args[2].dtype is torch.float16  # routing weights
    assert args[3:6] == (0, 8, 2560)


def test_early_coop_fails_closed_when_fat_route_could_exist(monkeypatch):
    monkeypatch.setattr(exl3, "_COOP", True)
    monkeypatch.setattr(exl3, "_COOP_EARLY_PRELUDE", True)
    monkeypatch.setattr(exl3, "FAT_EXPERT_THRESHOLD", 32)

    ext = FakeExt()
    x, weights, local, layer, ptrs, temps = _fixture()
    out = exl3._try_exl3_coop_early(
        x, weights, local, layer, ptrs, temps, 8, 10, None, ext
    )

    assert out is None
    assert ext.calls == []


def test_qualification_branch_drops_output_empty_probe():
    src = (
        __import__("pathlib").Path(exl3.__file__).read_text(encoding="utf-8")
    )
    assert "VLLM_EXL3_COOP_OUT_EMPTY" not in src
    assert "_COOP_OUT_EMPTY" not in src
    assert "out = torch.zeros(tokens, hidden, dtype=torch.float32, device=dev)" in src
