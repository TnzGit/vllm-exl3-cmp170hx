"""CPU exactness tests for all-expert routed metadata bulk commit."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import vllm_exl3.exl3 as exl3


class _MoeCfg:
    hidden_dim = 64
    num_experts = 4
    num_local_experts = 4
    experts_per_token = 2
    activation = "silu"
    rocm_aiter_fmoe_enabled = False
    swiglu_limit = None


class _Layer(torch.nn.Module):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.moe_config = _MoeCfg()
        self.layer_name = "layers.0.mlp.experts"
        self.global_num_experts = n
        self.local_num_experts = n
        self.starting_expert_offset = 0

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        eid = int(expert_id)
        return eid if 0 <= eid < self.local_num_experts else -1


def _new_method():
    method = object.__new__(exl3.Exl3MoEMethod)
    method.moe = _MoeCfg()
    method.quant_config = exl3.Exl3Config(bits=4, codebook="mul1", scope="test")
    method.bits = 4
    method._logged = False
    return method


def _reset_stats() -> None:
    stats = exl3._METADATA_BULK_AB_STATS
    for key in (
        "control_calls", "control_bytes", "deferred_calls", "deferred_bytes",
        "commit_calls", "commit_bytes", "committed_layers",
        "strided_commit_calls", "index_commit_calls",
    ):
        stats[key] = 0
    for key in ("control_wall_s", "deferred_stage_wall_s", "commit_wall_s"):
        stats[key] = 0.0
    for key in ("control_by_suffix", "deferred_by_suffix", "commit_by_suffix"):
        stats[key] = {}


def _make_layer(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("VLLM_EXL3_METADATA_BULK_AB", raising=False)
    monkeypatch.setenv("VLLM_EXL3_METADATA_FULL_BULK", "1")
    monkeypatch.setenv("VLLM_EXL3_ARENA_PRESCAN", "0")
    _reset_stats()
    method = _new_method()
    layer = _Layer(4)
    method.create_weights(
        layer,
        num_experts=4,
        hidden_size=64,
        intermediate_size_per_partition=64,
        params_dtype=torch.bfloat16,
    )
    layer.moe_tp_size = 1
    layer.tp_rank = 0
    for name in (
        "w13_suh", "w13_svh", "w13_mcg", "w13_mul1",
        "w2_suh", "w2_svh", "w2_mcg", "w2_mul1",
    ):
        getattr(layer, name).data.zero_()
    return method, layer


def _param(layer, proj: str, suffix: str):
    return (
        getattr(layer, f"w13_{suffix}")
        if proj in ("w1", "w3")
        else getattr(layer, f"w2_{suffix}")
    )


def test_full_bulk_preserves_exact_final_layout(monkeypatch: pytest.MonkeyPatch):
    method, layer = _make_layer(monkeypatch)
    expected = {}

    for eid in range(4):
        for proj_i, proj in enumerate(("w1", "w3", "w2")):
            suh = torch.full((64,), 10.0 * eid + proj_i + 1, dtype=torch.float16)
            svh = torch.full((64,), 20.0 * eid + proj_i + 1, dtype=torch.float16)
            mul1 = torch.tensor(exl3.MUL1_MARKER_SIGNED_INT32, dtype=torch.int32)
            for suffix, payload in (("suh", suh), ("svh", svh), ("mul1", mul1)):
                assert method._load_exl3(
                    _param(layer, proj, suffix),
                    payload,
                    f"experts.{eid}.{proj}.{suffix}",
                    shard_id=proj,
                    expert_id=eid,
                    return_success=True,
                )
                expected[(eid, proj, suffix)] = payload.clone()

    # Full mode defers even and odd experts alike.
    assert layer.w13_suh.count_nonzero().item() == 0
    assert layer.w13_mul1.count_nonzero().item() == 0

    exl3._metadata_bulk_ab_commit(layer)

    for eid in range(4):
        for proj in ("w1", "w3", "w2"):
            shard_idx = 0 if proj == "w1" else 1
            if proj in ("w1", "w3"):
                got_suh = layer.w13_suh[eid, shard_idx]
                got_svh = layer.w13_svh[eid, shard_idx]
                got_mul1 = layer.w13_mul1[eid, shard_idx]
            else:
                got_suh = layer.w2_suh[eid]
                got_svh = layer.w2_svh[eid]
                got_mul1 = layer.w2_mul1[eid]
            assert torch.equal(got_suh, expected[(eid, proj, "suh")])
            assert torch.equal(got_svh, expected[(eid, proj, "svh")])
            assert int(got_mul1.reshape(-1)[0].item()) == exl3.MUL1_MARKER_SIGNED_INT32

    stats = exl3.metadata_bulk_ab_stats()
    assert stats["mode"] == "full"
    assert stats["full_enabled"] is True
    assert stats["enabled"] is False
    assert stats["control_calls"] == 0
    assert stats["deferred_calls"] == 36
    assert stats["commit_calls"] == 9
    assert stats["commit_bytes"] == stats["deferred_bytes"]
    assert stats["strided_commit_calls"] == 9
    assert stats["index_commit_calls"] == 0
    assert stats["committed_layers"] == 1
    assert layer._exl3_metadata_bulk_ab_stage == {}


def test_full_and_ab_modes_are_mutually_exclusive(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_EXL3_METADATA_BULK_AB", "1")
    monkeypatch.setenv("VLLM_EXL3_METADATA_FULL_BULK", "1")
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        exl3._metadata_bulk_mode()


def test_full_mode_defers_every_expert(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("VLLM_EXL3_METADATA_BULK_AB", raising=False)
    monkeypatch.setenv("VLLM_EXL3_METADATA_FULL_BULK", "1")
    assert all(exl3._metadata_bulk_defer_expert(i) for i in range(8))
