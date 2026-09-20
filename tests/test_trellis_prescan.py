from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors")
safetensors_torch = pytest.importorskip("safetensors.torch")

import vllm_exl3.exl3 as exl3


def _write_pack(tmp_path, *, aliases: bool):
    model = tmp_path / "model"
    model.mkdir()

    keys_by_shard = {"a.safetensors": {}, "b.safetensors": {}}
    weight_map = {}
    projs = (
        ("gate_proj", "up_proj", "down_proj")
        if aliases
        else ("w1", "w3", "w2")
    )
    for eid in range(2):
        shard = "a.safetensors" if eid == 0 else "b.safetensors"
        for proj in projs:
            key = (
                f"model.language_model.layers.0.ffn.experts.{eid}."
                f"{proj}.trellis"
            )
            tensor = torch.arange(
                4 * 4 * 48, dtype=torch.int16
            ).reshape(4, 4, 48)
            keys_by_shard[shard][key] = tensor
            weight_map[key] = shard

    # Same source layer/expert tuple under MTP must not make the main key
    # ambiguous.
    mtp_key = "mtp.layers.0.ffn.experts.0.w1.trellis"
    keys_by_shard["a.safetensors"][mtp_key] = torch.zeros(
        (4, 4, 64), dtype=torch.int16
    )
    weight_map[mtp_key] = "a.safetensors"

    for shard, tensors in keys_by_shard.items():
        safetensors_torch.save_file(tensors, model / shard)

    (model / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    return model


@pytest.mark.parametrize("aliases", [False, True])
def test_prescan_groups_header_opens_by_shard(tmp_path, monkeypatch, aliases):
    model = _write_pack(tmp_path, aliases=aliases)
    monkeypatch.setenv("VLLM_EXL3_MODEL_DIR", str(model))
    monkeypatch.setenv("VLLM_EXL3_ARENA_PRESCAN", "1")
    exl3._TRELLIS_INDEX_CACHE.clear()

    calls = []
    real_safe_open = safetensors.safe_open

    def counted_safe_open(path, *args, **kwargs):
        calls.append(str(path))
        return real_safe_open(path, *args, **kwargs)

    monkeypatch.setattr(safetensors, "safe_open", counted_safe_open)

    layer = SimpleNamespace(
        layer_name="model.layers.0.ffn.experts",
        starting_expert_offset=0,
    )
    shapes = exl3._try_prescan_trellis_shapes(layer, 2)

    assert shapes is not None
    assert shapes["gate"] == {0: (4, 4, 48), 1: (4, 4, 48)}
    assert shapes["up"] == {0: (4, 4, 48), 1: (4, 4, 48)}
    assert shapes["down"] == {0: (4, 4, 48), 1: (4, 4, 48)}

    # Six requested tensors, but only two shard header opens.
    assert len(calls) == 2
    assert {p.rsplit("/", 1)[-1] for p in calls} == {
        "a.safetensors",
        "b.safetensors",
    }


def test_prescan_requires_explicit_model_dir(monkeypatch):
    monkeypatch.delenv("VLLM_EXL3_MODEL_DIR", raising=False)
    monkeypatch.delenv("VLLM_ENGRAM_MODEL_DIR", raising=False)
    layer = SimpleNamespace(
        layer_name="model.layers.0.ffn.experts",
        starting_expert_offset=0,
    )
    assert exl3._try_prescan_trellis_shapes(layer, 2) is None


def test_trellis_index_cache_invalidates_on_index_change(tmp_path, monkeypatch):
    model = _write_pack(tmp_path, aliases=False)
    index = model / "model.safetensors.index.json"
    exl3._TRELLIS_INDEX_CACHE.clear()

    first = exl3._trellis_index_from_checkpoint(str(index))
    assert (False, 0, 0, "gate") in first

    payload = json.loads(index.read_text())
    payload["weight_map"].pop(
        "model.language_model.layers.0.ffn.experts.0.w1.trellis"
    )
    index.write_text(json.dumps(payload) + " ", encoding="utf-8")

    second = exl3._trellis_index_from_checkpoint(str(index))
    assert (False, 0, 0, "gate") not in second
