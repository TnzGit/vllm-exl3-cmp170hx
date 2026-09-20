from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import vllm_exl3.exl3 as exl3


class CountingMapping(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        return super().__iter__()


def test_expert_match_cache_preserves_first_match_and_reuses_success():
    mapping = CountingMapping(
        [
            ("first_param", "experts.0.gate_proj", 0, "w1"),
            ("later_param", "gate_proj", 0, "w1"),
        ]
    )
    cache: dict[str, tuple[str, str, int, str]] = {}

    first = exl3._match_expert_mapping(
        mapping,
        "layers.0.experts.0.gate_proj.trellis",
        cache,
    )
    assert first == mapping[0]
    assert mapping.iterations == 1

    # Same expert/projection base, different EXL3 payload. This must reuse the
    # exact first candidate selected by the original ordered scan.
    second = exl3._match_expert_mapping(
        mapping,
        "layers.0.experts.0.gate_proj.suh",
        cache,
    )
    assert second == mapping[0]
    assert mapping.iterations == 1


def test_expert_match_miss_is_not_cached():
    mapping = CountingMapping(
        [("other", "experts.9.down_proj", 9, "w2")]
    )
    cache: dict[str, tuple[str, str, int, str]] = {}

    assert (
        exl3._match_expert_mapping(
            mapping,
            "layers.0.experts.0.gate_proj.trellis",
            cache,
        )
        is None
    )
    assert cache == {}

    mapping.append(("wanted", "experts.0.gate_proj", 0, "w1"))
    matched = exl3._match_expert_mapping(
        mapping,
        "layers.0.experts.0.gate_proj.suh",
        cache,
    )
    assert matched == mapping[-1]
    assert cache


def test_expert_match_cache_can_be_disabled_without_semantic_change():
    mapping = CountingMapping(
        [("wanted", "experts.0.gate_proj", 0, "w1")]
    )

    a = exl3._match_expert_mapping(
        mapping,
        "layers.0.experts.0.gate_proj.trellis",
        None,
    )
    b = exl3._match_expert_mapping(
        mapping,
        "layers.0.experts.0.gate_proj.svh",
        None,
    )
    assert a == b == mapping[0]
    assert mapping.iterations == 2


def test_moe_layer_gc_policy_defaults_on(monkeypatch):
    monkeypatch.delenv("VLLM_EXL3_GC_AFTER_MOE_LAYER", raising=False)
    assert exl3._exl3_gc_after_moe_layer_enabled() is True


@pytest.mark.parametrize(("value", "expected"), [("0", False), ("1", True)])
def test_moe_layer_gc_policy_env(monkeypatch, value, expected):
    monkeypatch.setenv("VLLM_EXL3_GC_AFTER_MOE_LAYER", value)
    assert exl3._exl3_gc_after_moe_layer_enabled() is expected
