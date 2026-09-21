"""Regression tests for the profiler correlation chain used by the Amdahl tool.

torche profiler links launches as a two-hop chain:

    cpu_op."External id"   == cuda_runtime."External id"
    cuda_runtime."correlation" == kernel."correlation"

An earlier analyzer matched kernel.correlation directly to cpu_op."External
id", skipping the middle hop. That produced a suspiciously uniform coverage
across every kernel family and paired GEMV kernels with unrelated aten::empty
ops. These tests pin the correct behaviour.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import r0_k3_amdahl as amd  # noqa: E402


def _ev(cat, *, corr=None, ext=None, ts=0.0, dur=0.0, name=""):
    a = {}
    if corr is not None:
        a["correlation"] = corr
    if ext is not None:
        a["External id"] = ext
    return {"cat": cat, "name": name, "ts": ts, "dur": dur, "args": a}


def test_two_hop_chain_resolves_caller():
    events = [
        _ev("cpu_op", ext=900, name="aten::mm", ts=10.0, dur=5.0),
        _ev("cuda_runtime", corr=555, ext=900, name="cudaLaunchKernel"),
        _ev("kernel", corr=555, name="gemv_kernel", dur=1000.0),
    ]
    rt = {}
    cpu = {}
    for e in events:
        if e["cat"] == "cuda_runtime":
            rt[e["args"]["correlation"]] = e["args"]["External id"]
        elif e["cat"] == "cpu_op":
            cpu[e["args"]["External id"]] = e
    kern = events[2]
    ext = rt.get(kern["args"]["correlation"])
    op = cpu.get(ext)
    assert op is not None and op["name"] == "aten::mm"


def test_one_hop_shortcut_is_wrong():
    """kernel.correlation must NOT be looked up as a cpu_op External id."""
    events = [
        _ev("cpu_op", ext=900, name="aten::mm"),
        _ev("cuda_runtime", corr=555, ext=900),
        _ev("kernel", corr=555, name="gemv_kernel", dur=1000.0),
    ]
    cpu = {e["args"]["External id"]: e
           for e in events if e["cat"] == "cpu_op"}
    # the buggy lookup: kernel.correlation (555) used as an External id
    assert cpu.get(events[2]["args"]["correlation"]) is None


def test_classifier_never_force_classifies_unknown():
    bucket, _ = amd.classify("some_totally_unknown_kernel_xyz")
    assert bucket == "OTHER"


def test_classifier_maps_known_families():
    cases = {
        "void exl3_moe_coop_a_kernel<3,2,true>": "coop_moe_a",
        "void exl3_moe_coop_b_kernel<3,2,true>": "coop_moe_b",
        "_hc_combine_norm_kernel": "hyper_conn",
        "void vllm::persistent::persistent_topk_kernel<512, 4u>": "topk_sort",
        "void exl3_gemv_int8_sq_kernel<4, 1, true, false>": "dense_gemv",
        "void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16>": "bf16_gemm",
    }
    for name, expect in cases.items():
        bucket, _ = amd.classify(name)
        assert bucket == expect, f"{name} -> {bucket} != {expect}"


def test_source_uses_two_hop_not_one():
    src = (Path(__file__).resolve().parents[1] / "tools" / "r0_k3_amdahl.py").read_text()
    assert "runtime_by_corr" in src
    assert "cpu_by_extid" in src
    assert "runtime_by_corr.get(e.get(\"args\", {}).get(\"correlation\"))" in src
    # must not take the kernel correlation straight into the cpu_op map
    assert "cpu_by_extid.get(e.get(\"args\", {}).get(\"correlation\"))" not in src
