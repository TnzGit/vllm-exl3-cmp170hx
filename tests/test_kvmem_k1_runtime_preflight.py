import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "kvmem_k1_runtime_preflight.py"


def _load():
    spec = importlib.util.spec_from_file_location("k1_preflight", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def _entry(*attrs):
    return {
        "imported": True,
        "attrs": {name: {"present": True} for name in attrs},
    }


def test_choose_route_prefers_hisparse_contract():
    mod = _load()
    entries = {
        "hisparse_coordinator": _entry("HiSparseCoordinator"),
        "hisparse_connector": _entry("HiSparseConnector"),
        "hisparse_managers": _entry(
            "HiSparseResidentManager",
            "HiSparseHotManager",
            "HiSparseSourceManager",
        ),
        "hisparse_specs": _entry("HiSparseResidentSpec", "HiSparseHotSpec"),
        "offloading_connector": _entry("OffloadingConnector"),
        "cpu_offload_spec": _entry("CPUOffloadingSpec"),
        "cpu_offload_manager": _entry("CPUOffloadingManager"),
    }
    out = mod.choose_integration_route(entries, {"HiSparse": []})
    assert out["hisparse_core_available"] is True
    assert out["recommended_integration_route"] == "adapt_hisparse_contract"


def test_choose_route_falls_back_to_generic_offloading():
    mod = _load()
    entries = {
        "hisparse_coordinator": {"imported": False, "attrs": {}},
        "hisparse_connector": {"imported": False, "attrs": {}},
        "hisparse_managers": {"imported": False, "attrs": {}},
        "hisparse_specs": {"imported": False, "attrs": {}},
        "offloading_connector": _entry("OffloadingConnector"),
        "cpu_offload_spec": _entry("CPUOffloadingSpec"),
        "cpu_offload_manager": _entry("CPUOffloadingManager"),
    }
    out = mod.choose_integration_route(entries, {"HiSparse": []})
    assert out["hisparse_core_available"] is False
    assert out["generic_cpu_offloading_available"] is True
    assert (
        out["recommended_integration_route"]
        == "extend_generic_offloading_contract"
    )


def test_choose_route_fails_closed_without_upstream_interfaces():
    mod = _load()
    entries = {
        key: {"imported": False, "attrs": {}}
        for key in mod.CAPABILITIES
    }
    out = mod.choose_integration_route(entries, {"HiSparse": []})
    assert out["hisparse_core_available"] is False
    assert out["generic_cpu_offloading_available"] is False
    assert out["recommended_integration_route"] == "custom_vllm_patch_required"


def test_qwen_hisparse_hit_is_reported_but_not_treated_as_compatibility():
    mod = _load()
    entries = {
        key: {"imported": False, "attrs": {}}
        for key in mod.CAPABILITIES
    }
    out = mod.choose_integration_route(
        entries,
        {"HiSparse": [{"path": "qsa.py", "line": 1, "text": "HiSparse"}]},
    )
    assert out["qwen4_exp_mentions_hisparse"] is True
    assert "not compatibility" in out["warning"].lower()
