import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "r0_metadata_bulk_ab_summary.py"


def _load():
    spec = importlib.util.spec_from_file_location("metadata_bulk", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_metadata_bulk_summary_valid_balanced_positive():
    m = _load()
    trace = [{
        "tag": "ALL_WEIGHTS_LOADED_BEFORE_POSTLOAD",
        "metadata_bulk_ab": {
            "enabled": True,
            "control_calls": 30,
            "control_bytes": 3000,
            "control_wall_s": 3.0,
            "deferred_calls": 30,
            "deferred_bytes": 3000,
            "deferred_stage_wall_s": 0.6,
            "commit_calls": 6,
            "commit_bytes": 3000,
            "commit_wall_s": 0.9,
            "committed_layers": 2,
            "control_by_suffix": {
                "mul1": {"calls":10,"bytes":40,"wall_s":1.0},
                "suh": {"calls":10,"bytes":1480,"wall_s":1.0},
                "svh": {"calls":10,"bytes":1480,"wall_s":1.0},
            },
            "deferred_by_suffix": {
                "mul1": {"calls":10,"bytes":40,"wall_s":0.2},
                "suh": {"calls":10,"bytes":1480,"wall_s":0.2},
                "svh": {"calls":10,"bytes":1480,"wall_s":0.2},
            },
            "commit_by_suffix": {
                "mul1": {"calls":2,"bytes":40,"wall_s":0.1},
                "suh": {"calls":2,"bytes":1480,"wall_s":0.4},
                "svh": {"calls":2,"bytes":1480,"wall_s":0.4},
            },
        },
    }]
    tensor = {
        "tensor_consumer_attribution_valid": True,
        "main_weights_s": 10.0,
        "totals": {},
        "by_suffix": {},
        "by_scope": {},
        "by_size_bin": {},
        "exl3_copy_reconciliation": {},
    }
    loader = {
        "loader_attribution_valid": True,
        "startup": {"main_weights_s": 10.0},
        "kernel_model_load_window": {},
    }
    out = m.summarize(trace, tensor, loader)
    assert out["metadata_bulk_ab_valid"] is True
    assert out["classification"] == "METADATA_BULK_AB_POSITIVE"
    assert out["comparison"]["deferred_vs_control_speedup"] == 2.0
    assert out["comparison"]["gpu_call_reduction"] == 5.0
    assert out["by_suffix"]["mul1"]["commit_bytes_exact"] is True


def test_metadata_bulk_summary_rejects_unbalanced_arms():
    m = _load()
    trace = [{
        "tag": "ALL_WEIGHTS_LOADED_BEFORE_POSTLOAD",
        "metadata_bulk_ab": {
            "enabled": True,
            "control_calls": 10,
            "control_bytes": 100,
            "control_wall_s": 1.0,
            "deferred_calls": 9,
            "deferred_bytes": 90,
            "deferred_stage_wall_s": 0.2,
            "commit_calls": 1,
            "commit_bytes": 90,
            "commit_wall_s": 0.1,
            "committed_layers": 1,
            "control_by_suffix": {
                "mul1": {"calls":10,"bytes":100,"wall_s":1.0}
            },
            "deferred_by_suffix": {
                "mul1": {"calls":9,"bytes":90,"wall_s":0.2}
            },
            "commit_by_suffix": {
                "mul1": {"calls":1,"bytes":90,"wall_s":0.1}
            },
        },
    }]
    tensor = {"tensor_consumer_attribution_valid": True}
    loader = {
        "loader_attribution_valid": True,
        "startup": {"main_weights_s": 10.0},
    }
    out = m.summarize(trace, tensor, loader)
    assert out["metadata_bulk_ab_valid"] is False
    assert out["classification"] == "METADATA_BULK_AB_INVALID"
