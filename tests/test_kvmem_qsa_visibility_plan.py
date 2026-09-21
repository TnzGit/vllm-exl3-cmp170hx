import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "kvmem_qsa_make_visibility_plan.py"


def _load():
    spec = importlib.util.spec_from_file_location("q1_plan", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def test_build_plan_uses_frozen_k1b_stateful_resident_set():
    mod = _load()
    resident = list(range(256))
    summary = {
        "contexts": [{
            "context": 160000,
            "policies": {
                "b256_budget65536": {
                    "replace_5": {
                        "turns": [{
                            "name": "ask_d_e",
                            "resident_blocks": resident,
                        }]
                    }
                }
            },
        }]
    }
    turn = {
        "name": "ask_d_e",
        "query_span": [159488, 159550],
        "target_facts": [
            {"marker": "D", "code": "violet-harbor-31"},
            {"marker": "E", "code": "granite-comet-72"},
        ],
    }

    out = mod.build_plan(
        summary,
        turn,
        context=160000,
        turn_name="ask_d_e",
        page_tokens=16,
    )
    assert out["resident_regions"] == resident
    assert out["resident_region_count"] == 256
    assert out["resident_page_count"] == 4096
    assert out["pages_per_region"] == 16
    assert out["apply_min_pos"] == 159488
    assert out["active_from_pos"] == 159488
    assert out["target_facts"] == turn["target_facts"]


def test_build_plan_rejects_nonintegral_page_geometry():
    mod = _load()
    summary = {
        "contexts": [{
            "context": 160000,
            "policies": {
                "b256_budget65536": {
                    "replace_5": {
                        "turns": [{
                            "name": "ask_d_e",
                            "resident_blocks": list(range(256)),
                        }]
                    }
                }
            },
        }]
    }
    turn = {
        "name": "ask_d_e",
        "query_span": [10, 20],
        "target_facts": [],
    }
    try:
        mod.build_plan(
            summary,
            turn,
            context=160000,
            turn_name="ask_d_e",
            page_tokens=48,
        )
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("expected ValueError")
