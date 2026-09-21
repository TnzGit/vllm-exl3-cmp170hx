import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "kvmem_qsa_make_physical_plan.py"


def _load():
    spec = importlib.util.spec_from_file_location("q2_plan", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def _fixtures():
    resident = list(range(256))
    summary = {
        "contexts": [{
            "context": 160000,
            "kv_geometry": {"qsa_layers_observed": 12},
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
        "query_span": [159488, 159533],
        "target_facts": [{"marker": "D", "code": "violet-harbor-31"}],
    }
    return summary, turn


def test_q2_plan_expands_64k_history_and_active_reserve():
    mod = _load()
    summary, turn = _fixtures()
    out = mod.build_plan(
        summary,
        turn,
        context=160000,
        turn_name="ask_d_e",
        page_tokens=16,
        active_reserve_tokens=1024,
    )
    assert out["resident_region_count"] == 256
    assert out["resident_page_count"] == 4096
    assert out["active_reserve_pages"] == 64
    assert out["physical_page_count"] == 4160
    assert out["active_page0"] == 159488 // 16
    assert out["expected_qsa_layers"] == 12


def test_q2_plan_requires_page_aligned_query_boundary():
    mod = _load()
    summary, turn = _fixtures()
    turn["query_span"] = [159489, 159533]
    try:
        mod.build_plan(
            summary,
            turn,
            context=160000,
            turn_name="ask_d_e",
            page_tokens=16,
            active_reserve_tokens=1024,
        )
    except ValueError as exc:
        assert "aligned" in str(exc)
    else:
        raise AssertionError("expected ValueError")
