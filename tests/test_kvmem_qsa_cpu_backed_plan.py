import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "kvmem_qsa_make_cpu_backed_plan.py"


def _load():
    spec = importlib.util.spec_from_file_location("q2b_plan", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def _fixtures():
    summary = {
        "contexts": [{
            "context": 160000,
            "kv_geometry": {"qsa_layers_observed": 12},
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
        "query_span": [159488, 159533],
        "target_facts": [{"marker": "D", "code": "violet-harbor-31"}],
    }
    return summary, turn


def test_q2b_plan_keeps_small_reusable_publication_window():
    mod = _load()
    summary, turn = _fixtures()
    out = mod.build_plan(
        summary,
        turn,
        context=160000,
        turn_name="ask_d_e",
        page_tokens=16,
        active_reserve_tokens=1024,
        publication_staging_pages=128,
    )
    assert out["mode"] == "qsa_cpu_backed_shadow"
    assert out["resident_page_count"] == 4096
    assert out["physical_page_count"] == 4160
    assert out["publication_staging_pages"] == 128
    assert out["transfer_tensor_page_count"] == 4288
    assert out["expected_qsa_layers"] == 12


def test_q2b_plan_caps_staging_window_at_resident_page_count():
    mod = _load()
    summary, turn = _fixtures()
    out = mod.build_plan(
        summary,
        turn,
        context=160000,
        turn_name="ask_d_e",
        page_tokens=16,
        active_reserve_tokens=1024,
        publication_staging_pages=99999,
    )
    assert out["publication_staging_pages"] == 4096
    assert out["transfer_tensor_page_count"] == 8256
