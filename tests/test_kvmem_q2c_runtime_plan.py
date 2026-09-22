import importlib.util
from pathlib import Path

from vllm_exl3.kvmem_qsa_scheduler_runtime import validate_runtime_plan


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "kvmem_qsa_make_q2c_runtime_plan.py"


def _load():
    spec = importlib.util.spec_from_file_location("q2c_plan", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _q2b():
    resident = list(range(4096))
    return {
        "schema": 1,
        "mode": "qsa_cpu_backed_shadow",
        "page_tokens": 16,
        "resident_pages": resident,
        "resident_page_count": 4096,
        "apply_min_pos": 160000,
        "active_from_pos": 160000,
        "active_page0": 10000,
        "active_reserve_tokens": 1024,
        "active_reserve_pages": 64,
        "physical_page_count": 4160,
        "publication_staging_pages": 128,
    }


def test_promote_q2b_plan_keeps_frozen_geometry():
    mod = _load()
    out = mod.promote(_q2b())
    assert out["mode"] == "qsa_scheduler_owned_transition"
    assert out["resident_pages"] == list(range(4096))
    assert out["resident_page_count"] == 4096
    assert out["active_page0"] == 10000
    assert out["active_reserve_pages"] == 64
    assert out["scheduler_chunk_tokens"] == 1024
    assert out["physical_page_count"] == 4160
    validate_runtime_plan(out)
