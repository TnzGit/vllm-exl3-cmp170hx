import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "kvmem_q2c_scheduler_preflight.py"


def _load():
    spec = importlib.util.spec_from_file_location("q2c_preflight", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_q2c_real_mixed_geometry_survives_packed_grouping():
    mod = _load()
    out = mod._probe_packed_grouping(1568)
    assert out["ok"] is True
    assert out["resolved_layout"] == "BLHNC"
    assert out["qsa_block_sizes"] == [16]
    assert out["qsa_page_sizes"] == [32768]


def test_q2c_installed_vllm_preflight_goes_read_only():
    mod = _load()
    out = mod.build_result(1568)
    assert out["classification"] == "Q2C_SCHEDULER_SHRINK_PREFLIGHT_GO"
    assert out["go"] is True
    assert all(out["checks"].values())
    assert out["geometry_161k"]["logical_table_pages"] == 10063
    assert out["geometry_161k"]["physical_page_cap"] == 4160
    assert out["prior_q2b_observation"]["cross_granularity_ratio"] == 98
