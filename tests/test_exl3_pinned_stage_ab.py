import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "r0_exl3_pinned_stage_ab.py"


def _load():
    spec = importlib.util.spec_from_file_location("pinned_ab", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pinned_ab_summary_balances_bytes_and_projects_full_paths():
    mod = _load()
    gib = 1024**3
    startup = {"timings": {"main_weights_s": 80.0}}
    first = {
        "tag": "FIRST_EXL3_COPY_BEFORE",
        "pinned_stage_ab": True,
        "loader_timing": {},
    }
    final = {
        "tag": "ALL_WEIGHTS_LOADED_BEFORE_POSTLOAD",
        "pinned_stage_ab": True,
        "loader_timing": {
            "DIRECT_TRELLIS_COPY_BYTES": 20 * gib,
            "DIRECT_TRELLIS_COPY_WALL_S": 30.0,
            "DIRECT_TRELLIS_CONTROL_CALLS": 100,
            "DIRECT_TRELLIS_CONTROL_BYTES": 10 * gib,
            "DIRECT_TRELLIS_CONTROL_WALL_S": 20.0,
            "DIRECT_TRELLIS_PINNED_CALLS": 100,
            "DIRECT_TRELLIS_PINNED_BYTES": 10 * gib,
            "DIRECT_TRELLIS_PINNED_ALLOC_WALL_S": 0.25,
            "DIRECT_TRELLIS_PINNED_CPU_STAGE_WALL_S": 8.0,
            "DIRECT_TRELLIS_PINNED_H2D_WALL_S": 2.0,
            "DIRECT_TRELLIS_PINNED_TOTAL_WALL_S": 10.0,
            "DIRECT_TRELLIS_PINNED_BUFFER_MAX_BYTES": 8 * 2**20,
        },
    }
    out = mod.summarize(startup, [first, final], 5.0)
    assert out["pinned_stage_ab_valid"] is True
    assert out["balance"]["pinned_byte_share"] == 0.5
    assert out["control"]["effective_gib_s"] == 0.5
    assert out["pinned"]["cpu_stage_gib_s"] == 1.25
    assert out["pinned"]["h2d_gib_s"] == 5.0
    assert out["pinned"]["end_to_end_gib_s_ex_alloc"] == 1.0
    assert out["comparison"]["pinned_vs_control_speedup"] == 2.0
    assert out["comparison"]["projected_all_control_direct_wall_s"] == 40.0
    assert out["comparison"]["projected_all_pinned_direct_wall_s"] == 20.25
    assert out["comparison"]["outside_direct_mixed_s"] == 49.75
    assert out["comparison"]["projected_all_pinned_main_weights_s"] == 70.0
    assert out["interpretation_contract"]["projection_is_not_qualification"] is True


def test_pinned_ab_summary_rejects_unbalanced_partition():
    mod = _load()
    gib = 1024**3
    startup = {"timings": {"main_weights_s": 10.0}}
    first = {
        "tag": "FIRST_EXL3_COPY_BEFORE",
        "pinned_stage_ab": True,
        "loader_timing": {},
    }
    final = {
        "tag": "ALL_WEIGHTS_LOADED_BEFORE_POSTLOAD",
        "pinned_stage_ab": True,
        "loader_timing": {
            "DIRECT_TRELLIS_COPY_BYTES": 10 * gib,
            "DIRECT_TRELLIS_CONTROL_CALLS": 1,
            "DIRECT_TRELLIS_CONTROL_BYTES": 9 * gib,
            "DIRECT_TRELLIS_CONTROL_WALL_S": 9.0,
            "DIRECT_TRELLIS_PINNED_CALLS": 1,
            "DIRECT_TRELLIS_PINNED_BYTES": 1 * gib,
            "DIRECT_TRELLIS_PINNED_TOTAL_WALL_S": 1.0,
        },
    }
    out = mod.summarize(startup, [first, final], 6.0)
    assert out["pinned_stage_ab_valid"] is False
    assert out["balance"]["byte_balance_ok"] is False
