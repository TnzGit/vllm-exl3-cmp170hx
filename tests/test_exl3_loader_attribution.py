import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "r0_exl3_loader_attribution.py"


def _load():
    spec = importlib.util.spec_from_file_location("loader_attr", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def test_loader_summary_combines_kernel_io_and_copy_wall():
    mod = _load()
    startup = {
        "timings": {
            "main_weights_s": 100.0,
            "draft_weights_s": 20.0,
            "total_weights_s": 120.0,
            "model_construct_postload_s": 15.0,
        },
        "checkpoint": {"checkpoint_gib": 60.0},
    }
    first = {
        "tag": "FIRST_EXL3_COPY_BEFORE",
        "pid": 123,
        "monotonic_s": 10.0,
        "loader_timing": {},
        "direct_fill": {},
        "proc_io": {"read_bytes": 100},
        "ru_minflt": 10,
        "ru_majflt": 2,
        "ru_inblock": 3,
    }
    final = {
        "tag": "ALL_WEIGHTS_LOADED_BEFORE_POSTLOAD",
        "pid": 123,
        "monotonic_s": 105.0,
        "loader_timing": {
            "GENERIC_COPY_CALLS": 100,
            "GENERIC_COPY_BYTES": 10 * 1024**3,
            "GENERIC_COPY_WALL_S": 20.0,
            "DIRECT_TRELLIS_PREP_CALLS": 50,
            "DIRECT_TRELLIS_PREP_BYTES": 20 * 1024**3,
            "DIRECT_TRELLIS_PREP_WALL_S": 1.5,
            "DIRECT_TRELLIS_COPY_CALLS": 50,
            "DIRECT_TRELLIS_COPY_BYTES": 20 * 1024**3,
            "DIRECT_TRELLIS_COPY_WALL_S": 40.0,
        },
        "direct_fill": {
            "DIRECT_FILL_CALLS": 50,
            "DIRECT_FILL_BYTES": 20 * 1024**3,
            "DIRECT_FILL_FALLBACK_CALLS": 0,
            "DIRECT_FILL_FALLBACK_BYTES": 0,
        },
        "proc_io": {"read_bytes": 50 * 1024**3 + 100},
        "ru_minflt": 1000,
        "ru_majflt": 500,
        "ru_inblock": 600,
    }
    proc_watch = {
        "deltas": {
            "model_start_to_main_weights_done": {
                "elapsed_s": 100.0,
                "io": {
                    "read_bytes": 55 * 1024**3,
                    "rchar": 4 * 1024**3,
                    "syscr": 10000,
                },
                "stat": {
                    "majflt": 1234,
                    "minflt": 5678,
                    "utime_s": 20.0,
                    "stime_s": 10.0,
                },
            }
        }
    }
    out = mod.summarize(startup, [first, final], proc_watch, 6.0)
    assert out["loader_attribution_valid"] is True
    assert out["exl3_trace"]["instrumented_copy_gib"] == 30.0
    assert out["exl3_trace"]["instrumented_copy_wall_s"] == 60.0
    assert out["exl3_trace"]["raw_h2d_floor_for_instrumented_bytes_s"] == 5.0
    assert out["kernel_model_load_window"]["read_gib"] == 55.0
    assert out["kernel_model_load_window"]["major_faults"] == 1234
    assert out["interpretation_contract"]["direct_fill_no_fallback"] is True


def test_loader_summary_rejects_missing_boundaries():
    mod = _load()
    startup = {
        "timings": {"main_weights_s": 1.0},
        "checkpoint": {"checkpoint_gib": 1.0},
    }
    try:
        mod.summarize(startup, [], {"deltas": {}}, 6.0)
    except ValueError as exc:
        assert "missing EXL3 loader trace boundaries" in str(exc)
    else:
        raise AssertionError("expected missing-boundary failure")
