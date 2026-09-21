from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "r0_run_exl3_loader_attribution.sh"


def test_loader_runner_shell_syntax():
    run = subprocess.run(
        ["bash", "-n", str(RUNNER)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr


def test_loader_runner_stops_at_main_weight_boundary_not_health():
    src = RUNNER.read_text()
    assert "Loading weights took" in src
    assert "ALL_WEIGHTS_LOADED_BEFORE_POSTLOAD" in src
    assert "stop before draft/compile/warmup" in src
    assert "/v1/completions" not in src
    assert "wait_health" not in src


def test_loader_runner_forces_current_branch_exl3_without_install_patch():
    src = RUNNER.read_text()
    assert 'PYTHONPATH="$REPO/src' in src
    assert 'VLLM_EXL3_LOAD_TRACE_PATH="$TRACE"' in src
    assert "pip install" not in src
    assert "site-packages/vllm_exl3" not in src


def test_loader_runner_preserves_production_loader_knobs():
    src = RUNNER.read_text()
    assert 'LOADER_MAX_MODEL_LEN:-4096' in src
    assert 'NUM_SPEC_TOKENS:-3' in src
    assert 'GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"' in src
    assert "VLLM_EXL3_COOP=1" in src
    assert "MAX_NUM_SEQS=1" in src


def test_loader_runner_collects_kernel_io_copy_and_h2d_reference():
    src = RUNNER.read_text()
    assert "r0_proc_io_watch.py" in src
    assert "r0_exl3_loader_attribution.py" in src
    assert "LOADER_REFERENCE_H2D_GIB_S:-6.3494" in src
    assert "direct_trellis_copy_wall_s" in src
    assert "kernel_read_vs_checkpoint_ratio" in src
    assert "major_faults" in src


def test_loader_runner_never_drops_page_cache_or_changes_loader_policy():
    src = RUNNER.read_text()
    assert "drop_caches" not in src
    assert "sync; echo 3" not in src
    assert "VLLM_EXL3_MADV_AFTER_H2D=" not in src
    assert "VLLM_EXL3_GC_AFTER_MOE_LAYER=" not in src
    assert "pkill" not in src


def test_loader_runner_keeps_health_cleanup_evidence():
    src = RUNNER.read_text()
    assert "xid_before=" in src
    assert "xid_after=" in src
    assert "xid_delta=" in src
    assert "gpu_processes=" in src
    assert "vllm_processes=" in src
    assert "port_$PORT=" in src
    assert "installed_exl3_sha_before=" in src
    assert "installed_exl3_sha_after=" in src
    assert "installed_exl3_unchanged=true" in src
