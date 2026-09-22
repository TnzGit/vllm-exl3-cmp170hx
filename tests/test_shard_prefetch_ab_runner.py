from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
RUNNER=ROOT/"tools"/"r0_run_exl3_shard_prefetch_ab.sh"
WATCHER=ROOT/"tools"/"r0_proc_io_watch.py"

def test_runner_shell_syntax():
    r=subprocess.run(["bash","-n",str(RUNNER)],capture_output=True,text=True); assert r.returncode==0,r.stderr

def test_runner_is_one_boundary_boot_and_no_full_prefetch():
    src=RUNNER.read_text()
    assert src.count("r0_run_exl3_loader_attribution.sh")==1
    assert "VLLM_EXL3_SHARD_PREFETCH_AB=1" in src
    assert "--safetensors-load-strategy=prefetch" not in src
    assert "drop_caches" not in src
    assert "/v1/completions" not in src

def test_runner_requires_ideal_window_and_restores_weight_utils():
    src=RUNNER.read_text()
    assert '"model_start_to_main_weights_done"' in src
    assert "weight_utils_sha_before=" in src
    assert "weight_utils_sha_patched=" in src
    assert "weight_utils_sha_after=" in src
    assert "installed_weight_utils_restored=true" in src

def test_watcher_accepts_local_model_start_marker():
    src=WATCHER.read_text()
    assert "Loading model from scratch" in src
