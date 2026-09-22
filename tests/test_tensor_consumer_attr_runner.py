from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "r0_run_tensor_consumer_attribution.sh"
WATCHER = ROOT / "tools" / "r0_proc_io_watch.py"


def test_tensor_attr_runner_shell_syntax():
    r = subprocess.run(["bash", "-n", str(RUNNER)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_tensor_attr_runner_is_one_loader_boundary_boot():
    src = RUNNER.read_text()
    assert src.count("r0_run_exl3_loader_attribution.sh") == 1
    assert "VLLM_EXL3_TENSOR_ATTR_PATH" in src
    assert "/v1/completions" not in src
    assert "drop_caches" not in src
    assert "_prefetch_checkpoint" not in src


def test_tensor_attr_runner_requires_ideal_window_and_restores_patch():
    src = RUNNER.read_text()
    assert "\"model_start_to_main_weights_done\"" in src
    assert "weight_utils_sha_before=" in src
    assert "weight_utils_sha_patched=" in src
    assert "weight_utils_sha_after=" in src
    assert "installed_weight_utils_restored=true" in src
    assert "SUMMARY_RC=${PIPESTATUS[0]}" in src
    assert src.index('echo "=== restore installed vLLM ==="') < src.index("if (( SUMMARY_RC != 0 ))")


def test_tensor_attr_runner_prints_category_and_copy_reconciliation():
    src = RUNNER.read_text()
    assert "by_suffix consumer wall descending" in src
    assert "by_scope consumer wall descending" in src
    assert "by_size_bin consumer wall descending" in src
    assert "copy reconciliation" in src
    assert "top 20 consumers" in src


def test_tensor_attr_watcher_accepts_local_model_start_marker():
    assert "Loading model from scratch" in WATCHER.read_text()
