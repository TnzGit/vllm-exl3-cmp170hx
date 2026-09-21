from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "r0_run_kvmem_k1q1_visibility.sh"


def test_k1q1_runner_shell_syntax():
    run = subprocess.run(
        ["bash", "-n", str(RUNNER)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr


def test_k1q1_runner_is_target_only_and_no_mtp():
    src = RUNNER.read_text()
    assert "NUM_SPEC_TOKENS=0" in src
    assert "MAX_NUM_SEQS=1" in src
    assert "ENFORCE_EAGER=1" in src
    assert "VLLM_EXL3_COOP=1" in src
    assert "ask_d_e" in src
    assert "ctx160000" in src


def test_k1q1_runner_has_baseline_and_masked_fresh_engines():
    src = RUNNER.read_text()
    assert "start_engine baseline" in src
    assert "start_engine masked" in src
    assert "env -u VLLM_QWEN_KVMEM_RESIDENT_PLAN" in src
    assert 'VLLM_QWEN_KVMEM_RESIDENT_PLAN="$PLAN"' in src
    assert 'VLLM_QWEN_KVMEM_VISIBILITY_STATS_PATH="$STATS"' in src


def test_k1q1_runner_never_uses_shadow_patch_or_changes_qsa_math():
    src = RUNNER.read_text()
    assert "patch_vllm_qsa_visibility.py" in src
    assert "patch_vllm_qsa_shadow.py" not in src
    assert "persistent_topk" not in src
    assert "tensor.copy_" not in src


def test_k1q1_runner_restores_qsa_byte_identically():
    src = RUNNER.read_text()
    assert "QSA_SHA_BEFORE" in src
    assert "QSA_SHA_AFTER" in src
    assert "installed QSA was not restored byte-identically" in src
    assert "kvmem_qsa_visibility.orig" in src


def test_k1q1_runner_requires_mask_to_be_exercised():
    src = RUNNER.read_text()
    assert '[[ ! -s "$STATS" ]]' in src
    assert "resident mask was not exercised" in src
    assert "historical_selected_dropped" in src
    assert "mask_exercised" in src


def test_k1q1_runner_does_not_kill_unrelated_vllm_processes():
    src = RUNNER.read_text()
    assert "pkill" not in src
    assert 'kill -TERM -- "-$LAUNCH_PID"' in src
