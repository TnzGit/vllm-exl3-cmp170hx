from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "r0_run_kvmem_k1q2b_cpu_backed.sh"


def test_q2b_runner_shell_syntax():
    run = subprocess.run(
        ["bash", "-n", str(RUNNER)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr


def test_q2b_runner_is_one_eager_target_only_engine():
    src = RUNNER.read_text()
    assert "ENFORCE_EAGER=1" in src
    assert "NUM_SPEC_TOKENS=0" in src
    assert "MAX_NUM_SEQS=1" in src
    assert "VLLM_EXL3_COOP=1" in src
    assert "start one CPU-backed engine" in src
    assert "baseline" not in src.lower()


def test_q2b_runner_uses_real_generic_cpu_adapter_from_current_repo():
    src = RUNNER.read_text()
    assert 'PYTHONPATH="$REPO/src' in src
    assert "single_tensor_cpu_backing=OK" in src
    assert "test_kvmem_vllm_offload_adapter.py" in src
    assert "patch_vllm_qsa_cpu_backed.py" in src
    assert "CPUOffloadingWorker(" not in src
    assert "CPUOffloadingManager(" not in src


def test_q2b_runner_pins_frozen_policy_and_small_staging_window():
    src = RUNNER.read_text()
    assert "ctx160000/turn_04_ask_d_e.json" in src
    assert "--context 160000" in src
    assert "--turn ask_d_e" in src
    assert "--publication-staging-pages 128" in src
    assert 'K1Q2B_MAX_TOKENS:-512' in src
    assert 'K1Q2B_ACTIVE_RESERVE_TOKENS:-1024' in src


def test_q2b_runner_sets_cpu_backing_env_in_engine():
    src = RUNNER.read_text()
    assert 'VLLM_QWEN_KVMEM_CPU_PLAN="$PLAN"' in src
    assert 'VLLM_QWEN_KVMEM_CPU_STATS_PATH="$STATS"' in src
    assert "VLLM_QWEN_KVMEM_CPU_CONTINUE_INPUT_EXACT_NONEXACT=1" in src


def test_q2b_runner_requires_transfer_mapping_semantic_and_restore_evidence():
    src = RUNNER.read_text()
    assert "transfer_gate" in src
    assert "physical_mapping_go" in src
    assert "semantic_go" in src
    assert "cpu_backing_all_present_by_layer" in src
    assert "bootstrap_all_pages_exact_by_layer" in src
    assert "expected_publish_jobs_per_layer" in src
    assert "transfer_tensor_geometry_ok" in src
    assert "input_mapping_exact_all_records" in src
    assert "QSA restore mismatch" in src
    assert "xid_delta" in src


def test_q2b_runner_refuses_stacked_research_patches_and_does_not_pkill():
    src = RUNNER.read_text()
    assert "# KVMEM_QSA_CPU_BACKED_V1" in src
    assert "# KVMEM_QSA_PHYSICAL_SHADOW_V1" in src
    assert "# KVMEM_QSA_RESIDENT_VISIBILITY_V1" in src
    assert "# KVMEM_QSA_SHADOW_V1" in src
    assert "pkill" not in src
