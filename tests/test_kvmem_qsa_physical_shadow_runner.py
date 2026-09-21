from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "r0_run_kvmem_k1q2a_physical_shadow.sh"


def test_q2a_runner_shell_syntax():
    run = subprocess.run(
        ["bash", "-n", str(RUNNER)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr


def test_q2a_runner_is_one_eager_target_only_engine():
    src = RUNNER.read_text()
    assert "ENFORCE_EAGER=1" in src
    assert "NUM_SPEC_TOKENS=0" in src
    assert "MAX_NUM_SEQS=1" in src
    assert "VLLM_EXL3_COOP=1" in src
    assert "start one physical-shadow engine" in src
    assert "baseline" not in src.lower()


def test_q2a_runner_uses_frozen_policy_and_exact_case():
    src = RUNNER.read_text()
    assert "ctx160000/turn_04_ask_d_e.json" in src
    assert "--context 160000" in src
    assert "--turn ask_d_e" in src
    assert 'K1Q2A_MAX_TOKENS:-512' in src
    assert 'K1Q2A_ACTIVE_RESERVE_TOKENS:-1024' in src


def test_q2a_runner_requires_physical_evidence_and_restore():
    src = RUNNER.read_text()
    assert "k1q2a_physical_summary.json" in src
    assert "attention_exact_all_records" in src
    assert "attention_max_abs" in src
    assert "bootstrap_ok" in src
    assert "QSA restore mismatch" in src
    assert "xid_delta" in src


def test_q2a_runner_does_not_patch_scheduler_or_enable_cpu_backing():
    src = RUNNER.read_text()
    assert "patch_vllm_qsa_physical_shadow.py" in src
    assert "kvmem_qsa_make_physical_plan.py" in src
    assert "kvmem_qsa_physical_shadow_summarize.py" in src
    assert "CPUOffloadingWorker" not in src
    assert "CPUOffloadingManager" not in src
    assert "patch_vllm_qsa_visibility.py" not in src
    assert "pkill" not in src



def test_q2a_runner_treats_default_block_size_as_resident_page_only():
    src = RUNNER.read_text()
    assert "resident_page_tokens=$PAGE_TOKENS" in src
    assert "independent of the hybrid scheduler full-cache block size" in src
    assert "kvmem-k1q2a-physical-shadow-crosspage" in src
