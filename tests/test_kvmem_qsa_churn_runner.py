from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "r0_run_kvmem_qsa_churn.sh"


def test_k1a_runner_shell_syntax():
    run = subprocess.run(
        ["bash", "-n", str(RUNNER)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr


def test_k1a_runner_is_shadow_only_and_eager_no_draft():
    src = RUNNER.read_text()
    assert "ENFORCE_EAGER=1" in src
    assert "NUM_SPEC_TOKENS=0" in src
    assert "VLLM_QWEN_KVMEM_SHADOW_PATH" in src
    assert "--enforce-eager" in src
    for forbidden in (
        "kv-offloading",
        "kv-connector",
        "enable-prefix-caching",
        "VLLM_EXL3_COOP_EARLY_PRELUDE",
        "EXL3_MOE_COOP_WIDE",
    ):
        assert forbidden not in src


def test_k1a_runner_uses_fixed_history_turns_and_churn_analyzer():
    src = RUNNER.read_text()
    assert "kvmem_qsa_make_turns.py" in src
    assert "for ctx in 160000 240000" in src
    assert 'turn_*.json' in src
    assert "kvmem_qsa_churn_analyze.py" in src
    assert "k1a_churn_summary.json" in src
    assert '--model-config "$MODEL_DIR/config.json"' in src


def test_k1a_runner_protects_installed_qsa_source():
    src = RUNNER.read_text()
    assert "installed QSA source is already shadow-patched" in src
    assert "stale QSA shadow backup exists" in src
    assert "--check-only" in src
    assert 'cmp -s "$QSA_BACKUP" "$QSA"' in src
    assert "restore_qsa" in src
    assert 'QSA_SHA_AFTER=$(sha256sum "$QSA"' in src


def test_k1a_runner_runs_contract_test_in_cpu_gate():
    src = RUNNER.read_text()
    assert "test_kvmem_qsa_churn_runner.py" in src
