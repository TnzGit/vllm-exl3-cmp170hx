from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "r0_run_kvmem_qsa_shadow.sh"


def test_k0_runner_shell_syntax():
    run = subprocess.run(
        ["bash", "-n", str(RUNNER)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr


def test_k0_runner_is_eager_no_draft_and_shadow_only():
    src = RUNNER.read_text()
    assert "ENFORCE_EAGER=1" in src
    assert "NUM_SPEC_TOKENS=0" in src
    assert "VLLM_QWEN_KVMEM_SHADOW_PATH" in src
    assert "VLLM_QWEN_KVMEM_SHADOW_MIN_POS" in src
    assert "--enforce-eager" in src
    assert "NOTE: eager shadow run is diagnostic only" in src


def test_k0_runner_checks_then_restores_installed_qsa():
    src = RUNNER.read_text()
    assert "--check-only" in src
    assert 'cmp -s "$QSA_BACKUP" "$QSA"' in src
    assert "patch_vllm_qsa_shadow.py" in src
    assert "restore_qsa" in src
    assert 'rm -f "$QSA.kvmem_qsa_shadow.orig"' in src
    assert 'QSA_SHA_AFTER=$(sha256sum "$QSA"' in src


def test_k0_runner_covers_both_contexts_and_all_case_outputs():
    src = RUNNER.read_text()
    assert "for ctx in 160000 240000" in src
    assert 'for casefile in "$OUT/cases/ctx' in src
    assert "kvmem_qsa_shadow_probe.py" in src
    assert "kvmem_qsa_shadow_summarize.py" in src
    assert "k0_suite_summary.json" in src


def test_k0_runner_never_changes_attention_or_kv_policy():
    src = RUNNER.read_text()
    for forbidden in (
        "enable-prefix-caching",
        "kv-offloading",
        "kv-connector",
        "EXL3_MOE_COOP_WIDE",
        "VLLM_EXL3_COOP_EARLY_PRELUDE",
    ):
        assert forbidden not in src
