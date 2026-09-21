from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "r0_run_kvmem_k1t_transfer.sh"


def test_k1t_runner_shell_syntax():
    run = subprocess.run(
        ["bash", "-n", str(RUNNER)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr


def test_k1t_runner_uses_real_generic_offload_probe_without_model_engine():
    src = RUNNER.read_text()
    assert "kvmem_k1t_transfer_probe.py" in src
    assert "kvmem_k1t_summarize.py" in src
    assert "CPUOffloadingWorker" in src
    assert "CPUOffloadingManager" in src
    for forbidden in (
        "serve_cmp170hx_qwen_firstboot",
        "patch_vllm_qsa_shadow.py",
        "apply_qwen4_exp_patches.py",
    ):
        assert forbidden not in src


def test_k1t_runner_uses_installed_page_size_and_measured_primary_geometry():
    src = RUNNER.read_text()
    assert "CacheConfig.DEFAULT_BLOCK_SIZE" in src
    assert "--logical-tokens 240000" in src
    assert "--capacity-tokens 65536" in src
    assert "--region-tokens 256" in src
    assert '--page-tokens "$PAGE_TOKENS"' in src
    assert "--replacement-fraction 0.05" in src
    assert "--qsa-layers 12" in src
    assert "--num-kv-heads 2" in src
    assert "--head-dim 256" in src
    assert "--dtype-bytes 2" in src


def test_k1t_runner_preserves_correctness_as_hard_gate():
    src = RUNNER.read_text()
    assert "transfer_correctness_go" in src
    assert "all_repeats_byte_exact" in src
    assert "qsa_stage_in_mapping_exact" in src
    assert "qsa_evicted_pages_negative" in src
    assert "xid_delta" in src


def test_k1t_runner_keeps_performance_classification_diagnostic():
    src = RUNNER.read_text()
    assert "--reference-gib-s 6.3494" in src
    assert "performance_class" in src
    assert "wall_over_raw_floor" in src
