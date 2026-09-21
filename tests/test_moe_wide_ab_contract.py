from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "r0_run_moe_wide_ab.sh"


def test_moe_wide_runner_attests_installed_plugin_before_gpu():
    src = RUNNER.read_text()
    attest = src.index("source attestation")
    first_engine = src.index("start_engine auto")
    assert attest < first_engine
    assert 'cmp -s "$REPO/src/vllm_exl3/exl3.py" "$INSTALLED_PLUGIN"' in src
    assert "REFUSE: installed vllm_exl3/exl3.py does not match this branch" in src


def test_moe_wide_runner_compares_auto_and_forced_narrow():
    src = RUNNER.read_text()
    assert "env_cmd+=(-u EXL3_MOE_COOP_WIDE)" in src
    assert "env_cmd+=(EXL3_MOE_COOP_WIDE=0)" in src
    assert "capture_trace auto" in src
    assert "capture_trace narrow" in src
    assert "r0_moe_wide_compare.py" in src


def test_moe_wide_runner_never_patches_or_rebuilds_exllamav3():
    src = RUNNER.read_text()
    for forbidden in ("pip install", "setup.py build", "ninja -C", "patch_exllamav3"):
        assert forbidden not in src
