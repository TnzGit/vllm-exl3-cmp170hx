from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "r0_run_coop_prelude_qualify.sh"


def test_qualify_runner_shell_syntax():
    run = subprocess.run(
        ["bash", "-n", str(RUNNER)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr


def test_qualify_runner_is_early_only():
    src = RUNNER.read_text()
    assert 'VLLM_EXL3_COOP_EARLY_PRELUDE="$early"' in src
    assert "-u VLLM_EXL3_COOP_OUT_EMPTY" in src
    assert "EARLY qualification cells" in src
    assert "EARLY+EMPTY" not in src


def test_qualify_runner_covers_long_context_and_parity():
    src = RUNNER.read_text()
    assert "for ctx in 4096 160000 240000" in src
    assert "repeats=7" in src
    assert "repeats=3" in src
    assert '--parity-reference "$OUT/ref_${ctx}.json"' in src
    assert "r0_k3_parity_recheck.py" in src


def test_qualify_runner_protects_installed_plugin_and_extension():
    src = RUNNER.read_text()
    assert 'git -C "$REPO" diff --quiet "$BASE_SHA" HEAD -- csrc' in src
    assert 'cmp -s "$BASE_EXPECTED" "$INSTALLED_PLUGIN"' in src
    assert 'cp "$REPO/src/vllm_exl3/exl3.py" "$INSTALLED_PLUGIN"' in src
    assert "SO_SHA_BEFORE" in src
    assert "SO_SHA_AFTER" in src
    assert "restored_installed_plugin=1" in src


def test_qualify_runner_derives_ld_library_path():
    src = RUNNER.read_text()
    assert 'LIB_DIRS=("$SP/torch/lib")' in src
    assert 'for d in "$SP"/nvidia/*/lib' in src
    assert "export LD_LIBRARY_PATH" in src
