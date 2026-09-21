from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "r0_run_coop_prelude_ab.sh"


def test_runner_shell_syntax_is_valid():
    run = subprocess.run(
        ["bash", "-n", str(RUNNER)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr


def test_runner_proves_pure_python_delta_before_installing():
    src = RUNNER.read_text()
    assert 'git -C "$REPO" diff --quiet "$BASE_SHA" HEAD -- csrc' in src
    assert "REFUSE: branch changes csrc" in src
    assert 'cmp -s "$BASE_EXPECTED" "$INSTALLED_PLUGIN"' in src
    assert 'cp "$REPO/src/vllm_exl3/exl3.py" "$INSTALLED_PLUGIN"' in src
    assert "restored_installed_plugin=1" in src


def test_runner_has_three_isolated_modes():
    src = RUNNER.read_text()
    assert "base) early=0; empty=0" in src
    assert "early) early=1; empty=0" in src
    assert "empty) early=1; empty=1" in src
    assert 'VLLM_EXL3_COOP_EARLY_PRELUDE="$early"' in src
    assert 'VLLM_EXL3_COOP_OUT_EMPTY="$empty"' in src


def test_runner_keeps_formal_perf_profiler_off_and_traces_all_modes():
    src = RUNNER.read_text()
    assert "run_cell base" in src
    assert "run_cell early" in src
    assert "run_cell empty" in src
    assert "capture_trace base" in src
    assert "capture_trace early" in src
    assert "capture_trace empty" in src
    assert "r0_coop_prelude_compare.py" in src
