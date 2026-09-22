from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "r0_run_exl3_metadata_full_bulk_qual.sh"


def test_full_bulk_runner_shell_syntax():
    r = subprocess.run(["bash", "-n", str(RUNNER)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_full_bulk_runner_is_exact_and_production_shaped():
    src = RUNNER.read_text()
    assert 'EXPECTED_SHA="${EXPECTED_SHA:?set EXPECTED_SHA to the exact qualification head}"' in src
    assert 'VLLM_EXL3_METADATA_FULL_BULK="$full"' in src
    assert 'NUM_SPEC_TOKENS=3' in src
    assert '/health' in src
    assert 'run_case control_a 0' in src
    assert 'run_case full 1' in src
    assert 'run_case control_b 0' in src
    assert 'full_bulk_qualification.json' in src
    assert 'drop_caches' not in src
    assert 'STOP HERE' in src
