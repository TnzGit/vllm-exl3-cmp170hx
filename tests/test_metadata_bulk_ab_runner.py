from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "r0_run_exl3_metadata_bulk_ab.sh"
EXL3 = ROOT / "src" / "vllm_exl3" / "exl3.py"


def test_metadata_bulk_runner_shell_syntax():
    r = subprocess.run(["bash", "-n", str(RUNNER)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_metadata_bulk_runner_is_single_boot_and_opt_in():
    src = RUNNER.read_text()
    assert src.count("r0_run_tensor_consumer_attribution.sh") == 1
    assert "VLLM_EXL3_METADATA_BULK_AB=1" in src
    assert "/v1/completions" not in src
    assert "drop_caches" not in src
    assert "metadata_bulk_ab_summary.json" in src


def test_metadata_bulk_is_even_control_odd_deferred_and_trellis_untouched():
    src = EXL3.read_text()
    assert '(int(expert_id) & 1)' in src
    assert '_metadata_bulk_ab_stage(' in src
    assert '_metadata_bulk_ab_commit(layer)' in src
    block = src[src.index("def _load_exl3("):src.index("def process_weights_after_loading", src.index("def _load_exl3("))]
    assert 'if suffix == "trellis":' in block
    assert 'VLLM_EXL3_METADATA_BULK_AB' not in block[block.index('if suffix == "trellis":'):block.index('# suh / svh remain stacked')]


def test_metadata_bulk_commit_stays_inside_layer_load_weights():
    src = EXL3.read_text()
    start = src.index("def _exl3_routed_experts_loader")
    end = src.index("def _moe_marker_or_none", start)
    block = src[start:end]
    assert block.index("_metadata_bulk_ab_commit(layer)") < block.index("return load_weights")
