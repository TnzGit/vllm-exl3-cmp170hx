import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLIENT = ROOT / "tools" / "kvmem_q2e_benchmark_client.py"


def _load_client():
    spec = importlib.util.spec_from_file_location("q2e_benchmark_client", CLIENT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_prefill_and_decode_denominators_exclude_first_output_token():
    metrics = _load_client()._metrics(
        prompt_tokens=16000,
        completion_tokens=256,
        ttft_s=8.0,
        wall_s=10.55,
    )
    assert metrics["prefill_tok_s"] == 2000.0
    assert metrics["decode_tokens"] == 255
    assert round(metrics["decode_s"], 6) == 2.55
    assert round(metrics["decode_tok_s"], 6) == 100.0
    assert round(metrics["decode_ms_per_token"], 6) == 10.0


def test_target_codes_must_appear_in_order():
    codes_in_order = _load_client()._codes_in_order
    assert codes_in_order("x violet-harbor-31 y granite-comet-72", [
        "violet-harbor-31", "granite-comet-72"
    ])
    assert not codes_in_order("granite-comet-72 violet-harbor-31", [
        "violet-harbor-31", "granite-comet-72"
    ])
