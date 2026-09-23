import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARIZER = ROOT / "tools" / "kvmem_q2e_benchmark_summarize.py"


def _load_summarizer():
    spec = importlib.util.spec_from_file_location("q2e_benchmark_summary", SUMMARIZER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_summary_requires_semantics_layers_cpu_exactness_and_capacity(tmp_path):
    cell_dir = tmp_path / "ctx16000"
    cell_dir.mkdir()
    (cell_dir / "cell.json").write_text(json.dumps({
        "context_limit": 16000,
        "prompt_tokens": 15533,
        "completion_tokens": 256,
        "prefill_s": 8.0,
        "prefill_tok_s": 1941.625,
        "decode_tokens": 255,
        "decode_s": 2.55,
        "decode_tok_s": 100.0,
        "decode_ms_per_token": 10.0,
        "wall_s": 10.55,
        "target_codes_in_order": True,
        "count_exact": True,
    }))
    runtime = [
        {
            "event": "q2d_streaming_runtime",
            "layer_id": layer,
            "combined_real_pages": 4160,
            "peak_read_pages": 4032,
            "max_working_pages": 4000,
            "cpu_roundtrip_exact": True,
        }
        for layer in range(12)
    ]
    (cell_dir / "worker.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in runtime)
    )
    (cell_dir / "scheduler.jsonl").write_text(json.dumps({
        "event": "q2d_scheduler_assign", "peak_real_write_pages": 128
    }) + "\n")
    result = _load_summarizer().summarize(tmp_path, [16000])
    assert result["status"] == "VALID"
    assert result["cells"][0]["combined_real_page_peak"] == 4160

    runtime[0]["combined_real_pages"] = 4161
    (cell_dir / "worker.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in runtime)
    )
    result = _load_summarizer().summarize(tmp_path, [16000])
    assert result["status"] == "INVALID"
    assert "physical page cap exceeded" in result["errors"][0]


def test_layer_id_can_be_derived_from_live_layer_name():
    derive = _load_summarizer()._layer_id
    assert derive({"layer": "model.layers.17.self_attn"}) == 17
    assert derive({"layer_id": 9}) == 9
