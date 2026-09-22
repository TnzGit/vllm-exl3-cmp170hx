import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "r0_tensor_consumer_attribution.py"


def _load():
    spec = importlib.util.spec_from_file_location("tensor_attr", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cat(count, nbytes, getw, getc, consw, consc):
    return {
        "count": count,
        "bytes": nbytes,
        "get_tensor_wall_s": getw,
        "get_tensor_cpu_s": getc,
        "consumer_wall_s": consw,
        "consumer_cpu_s": consc,
    }


def test_summary_reconciles_tensor_consumer_with_exl3_copy_wall():
    m = _load()
    gib = 1024**3
    payload = {
        "mode": "lazy_safetensors_tensor_consumer",
        "rows": [
            {
                "file": "model-00001.safetensors",
                "file_bytes": 10 * gib,
                "tensor_count": 100,
                "tensor_bytes": 8 * gib,
                "get_tensor_wall_s": 1.0,
                "get_tensor_cpu_s": 0.5,
                "consumer_wall_s": 20.0,
                "consumer_cpu_s": 30.0,
                "iterator_other_wall_s": 1.0,
                "shard_wall_s": 22.0,
                "by_suffix": {
                    "trellis": _cat(60, 6*gib, 0.4, 0.2, 14.0, 20.0),
                    "suh": _cat(20, 1*gib, 0.2, 0.1, 2.0, 3.0),
                    "weight": _cat(20, 1*gib, 0.4, 0.2, 4.0, 7.0),
                },
                "by_scope": {"routed_expert": _cat(80, 7*gib, 0.6, 0.3, 16.0, 23.0)},
                "by_size_bin": {"64k_1m": _cat(100, 8*gib, 1.0, 0.5, 20.0, 30.0)},
                "by_scope_suffix": {"routed_expert|trellis": _cat(60, 6*gib, 0.4, 0.2, 14.0, 20.0)},
                "top_consumers": [{"name":"x.trellis","bytes":gib,"consumer_wall_s":2.0,"get_tensor_wall_s":0.01}],
            },
        ],
    }
    loader = {
        "startup": {"main_weights_s": 25.0},
        "exl3_trace": {
            "direct_trellis_copy_wall_s": 10.0,
            "direct_trellis_prep_wall_s": 0.1,
            "generic_copy_wall_s": 1.5,
        },
    }
    out = m.summarize(payload, loader)
    assert out["tensor_consumer_attribution_valid"] is True
    assert out["totals"]["consumer_fraction_of_main_weights"] == 0.8
    assert out["by_suffix"]["trellis"]["consumer_wall_s"] == 14.0
    rec = out["exl3_copy_reconciliation"]
    assert rec["routed_trellis_consumer_minus_direct_copy_s"] == 4.0
    assert rec["generic_candidate_consumer_wall_s"] == 2.0
    assert rec["generic_candidate_minus_generic_copy_s"] == 0.5
    assert rec["consumer_wall_outside_instrumented_exl3_copy_s"] == 8.5
    assert out["top_consumers"][0]["name"] == "x.trellis"


def test_summary_merges_categories_across_shards():
    m = _load()
    row = {
        "file_bytes": 100, "tensor_count": 1, "tensor_bytes": 50,
        "get_tensor_wall_s": 0.1, "get_tensor_cpu_s": 0.1,
        "consumer_wall_s": 1.0, "consumer_cpu_s": 1.5,
        "iterator_other_wall_s": 0.1, "shard_wall_s": 1.2,
        "by_suffix": {"trellis": _cat(1,50,0.1,0.1,1.0,1.5)},
        "by_scope": {}, "by_size_bin": {}, "by_scope_suffix": {},
        "top_consumers": [],
    }
    payload={"mode":"lazy_safetensors_tensor_consumer","rows":[{**row,"file":"a"},{**row,"file":"b"}]}
    loader={"startup":{"main_weights_s":3.0},"exl3_trace":{"direct_trellis_copy_wall_s":1.0,"generic_copy_wall_s":0.0,"direct_trellis_prep_wall_s":0.0}}
    out=m.summarize(payload,loader)
    assert out["by_suffix"]["trellis"]["count"] == 2
    assert out["by_suffix"]["trellis"]["consumer_wall_s"] == 2.0
    assert out["totals"]["shards"] == 2
