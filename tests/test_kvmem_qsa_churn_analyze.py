import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "kvmem_qsa_churn_analyze.py"


def _load():
    spec = importlib.util.spec_from_file_location("churn", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def _turn(name, index, marker, start):
    return {
        "index": index,
        "name": name,
        "prompt_tokens": 10000,
        "query_span": [9000, 9010],
        "target_facts": [{
            "marker": marker,
            "start": start,
            "end": start + 8,
        }],
    }


def test_extract_votes_uses_query_rows_and_history_only():
    mod = _load()
    turn = _turn("a", 0, "A", 1000)
    records = [{
        "schema": 1,
        "layer_name": "l0",
        "skip_topk": False,
        "rows": [
            {"pos": 8999, "selected": [1001]},
            {"pos": 9001, "selected": [1001, 5000, 9002]},
        ],
    }]
    out = mod.extract_votes(turn, records)
    assert out["query_rows"] == 1
    assert out["votes"][1001] == 1
    assert out["votes"][5000] == 1
    assert 9002 not in out["votes"]
    assert out["targets"][0]["direct_qsa_token_hit"] is True


def test_working_set_and_kv_geometry_are_deterministic():
    mod = _load()
    turn = _turn("a", 0, "A", 1000)
    votes = __import__("collections").Counter({1001: 10, 5000: 5, 7000: 1})
    chosen, _, mandatory = mod.working_set(
        turn, votes, block_size=256, budget=32768
    )
    assert 1001 // 256 in chosen
    assert mandatory <= chosen

    geom = mod.kv_geometry(
        {
            "num_key_value_heads": 4,
            "head_dim": 128,
            "hidden_size": 2560,
            "num_attention_heads": 20,
        },
        qsa_layers=12,
    )
    assert geom["main_kv_bytes_per_token"] == 12 * 2 * 4 * 128 * 2


def test_classification_low_churn_contract():
    mod = _load()
    ctx = {
        "policies": {
            "b256_budget65536": {
                "summary": {
                    "median_stage_in_fraction": 0.10,
                    "max_stage_in_fraction": 0.20,
                    "same_query_replay_jaccard": 1.0,
                    "target_fact_recall_rate": 1.0,
                }
            }
        }
    }
    out = mod.classify([ctx, ctx])
    assert out["classification"] == "LOW_CHURN"
    assert out["primary_policy"] == "b256_budget65536"
