import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "kvmem_qsa_shadow_summarize.py"


def _load():
    spec = importlib.util.spec_from_file_location("shadow_sum", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def test_shadow_summary_filters_to_query_rows_and_history():
    mod = _load()
    case = {
        "name": "x",
        "target_tokens": 10000,
        "query_span": [9000, 9010],
        "needles": [
            {"marker": "N", "code": "c", "start": 1000, "end": 1010}
        ],
    }
    records = [
        {
            "schema": 1,
            "layer_name": "l0",
            "skip_topk": False,
            "rows": [
                {"pos": 8999, "selected": [1001, 5000]},
                {"pos": 9001, "selected": [1001, 5000, 9002]},
            ],
        },
        {
            "schema": 1,
            "layer_name": "l1",
            "skip_topk": False,
            "rows": [{"pos": 9002, "selected": [1002, 6000]}],
        },
        {
            "schema": 1,
            "layer_name": "mtp",
            "skip_topk": True,
            "rows": [{"pos": 9003, "selected": [1003]}],
        },
    ]
    out = mod.summarize_case(case, records)
    assert out["query_rows_used"] == 2
    assert out["layers_used"] == 2
    assert out["direct_needle_hits"][0]["direct_qsa_token_hit"] is True
    assert out["direct_needle_hits"][0]["direct_vote_count"] == 2
    assert out["policies"]["b256_budget32768"]["needles"][0][
        "all_required_blocks_selected"
    ] is True


def test_suite_primary_go_signal_requires_64k_and_32k_recall(tmp_path):
    mod = _load()
    case = {
        "schema": 1,
        "name": "c",
        "target_tokens": 10000,
        "prompt_token_ids": [1],
        "query_span": [9000, 9010],
        "needles": [
            {"marker": "N", "code": "c", "start": 1000, "end": 1010}
        ],
    }
    case_path = tmp_path / "c.json"
    case_path.write_text(__import__("json").dumps(case))
    manifest = {"cases": [{"name": "c", "path": str(case_path)}]}
    shadow = tmp_path / "c.jsonl"
    shadow.write_text(
        __import__("json").dumps({
            "schema": 1,
            "layer_name": "l0",
            "skip_topk": False,
            "rows": [{"pos": 9001, "selected": [1001, 5000]}],
        }) + "\n"
    )
    out = mod.summarize_suite(manifest, tmp_path)
    assert out["suite"]["needle_count"] == 1
    assert out["suite"]["primary_policy"]["go_signal"] is True
