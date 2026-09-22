import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "r0_metadata_full_bulk_qual_summary.py"


def _load():
    spec = importlib.util.spec_from_file_location("full_bulk_summary", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _log(path: Path, main: float, draft: float) -> None:
    path.write_text(
        f"Loading weights took {main:.2f} seconds\n"
        f"Loading weights took {draft:.2f} seconds\n"
    )


def _cell(path: Path, tokens=None) -> None:
    path.write_text(json.dumps({"token_pieces": tokens or ["a", "b", "c"]}))


def test_full_bulk_summary_qualified(tmp_path: Path):
    m = _load()
    ca, full, cb = (tmp_path / x for x in ("ca.log", "full.log", "cb.log"))
    _log(ca, 100.0, 20.0)
    _log(full, 96.0, 19.0)
    _log(cb, 101.0, 20.5)
    ca_cell, f_cell, cb_cell = (tmp_path / x for x in ("ca.json", "f.json", "cb.json"))
    for p in (ca_cell, f_cell, cb_cell):
        _cell(p)

    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps({
        "tag": "ALL_WEIGHTS_LOADED_BEFORE_POSTLOAD",
        "metadata_bulk_ab": {
            "mode": "full",
            "full_enabled": True,
            "control_calls": 0,
            "deferred_calls": 100,
            "deferred_bytes": 1000,
            "commit_calls": 10,
            "commit_bytes": 1000,
            "strided_commit_calls": 10,
            "index_commit_calls": 0,
            "committed_layers": 2,
        },
    }) + "\n")
    xid = tmp_path / "xid.json"
    xid.write_text(json.dumps({"xid_delta": 0}))
    restore = tmp_path / "restore.json"
    restore.write_text(json.dumps({
        "weight_utils_before": "aa", "weight_utils_after": "aa",
        "qsa_before": "bb", "qsa_after": "bb",
    }))

    out = m.summarize(
        ca, full, cb, ca_cell, f_cell, cb_cell, trace, xid, restore
    )
    assert out["qualification_valid"] is True
    assert out["performance_gate_pass"] is True
    assert out["classification"] == "METADATA_FULL_BULK_QUALIFIED"
    assert out["correctness"]["greedy_token_parity"] is True
    assert out["bulk_accounting"]["valid"] is True


def test_full_bulk_summary_rejects_index_fallback(tmp_path: Path):
    m = _load()
    ca, full, cb = (tmp_path / x for x in ("ca.log", "full.log", "cb.log"))
    for p, main in ((ca, 100.0), (full, 96.0), (cb, 100.0)):
        _log(p, main, 20.0)
    ca_cell, f_cell, cb_cell = (tmp_path / x for x in ("ca.json", "f.json", "cb.json"))
    for p in (ca_cell, f_cell, cb_cell):
        _cell(p)
    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps({
        "tag": "ALL_WEIGHTS_LOADED_BEFORE_POSTLOAD",
        "metadata_bulk_ab": {
            "mode": "full", "full_enabled": True,
            "control_calls": 0, "deferred_calls": 100,
            "deferred_bytes": 1000, "commit_calls": 10, "commit_bytes": 1000,
            "strided_commit_calls": 9, "index_commit_calls": 1,
            "committed_layers": 2,
        },
    }) + "\n")
    xid = tmp_path / "xid.json"
    xid.write_text('{"xid_delta":0}')
    restore = tmp_path / "restore.json"
    restore.write_text(json.dumps({
        "weight_utils_before": "aa", "weight_utils_after": "aa",
        "qsa_before": "bb", "qsa_after": "bb",
    }))
    out = m.summarize(
        ca, full, cb, ca_cell, f_cell, cb_cell, trace, xid, restore
    )
    assert out["qualification_valid"] is False
    assert out["classification"] == "METADATA_FULL_BULK_INVALID"
