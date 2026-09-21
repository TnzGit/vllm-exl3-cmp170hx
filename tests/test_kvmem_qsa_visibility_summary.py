import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "kvmem_qsa_visibility_summarize.py"


def _load():
    spec = importlib.util.spec_from_file_location("q1_summary", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def _plan():
    return {
        "context": 160000,
        "turn": "ask_d_e",
        "budget_tokens": 65536,
        "replacement_fraction": 0.05,
        "resident_region_count": 256,
        "expected_qsa_layers": 1,
    }


def test_exact_parity_go_when_mask_is_exercised():
    mod = _load()
    baseline = {
        "target_codes_in_order": True,
        "text": "violet-harbor-31,granite-comet-72",
        "logprob_tokens": ["violet", "-", "harbor"],
    }
    masked = dict(baseline)
    stats = [{
        "layer_name": "layer.1",
        "rows_applied": 2,
        "selected_valid_before": 100,
        "historical_selected": 90,
        "historical_resident_kept": 75,
        "active_selected_kept": 10,
        "historical_selected_dropped": 15,
    }]
    out = mod.summarize(baseline, masked, stats, _plan())
    assert out["semantic_go"] is True
    assert out["classification"] == "EXACT_PARITY_GO"
    assert out["visibility"]["mask_exercised"] is True
    assert out["visibility"]["historical_selected_dropped"] == 15
    assert out["visibility"]["layer_coverage_ok"] is True
    assert out["visibility"]["accounting_ok"] is True


def test_correct_but_nonexact_output_is_semantic_go():
    mod = _load()
    baseline = {
        "target_codes_in_order": True,
        "text": "violet-harbor-31,granite-comet-72",
        "logprob_tokens": ["a"],
    }
    masked = {
        "target_codes_in_order": True,
        "text": "violet-harbor-31, granite-comet-72",
        "logprob_tokens": ["b"],
    }
    stats = [{
        "layer_name": "layer.1",
        "rows_applied": 1,
        "selected_valid_before": 10,
        "historical_selected": 10,
        "historical_resident_kept": 8,
        "active_selected_kept": 0,
        "historical_selected_dropped": 2,
    }]
    out = mod.summarize(baseline, masked, stats, _plan())
    assert out["semantic_go"] is True
    assert out["exact_token_parity"] is False
    assert out["classification"] == "SEMANTIC_GO_NONEXACT"


def test_mask_must_actually_drop_historical_selection():
    mod = _load()
    baseline = {
        "target_codes_in_order": True,
        "text": "ok",
        "logprob_tokens": ["ok"],
    }
    masked = dict(baseline)
    stats = [{
        "layer_name": "layer.1",
        "rows_applied": 1,
        "selected_valid_before": 10,
        "historical_selected": 10,
        "historical_resident_kept": 10,
        "active_selected_kept": 0,
        "historical_selected_dropped": 0,
    }]
    out = mod.summarize(baseline, masked, stats, _plan())
    assert out["semantic_go"] is False
    assert out["classification"] == "INVALID_MASK_EVIDENCE"


def test_baseline_failure_is_classified_as_measurement_invalid():
    mod = _load()
    baseline = {
        "target_codes_in_order": False,
        "text": "<think> truncated",
        "logprob_tokens": ["<think>"],
    }
    masked = {
        "target_codes_in_order": False,
        "text": "<think> truncated",
        "logprob_tokens": ["<think>"],
    }
    stats = [{
        "layer_name": "layer.1",
        "rows_applied": 1,
        "selected_valid_before": 10,
        "historical_selected": 10,
        "historical_resident_kept": 8,
        "active_selected_kept": 0,
        "historical_selected_dropped": 2,
    }]
    out = mod.summarize(baseline, masked, stats, _plan())
    assert out["baseline_measurement_valid"] is False
    assert out["evidence_valid"] is True
    assert out["semantic_go"] is False
    assert out["classification"] == "BASELINE_INVALID"


def test_wrong_masked_target_is_real_semantic_no_go():
    mod = _load()
    baseline = {
        "target_codes_in_order": True,
        "text": "correct",
        "logprob_tokens": ["a"],
    }
    masked = {
        "target_codes_in_order": False,
        "text": "wrong",
        "logprob_tokens": ["b"],
    }
    stats = [{
        "layer_name": "layer.1",
        "rows_applied": 1,
        "selected_valid_before": 10,
        "historical_selected": 10,
        "historical_resident_kept": 8,
        "active_selected_kept": 0,
        "historical_selected_dropped": 2,
    }]
    out = mod.summarize(baseline, masked, stats, _plan())
    assert out["semantic_go"] is False
    assert out["classification"] == "MASK_SEMANTIC_NO_GO"
