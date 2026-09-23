#!/usr/bin/env python3
"""Fail-closed summary of a completed non-KVMEM C1 prefill scan."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any


CONFIGS = ("auto", "1024", "4096")
CONTEXTS = (15533, 79533, 159533)
MAX_TOKENS = 256
METRICS = ("ttft_s", "prefill_tok_s", "tpot_s", "wall_s")


class EvidenceError(ValueError):
    """Raised when evidence is missing, malformed, or fails the scan contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def _read_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except FileNotFoundError as exc:
        raise EvidenceError(f"missing required evidence: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read JSON evidence {path}: {exc}") from exc


def _read_kv(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise EvidenceError(f"cannot read required evidence {path}: {exc}") from exc
    result: dict[str, str] = {}
    for line in lines:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def _int(value: Any, label: str, *, minimum: int | None = None) -> int:
    _require(not isinstance(value, bool), f"{label} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise EvidenceError(f"{label} must be an integer") from exc
    _require(str(parsed) == str(value).strip(), f"{label} must be an exact integer")
    if minimum is not None:
        _require(parsed >= minimum, f"{label} must be >= {minimum}")
    return parsed


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    _require(not isinstance(value, bool), f"{label} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise EvidenceError(f"{label} must be numeric") from exc
    _require(math.isfinite(parsed), f"{label} must be finite")
    _require(parsed > 0 if positive else parsed >= 0, f"{label} is out of range")
    return parsed


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-9)


def _check_metrics(sample: dict[str, Any], label: str) -> dict[str, float]:
    tokens = _int(sample.get("prompt_tokens"), f"{label}.prompt_tokens", minimum=1)
    completion = _int(sample.get("completion_tokens"), f"{label}.completion_tokens", minimum=2)
    ttft = _number(sample.get("ttft_s"), f"{label}.ttft_s", positive=True)
    wall = _number(sample.get("wall_s"), f"{label}.wall_s", positive=True)
    prefill = _number(sample.get("prefill_tok_s"), f"{label}.prefill_tok_s", positive=True)
    tpot = _number(sample.get("tpot_s"), f"{label}.tpot_s", positive=True)
    _require(wall >= ttft, f"{label}: wall_s is less than ttft_s")
    _require(_close(prefill, tokens / ttft), f"{label}: prefill_tok_s has an invalid denominator")
    _require(_close(tpot, (wall - ttft) / (completion - 1)),
             f"{label}: tpot_s has an invalid denominator")
    _require(_number(sample.get("preemption_delta"), f"{label}.preemption_delta") == 0,
             f"{label}: preemption detected")
    return {"ttft_s": ttft, "prefill_tok_s": prefill, "tpot_s": tpot, "wall_s": wall}


def _validate_cell(path: Path, *, expected_context: int, expected_repeats: int,
                   label: str, expected_answer: str, sentinel: bool = False) -> dict[str, Any]:
    cell = _read_json(path)
    _require(isinstance(cell, dict), f"{label}: cell JSON must be an object")
    _require(cell.get("benchmark") == "non_kvmem_c1_mtp_k3_prefill_scan",
             f"{label}: unexpected benchmark identity")
    _require(cell.get("status") in ("VALID", "VALID_PERFORMANCE_WITH_OUTPUT_VARIATION"),
             f"{label}: cell status is not valid")
    _require(cell.get("context_requested") == expected_context,
             f"{label}: requested context does not match filename")
    _require(cell.get("max_tokens_requested") == MAX_TOKENS,
             f"{label}: expected {MAX_TOKENS} requested output tokens")
    _require(cell.get("prompt_source") == "token_ids", f"{label}: prompt was not exact token IDs")
    prompt_hash = cell.get("prompt_sha256")
    _require(isinstance(prompt_hash, str) and re.fullmatch(r"[0-9a-f]{64}", prompt_hash) is not None,
             f"{label}: missing or malformed prompt_sha256")
    _require(isinstance(cell.get("output_parity"), bool), f"{label}: output_parity must be boolean")

    samples = cell.get("samples")
    _require(isinstance(samples, list) and len(samples) == expected_repeats,
             f"{label}: expected {expected_repeats} complete measured samples")
    seen_repeats: set[int] = set()
    clean_samples: list[dict[str, Any]] = []
    hashes: set[str] = set()
    prompt_counts: set[int] = set()
    for index, sample in enumerate(samples, 1):
        _require(isinstance(sample, dict), f"{label}: sample {index} must be an object")
        repeat = _int(sample.get("repeat"), f"{label}.samples[{index}].repeat", minimum=1)
        _require(repeat not in seen_repeats, f"{label}: duplicate repeat number {repeat}")
        seen_repeats.add(repeat)
        _require(repeat <= expected_repeats and "error" not in sample,
                 f"{label}: sample {repeat} is incomplete")
        _require(_int(sample.get("completion_tokens"), f"{label}.sample[{repeat}].completion_tokens") == MAX_TOKENS,
                 f"{label}: incorrect output count in sample {repeat}")
        _check_metrics(sample, f"{label}.sample[{repeat}]")
        prompt_counts.add(_int(sample.get("prompt_tokens"), f"{label}.sample[{repeat}].prompt_tokens", minimum=1))
        digest = sample.get("output_text_sha256")
        _require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
                 f"{label}: malformed output hash in sample {repeat}")
        output_text = sample.get("output_text")
        _require(isinstance(output_text, str) and hashlib.sha256(output_text.encode("utf-8")).hexdigest() == digest,
                 f"{label}: output hash does not match saved text in sample {repeat}")
        # ignore_eos keeps generating after the first answer and may even
        # begin a second synthetic turn; check the first completed answer.
        _require("</think>" in output_text and output_text.split("</think>", 1)[1].lstrip().startswith(expected_answer),
                 f"{label}: measured sample {repeat} has no correct visible final answer")
        hashes.add(digest)
        clean_samples.append(sample)
    _require(seen_repeats == set(range(1, expected_repeats + 1)), f"{label}: measured repeats are not complete")
    _require(len(prompt_counts) == 1, f"{label}: prompt token counts differ between repeats")
    _require(cell["output_parity"] == (len(hashes) == 1), f"{label}: output_parity disagrees with output hashes")
    expected_status = "VALID" if cell["output_parity"] else "VALID_PERFORMANCE_WITH_OUTPUT_VARIATION"
    _require(cell.get("status") == expected_status, f"{label}: cell status disagrees with output parity")

    warmup = cell.get("warmup")
    _require(isinstance(warmup, dict), f"{label}: warmup evidence is missing")
    _require(_int(warmup.get("completion_tokens"), f"{label}.warmup.completion_tokens", minimum=1) == min(16, MAX_TOKENS),
             f"{label}: incorrect warmup output count")
    _require(_number(warmup.get("preemption_delta"), f"{label}.warmup.preemption_delta") == 0,
             f"{label}: warmup preemption detected")
    _require(_int(warmup.get("prompt_tokens"), f"{label}.warmup.prompt_tokens", minimum=1) in prompt_counts,
             f"{label}: warmup prompt token count differs from measured requests")

    if sentinel:
        _require(len(clean_samples) == 1, f"{label}: sentinel must contain exactly one sample")
    return {"cell": cell, "prompt_sha256": prompt_hash,
            "prompt_tokens": next(iter(prompt_counts)), "samples": clean_samples,
            "metrics": [_check_metrics(s, f"{label}.sample[{s['repeat']}]") for s in clean_samples],
            "parity": cell["output_parity"]}


def _read_identity(root: Path) -> tuple[dict[str, str], int]:
    identity = _read_kv(root / "run_identity.txt")
    _require(identity.get("run_id"), "run_identity.txt is missing run_id")
    expected_sha = identity.get("expected_sha", "")
    _require(re.fullmatch(r"[0-9a-f]{40}", expected_sha) is not None and identity.get("repo_head") == expected_sha,
             "run_identity exact repo SHA is missing or mismatched")
    _require(re.fullmatch(r"[0-9a-f]{64}", identity.get("installed_plugin_sha256", "")) is not None,
             "run_identity installed plugin hash is missing")
    context_specs = ("15533:ctx16000", "79533:ctx80000", "159533:ctx160000")
    _require(identity.get("contexts", "").split() == list(context_specs),
             "run_identity contexts do not match required context set/order")
    _require(identity.get("candidates", "").split() == list(CONFIGS),
             "run_identity candidates must be exactly auto 1024 4096")
    repeats = _int(identity.get("repeats"), "run_identity repeats", minimum=1)
    _require(_int(identity.get("max_tokens"), "run_identity max_tokens") == MAX_TOKENS,
             f"run_identity max_tokens must be {MAX_TOKENS}")
    _require(identity.get("cell_contract") == "one_warmup_then_repeats_measured_requests_per_context",
             "run_identity cell contract is unexpected")
    return identity, repeats


def _validate_run_lifecycle(root: Path, identity: dict[str, str], *,
                            partial_4096_boot_capacity: bool = False) -> None:
    run_exit = _read_kv(root / "run_exit.txt")
    _require(run_exit.get("run_id") == identity["run_id"], "run_exit run_id mismatch")
    expected_exit = 2 if partial_4096_boot_capacity else 0
    _require(_int(run_exit.get("exit_status"), "run_exit exit_status") == expected_exit,
             "scan exit status does not match requested classification")

    events_path = root / "events.log"
    try:
        events = events_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise EvidenceError(f"cannot read required evidence {events_path}: {exc}") from exc
    _require(events, "events.log is empty")
    run_lines = [line for line in events if f"run={identity['run_id']} " in line]
    _require(run_lines, "events.log has no events for run_identity run_id")
    if partial_4096_boot_capacity:
        _require(any("CONFIG_START config=4096" in line for line in run_lines),
                 "events.log is missing CONFIG_START for 4096")
        _require(any("STARTUP_FAILED config=4096" in line for line in run_lines),
                 "events.log is missing the 4096 startup failure")
        _require(any("STOPPED config=batched_tokens_4096" in line for line in run_lines),
                 "events.log is missing 4096 process cleanup")
        _require(any("RUN_EXIT status=2" in line for line in run_lines),
                 "events.log is missing the failed-run exit status")
        _require(not any("RUN_COMPLETE " in line or "ENGINE_HEALTHY config=4096" in line
                         or "CELL_" in line and "config=4096" in line
                         or "SENTINEL_" in line and "config=4096" in line
                         or "CONFIG_DONE config=4096" in line for line in run_lines),
                 "events.log shows 4096 progressed beyond startup failure")
        errors = [line for line in run_lines if " ERROR " in f" {line} "]
        _require(len(errors) == 1 and "4096" in errors[0] and "engine failed to become healthy" in errors[0],
                 "events.log contains an unexpected error")
        _require(not any(token in f" {line} " for line in run_lines
                         for token in ("CELL_FAILED ", "SENTINEL_FAILED ")),
                 "events.log records a failed cell or sentinel")
    else:
        _require(any("RUN_COMPLETE xid_delta=0" in line for line in run_lines),
                 "events.log has no clean RUN_COMPLETE event")
        _require(any("RUN_EXIT status=0" in line for line in run_lines),
                 "events.log has no successful RUN_EXIT event")
        failures = (" ERROR ", "CELL_FAILED ", "SENTINEL_FAILED ", "STARTUP_FAILED ")
        _require(not any(any(token in f" {line} " for token in failures) for line in run_lines),
                 "events.log records a failed scan step")
    configs_to_validate = CONFIGS[:2] if partial_4096_boot_capacity else CONFIGS
    for config in configs_to_validate:
        name = f"batched_tokens_{config}"
        _require(any(f"CONFIG_DONE config={config}" in line for line in run_lines),
                 f"events.log is missing CONFIG_DONE for {config}")
        _require(any(f"SENTINEL_DONE config={config}" in line for line in run_lines),
                 f"events.log is missing SENTINEL_DONE for {config}")
        for context in CONTEXTS:
            _require(any(f"CELL_DONE config={config} context={context}" in line for line in run_lines),
                     f"events.log is missing CELL_DONE for {config}/{context}")


def _validate_xids(root: Path, *, partial_4096_boot_capacity: bool = False) -> None:
    before = _read_kv(root / "xid_before.txt")
    before_count = _int(before.get("xid_count_before"), "xid_before count", minimum=0)
    after_path = root / "xid_after.txt"
    if after_path.exists():
        after = _read_kv(after_path)
        after_count = _int(after.get("xid_count_after"), "xid_after count", minimum=0)
        _require(before_count == after_count, "global Xid count changed during scan")
        _require(_int(after.get("xid_delta"), "global Xid delta") == 0, "global Xid delta is nonzero")
    else:
        _require(partial_4096_boot_capacity,
                 f"missing required evidence: {after_path}")
    for config in CONFIGS:
        data = _read_kv(root / f"batched_tokens_{config}" / "xid.txt")
        first = _int(data.get("xid_count_before"), f"{config} Xid before", minimum=0)
        last = _int(data.get("xid_count_after"), f"{config} Xid after", minimum=0)
        _require(first == before_count, f"{config} Xid baseline differs from run baseline")
        _require(first == last, f"Xid count changed during {config}")
        _require(_int(data.get("xid_delta"), f"{config} Xid delta") == 0,
                 f"Xid delta is nonzero during {config}")


def _validate_ports(root: Path, *, partial_4096_boot_capacity: bool = False) -> None:
    before = _read_kv(root / "port_before.txt")
    _require(before.get("state") == "CLOSED", "scan port was not closed before run")
    for config in CONFIGS[:2] if partial_4096_boot_capacity else CONFIGS:
        base = root / f"batched_tokens_{config}"
        healthy = _read_kv(base / "port_healthy.txt")
        after = _read_kv(base / "port_after.txt")
        _require(healthy.get("state") == "OPEN", f"{config}: serving port was not open when healthy")
        _require(after.get("state") == "CLOSED", f"{config}: serving port remained open after cleanup")
        for record in (healthy, after):
            _require(record.get("port") == before.get("port") and record.get("host") == before.get("host"),
                     f"{config}: port/host differs from initial snapshot")
    if partial_4096_boot_capacity:
        base = root / "batched_tokens_4096"
        after = _read_kv(base / "port_after.txt")
        _require(after.get("state") == "CLOSED", "4096: serving port remained open after failed startup")
        _require(after.get("port") == before.get("port") and after.get("host") == before.get("host"),
                 "4096: port/host differs from initial snapshot")
        _require(not (base / "port_healthy.txt").exists(), "4096: failed startup unexpectedly has healthy port evidence")


def _validate_process_cleanup(root: Path, configs: tuple[str, ...]) -> None:
    for config in configs:
        path = root / f"processes_batched_tokens_{config}_after.txt"
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
            raise EvidenceError(f"cannot read required cleanup evidence {path}: {exc}") from exc
        _require(len(lines) >= 2 and lines[1].strip().startswith("PID "),
                 f"{config}: process cleanup snapshot is malformed")
        _require(not any(line.strip() for line in lines[2:]),
                 f"{config}: scan processes remain after cleanup")


def _validate_final_gpu(root: Path) -> None:
    try:
        lines = (root / "gpu_final.csv").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise EvidenceError(f"cannot read final GPU evidence: {exc}") from exc
    _require(any(line == "compute_processes:" for line in lines),
             "gpu_final.csv has no compute-process marker")
    marker = lines.index("compute_processes:")
    _require(not any(line.strip() for line in lines[marker + 1:]),
             "GPU still has compute processes at final snapshot")


def _validate_fingerprints(root: Path, configs: tuple[str, ...] = CONFIGS) -> None:
    baseline = _read_json(root / "installed_source_before.json")
    _require(isinstance(baseline, dict) and re.fullmatch(
        r"[0-9a-f]{64}", str(baseline.get("sha256_python_sources", ""))) is not None,
        "installed_source_before.json has no valid source fingerprint")
    for config in configs:
        base = root / f"batched_tokens_{config}"
        for filename in ("installed_source.json", "installed_source_after.json"):
            current = _read_json(base / filename)
            _require(current == baseline,
                     f"{config}/{filename} differs from baseline installed source fingerprint")


def _summary(values: list[float]) -> dict[str, float]:
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def _expected_answer(root: Path, context: int) -> str:
    case = _read_json(root / "prompt_cases" / f"prompt_{context}.json")
    _require(isinstance(case, dict), f"{context}: prompt case must be an object")
    facts = case.get("target_facts")
    query = case.get("query_text")
    _require(isinstance(facts, list) and len(facts) == 2 and isinstance(query, str),
             f"{context}: prompt case lacks the two target facts/query")
    codes: list[str] = []
    for fact in facts:
        _require(isinstance(fact, dict) and isinstance(fact.get("marker"), str)
                 and isinstance(fact.get("code"), str) and fact["marker"] in query,
                 f"{context}: target fact is malformed or absent from query")
        codes.append(fact["code"])
    return ", ".join(codes)


def _validate_4096_boot_capacity_failure(root: Path) -> dict[str, str]:
    base = root / "batched_tokens_4096"
    startup = _read_kv(base / "startup.txt")
    _require(startup.get("config") == "4096" and startup.get("startup_failed_utc")
             and startup.get("server_exit_status") == "1" and startup.get("server_exit_utc"),
             "4096: startup failure evidence is incomplete")
    _require(not startup.get("healthy_utc") and not startup.get("completed_utc"),
             "4096: startup evidence unexpectedly records a healthy or completed engine")
    try:
        log = (base / "server.log").read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise EvidenceError(f"cannot read required 4096 startup log: {exc}") from exc
    reason = re.compile(
        r"ValueError:.*max seq len \(246000\).*\(7\.0 GiB KV cache is needed, "
        r"which is larger than the available KV cache memory \(6\.63 GiB\)", re.DOTALL)
    matching_reasons = [line[line.index("ValueError:"):].strip()
                        for line in log.splitlines()
                        if "ValueError:" in line and reason.search(line) is not None]
    _require(matching_reasons,
             "4096: server log does not contain the expected 246000-token KV capacity ValueError")
    _require(not list(base.glob("cell_*")) and not (base / "metrics_healthy.prom").exists(),
             "4096: cell or healthy-engine metrics exist despite startup failure")
    _require(not (base / "port_healthy.txt").exists(),
             "4096: healthy-port evidence exists despite startup failure")
    _require(not (base / "installed_source.json").exists()
             and not (base / "installed_source_after.json").exists(),
             "4096: serving source fingerprints exist despite startup failure")
    return {"failure_type": "vllm_kv_cache_capacity", "requested_max_model_len": "246000",
            "required_kv_cache": "7.0 GiB", "available_kv_cache": "6.63 GiB",
            "raw_log_reason": matching_reasons[0]}


def analyze(scan_dir: Path, *, classify_partial_4096_boot_capacity: bool = False) -> dict[str, Any]:
    root = scan_dir.expanduser().resolve()
    _require(root.is_dir(), f"scan directory does not exist or is not a directory: {root}")
    identity, repeats = _read_identity(root)
    failure = (_validate_4096_boot_capacity_failure(root)
               if classify_partial_4096_boot_capacity else None)
    completed_configs = CONFIGS[:2] if failure is not None else CONFIGS
    _validate_run_lifecycle(root, identity, partial_4096_boot_capacity=failure is not None)
    _validate_xids(root, partial_4096_boot_capacity=failure is not None)
    _validate_ports(root, partial_4096_boot_capacity=failure is not None)
    if failure is not None:
        _validate_process_cleanup(root, CONFIGS)
    _validate_final_gpu(root)
    _validate_fingerprints(root, completed_configs)

    data: dict[str, dict[int, dict[str, Any]]] = {}
    reference: dict[int, tuple[str, int]] = {}
    expected_answers = {context: _expected_answer(root, context) for context in CONTEXTS}
    summaries: dict[str, Any] = {}
    for config in completed_configs:
        base = root / f"batched_tokens_{config}"
        startup = _read_kv(base / "startup.txt")
        _require(startup.get("config") == config and startup.get("healthy_utc") and startup.get("completed_utc"),
                 f"{config}: startup evidence is incomplete")
        data[config] = {}
        summaries[config] = {}
        for context in CONTEXTS:
            label = f"{config}/{context}"
            status = _read_kv(base / f"cell_{context}.status.txt")
            _require(_int(status.get("cell_exit_status"), f"{label} cell exit", minimum=0) == 0,
                     f"{label}: cell process did not exit cleanly")
            cell = _validate_cell(base / f"cell_{context}.json", expected_context=context,
                                  expected_repeats=repeats, label=label,
                                  expected_answer=expected_answers[context])
            key = (cell["prompt_sha256"], cell["prompt_tokens"])
            if context in reference:
                _require(key == reference[context], f"{label}: prompt identity/count differs across configurations")
            else:
                reference[context] = key
            data[config][context] = cell
            summaries[config][str(context)] = {
                "prompt_tokens": cell["prompt_tokens"],
                "prompt_sha256": cell["prompt_sha256"],
                "output_parity": cell["parity"],
                "visible_final_answer_correct": True,
                "classification": "prefill_only_output_variation" if not cell["parity"] else "prefill_only_256_token_ignore_eos",
                "metrics": {metric: _summary([row[metric] for row in cell["metrics"]]) for metric in METRICS},
            }

            sentinel = (_validate_cell(base / "cell_15533_sentinel.json",
                                       expected_context=15533, expected_repeats=1,
                                       label=f"{config}/sentinel",
                                       expected_answer=expected_answers[15533], sentinel=True)
                        if context == CONTEXTS[0] else None)
            if sentinel is not None:
                _require((sentinel["prompt_sha256"], sentinel["prompt_tokens"]) ==
                         (data[config][15533]["prompt_sha256"], data[config][15533]["prompt_tokens"]),
                         f"{config}: sentinel prompt identity/count differs from measured 15533 cell")
                summaries[config]["sentinel_15533"] = {
                    "output_parity": sentinel["parity"],
                    "classification": "prefill_only_256_token_ignore_eos",
                    "metrics": {metric: _summary([row[metric] for row in sentinel["metrics"]]) for metric in METRICS},
                }

    comparisons: dict[str, Any] = {}
    for config in completed_configs:
        if config == "auto":
            continue
        comparisons[config] = {}
        for context in CONTEXTS:
            comparisons[config][str(context)] = {}
            for metric in METRICS:
                auto = summaries["auto"][str(context)]["metrics"][metric]["median"]
                candidate = summaries[config][str(context)]["metrics"][metric]["median"]
                comparisons[config][str(context)][metric] = {
                    "auto_median": auto,
                    "candidate_median": candidate,
                    "change_pct_vs_auto": (candidate / auto - 1.0) * 100.0,
                }

    report = {
        "schema": 1,
        "analyzer": "r0_summarize_prefill_scan",
        "status": ("VALID_PARTIAL_4096_BOOT_CAPACITY_NO_GO" if failure is not None
                   else "VALID_PREFILL_PERFORMANCE_ONLY"),
        "scan_dir": str(root),
        "run_id": identity["run_id"],
        "configs": list(CONFIGS),
        "completed_configs": list(completed_configs),
        "contexts": list(CONTEXTS),
        "repeats_per_cell": repeats,
        "requested_output_tokens": MAX_TOKENS,
        "decode_semantic_qualification": False,
        "qualification_note": "256-token ignore_eos continuations are prefill-only evidence; output parity does not qualify decode semantics.",
        "summary": summaries,
        "comparisons_vs_auto": comparisons,
    }
    if failure is not None:
        report["failed_config"] = {"config": "4096", **failure,
                                   "cell_metrics": "not produced; no 4096 request was sent"}
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, type=Path, help="completed scan OUT directory")
    parser.add_argument("--output", type=Path, help="write JSON report to a new file (existing files are never overwritten)")
    parser.add_argument("--classify-partial-4096-boot-capacity", action="store_true",
                        help="validate the specific 4096 vLLM KV-capacity startup failure")
    args = parser.parse_args(argv)
    try:
        report = analyze(args.dir,
                         classify_partial_4096_boot_capacity=args.classify_partial_4096_boot_capacity)
        status = 0
    except EvidenceError as exc:
        report = {"schema": 1, "analyzer": "r0_summarize_prefill_scan", "status": "INVALID",
                  "scan_dir": str(args.dir.expanduser().resolve()), "error": str(exc),
                  "decode_semantic_qualification": False}
        status = 2

    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    print(rendered, end="")
    if args.output is not None:
        target = args.output.expanduser()
        try:
            with target.open("x", encoding="utf-8") as stream:
                stream.write(rendered)
        except FileExistsError:
            print(f"error: refusing to overwrite existing output file: {target}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"error: cannot write output file {target}: {exc}", file=sys.stderr)
            return 2
    return status


if __name__ == "__main__":
    raise SystemExit(main())
