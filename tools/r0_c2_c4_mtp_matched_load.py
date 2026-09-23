#!/usr/bin/env python3
"""Run one C2/C4 MTP matched-load cell against an already-running local engine.

Default mode is a CPU-only plan/dry-run. Execution requires --execute, a frozen
exact-token manifest, and an observed-runtime proof JSON. This tool never
launches or stops an engine; invoke it once per fresh k/load engine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


PORT = 8002
MAX_OUTPUT_TOKENS = 256
MEASURED_WAVES = 10
MAX_SKEW_REPEATS = 3
MAX_MANIFEST_BYTES = 128 * 1024 * 1024
MAX_IDS_PER_PROMPT = 100_000
PROMPT_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
ENGINECORE_PREFIX = re.compile(r"\(EngineCore pid=[0-9]+\)")
SCHEDULED_TOKEN_LINE = re.compile(
    r"max_num_scheduled_tokens is set to\s+([0-9][0-9,]*)\s+based on the speculative decoding settings\."
)
KV_POOL_LINE = re.compile(r"GPU KV cache size:\s*([0-9][0-9,]*) tokens\b")
COUNTERS = {
    "drafts": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted": "vllm:spec_decode_num_accepted_tokens_total",
    "preemptions": "vllm:num_preemptions_total",
}
GAUGES = {
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
}
POINTS = {
    "c2_16k": {"concurrency": 2, "prompt_tokens": 15_533},
    "c2_80k": {"concurrency": 2, "prompt_tokens": 79_533},
    "c4_16k": {"concurrency": 4, "prompt_tokens": 15_533},
    "c4_32k": {"concurrency": 4, "prompt_tokens": 27_250},
}
FROZEN_CONFIG = {
    "coop": True,
    "graph_mode": "PIECEWISE",
    "text_only": True,
    "disk_ngram": True,
    "profiler": False,
    "prefix_caching": False,
    "gpu_memory_utilization": 0.92,
    "max_model_len": 246_000,
    "max_num_seqs": 4,
    "max_num_batched_tokens": "auto",
}


class CellError(RuntimeError):
    """A precondition or measurement gate failed."""


def parse_enginecore_startup_evidence(log_text: str) -> dict[str, Any]:
    """Extract only EngineCore startup observations; APIServer lines are not proof."""
    scheduled: list[tuple[int, str]] = []
    kv_sizes: list[tuple[int, str]] = []
    for line in log_text.splitlines():
        if not ENGINECORE_PREFIX.search(line):
            continue
        match = SCHEDULED_TOKEN_LINE.search(line)
        if match:
            scheduled.append((int(match.group(1).replace(",", "")), line.strip()))
        match = KV_POOL_LINE.search(line)
        if match:
            kv_sizes.append((int(match.group(1).replace(",", "")), line.strip()))
    if len(scheduled) != 1:
        raise CellError(
            "startup proof incomplete: expected exactly one EngineCore "
            "max_num_scheduled_tokens warning, found "
            f"{len(scheduled)}"
        )
    if not kv_sizes:
        raise CellError("startup proof incomplete: EngineCore KV-pool line is absent")
    if len({value for value, _ in kv_sizes}) != 1:
        raise CellError("startup proof incomplete: EngineCore KV-pool lines conflict")
    return {
        "observed_enginecore_max_num_scheduled_tokens": scheduled[0][0],
        "scheduled_tokens_log_line": scheduled[0][1],
        "observed_enginecore_kv_pool_tokens": kv_sizes[0][0],
        "kv_pool_log_lines": [line for _, line in kv_sizes],
    }


def make_runtime_proof(
    manifest: dict[str, Any],
    log_text: str,
    point: str,
    k: int,
    observed: dict[str, Any],
    expected_budget: int,
) -> dict[str, Any]:
    """Build proof from pinned expectations plus EngineCore-owned log evidence."""
    evidence = parse_enginecore_startup_evidence(log_text)
    if evidence["observed_enginecore_max_num_scheduled_tokens"] != expected_budget:
        raise CellError(
            "EngineCore scheduled-token observation differs from frozen auto budget"
        )
    if evidence["observed_enginecore_kv_pool_tokens"] < 1:
        raise CellError("EngineCore reported an invalid KV pool")
    if k not in (2, 3) or point not in POINTS:
        raise CellError("Invalid cell identity for runtime proof")
    if observed.get("scheduler_override_absent") is not True:
        raise CellError("Cannot infer auto batched-token budget with a scheduled-token override")
    proof = {
        "engine_id": observed["engine_id"],
        "model": manifest["model"],
        "num_speculative_tokens": k,
        "vllm_version": observed["vllm_version"],
        "model_pack_sha256": observed["model_pack_sha256"],
        "model_revision_sha256": manifest["provenance"]["model_revision_sha256"],
        "model_config_sha256": observed["model_config_sha256"],
        "tokenizer_sha256": observed["tokenizer_sha256"],
        "exllamav3_revision": observed["exllamav3_revision"],
        "installed_exl3_source_sha256": observed["installed_exl3_source_sha256"],
        "r0_source_commit": observed["r0_source_commit"],
        "installed_vllm_config_source_sha256": observed[
            "installed_vllm_config_source_sha256"
        ],
        "installed_vllm_arg_utils_source_sha256": observed[
            "installed_vllm_arg_utils_source_sha256"
        ],
        "driver_version": observed["driver_version"],
        "cuda_version": observed["cuda_version"],
        "torch_version": observed["torch_version"],
        "kv_pool_tokens": evidence["observed_enginecore_kv_pool_tokens"],
        "config": {
            **FROZEN_CONFIG,
            "effective_max_num_batched_tokens": expected_budget,
        },
        "effective_budget_evidence": {
            "kind": "vllm_0.29.0_enginecore_source_inference",
            "log_line_source": "EngineCore",
            "observed_enginecore_max_num_scheduled_tokens": evidence[
                "observed_enginecore_max_num_scheduled_tokens"
            ],
            "scheduled_tokens_log_line": evidence["scheduled_tokens_log_line"],
            "observed_enginecore_kv_pool_tokens": evidence[
                "observed_enginecore_kv_pool_tokens"
            ],
            "kv_pool_log_lines": evidence["kv_pool_log_lines"],
            "source_default_relation": "max_num_scheduled_tokens=None -> max_num_batched_tokens",
            "scheduler_override_absent": True,
            "engine_args_default_none_verified": True,
            "scheduled_token_cli_arg_absent": True,
            "batched_token_cli_arg_absent": True,
            "log_line_count": 1,
            "interpretation": "inferred from the vLLM 0.29.0 defaulting code and EngineCore warning; not directly logged as max_num_batched_tokens",
            "source_url": "https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/config/vllm.py",
        },
    }
    if proof["vllm_version"] != "0.29.0":
        raise CellError("Observed vLLM version is not the frozen 0.29.0")
    if proof["installed_exl3_source_sha256"] != manifest["provenance"][
        "installed_exl3_source_sha256"
    ]:
        raise CellError("Installed EXL3 source hash differs from frozen manifest")
    if proof["r0_source_commit"] != manifest["provenance"]["r0_source_commit"]:
        raise CellError("R0 source commit differs from frozen manifest")
    if proof["model_pack_sha256"] != manifest["provenance"]["model_pack_sha256"]:
        raise CellError("Model pack hash differs from frozen manifest")
    if proof["tokenizer_sha256"] != manifest["provenance"]["tokenizer_sha256"]:
        raise CellError("Tokenizer hash differs from frozen manifest")
    if proof["model_config_sha256"] != manifest["provenance"]["model_config_sha256"]:
        raise CellError("Model config hash differs from frozen manifest")
    validate_runtime_proof(proof, manifest, point, k)
    return proof


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def token_ids_sha256(ids: list[int]) -> str:
    payload = json.dumps(ids, separators=(",", ":")).encode("ascii")
    return sha256_bytes(payload)


def _load_json(path: Path, limit: int) -> tuple[Any, bytes]:
    if path.is_symlink() or not path.is_file():
        raise CellError(f"JSON input must be a regular non-symlink file: {path}")
    size = path.stat().st_size
    if size <= 0 or size > limit:
        raise CellError(f"JSON input size is invalid: {path} ({size} bytes)")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) != size:
        raise CellError(f"JSON input changed or was short-read: {path}")
    try:
        return json.loads(raw), raw
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CellError(f"Invalid JSON in {path}: {exc}") from exc


def load_prompt_manifest(path: Path) -> tuple[dict[str, Any], bytes]:
    doc, raw = _load_json(path, MAX_MANIFEST_BYTES)
    if not isinstance(doc, dict) or doc.get("schema") != 1:
        raise CellError("Prompt manifest must be an object with schema=1")
    provenance = doc.get("provenance")
    required_hashes = (
        "model_pack_sha256",
        "model_revision_sha256",
        "model_config_sha256",
        "tokenizer_sha256",
        "installed_exl3_source_sha256",
    )
    if not isinstance(provenance, dict):
        raise CellError("Prompt manifest has no provenance object")
    if not isinstance(doc.get("model"), str) or not doc["model"].strip():
        raise CellError("Prompt manifest must identify the model")
    for name in required_hashes:
        if not isinstance(provenance.get(name), str) or not PROMPT_SHA_RE.fullmatch(
            provenance[name]
        ):
            raise CellError(f"Missing or malformed provenance hash: {name}")
    source_commit = provenance.get("r0_source_commit")
    if not isinstance(source_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise CellError("Missing or malformed provenance commit: r0_source_commit")
    runtime_expectations = doc.get("runtime_expectations")
    expected_runtime_fields = (
        "vllm_version",
        "exllamav3_revision",
        "driver_version",
        "cuda_version",
        "torch_version",
        "effective_max_num_batched_tokens",
    )
    if not isinstance(runtime_expectations, dict):
        raise CellError("Prompt manifest has no frozen runtime_expectations")
    for name in expected_runtime_fields:
        if runtime_expectations.get(name) in (None, ""):
            raise CellError(f"Missing frozen runtime expectation: {name}")
    if runtime_expectations["vllm_version"] != "0.29.0":
        raise CellError("Frozen vLLM version must be 0.29.0")
    revision = runtime_expectations["exllamav3_revision"]
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise CellError("Malformed frozen ExLlamaV3 revision")
    if type(runtime_expectations["effective_max_num_batched_tokens"]) is not int:
        raise CellError("Frozen effective auto token budget must be an integer")
    for name in ("driver_version", "cuda_version", "torch_version"):
        if (
            not isinstance(runtime_expectations[name], str)
            or not runtime_expectations[name].strip()
        ):
            raise CellError(f"Frozen runtime expectation must be text: {name}")

    points = doc.get("points")
    if not isinstance(points, dict) or set(points) != set(POINTS):
        raise CellError(f"Manifest points must be exactly {sorted(POINTS)}")
    for point, contract in POINTS.items():
        item = points[point]
        if not isinstance(item, dict):
            raise CellError(f"Manifest load geometry is invalid for {point}")
        requests = item.get("requests")
        if (
            type(item.get("concurrency")) is not int
            or item.get("concurrency") != contract["concurrency"]
            or type(item.get("prompt_tokens")) is not int
            or item.get("prompt_tokens") != contract["prompt_tokens"]
            or not isinstance(requests, list)
            or len(requests) != contract["concurrency"]
        ):
            raise CellError(f"Manifest load geometry is invalid for {point}")
        by_slot: dict[int, list[int]] = {}
        for request in requests:
            if not isinstance(request, dict):
                raise CellError(f"Invalid request in {point}")
            slot = request.get("slot")
            ids = request.get("token_ids")
            if type(slot) is not int or slot in by_slot or not isinstance(ids, list):
                raise CellError(f"Invalid or duplicate slot in {point}")
            if slot < 0 or slot >= contract["concurrency"]:
                raise CellError(f"Out-of-range slot in {point}")
            if (
                len(ids) != contract["prompt_tokens"]
                or len(ids) > MAX_IDS_PER_PROMPT
                or any(type(token) is not int or token < 0 for token in ids)
            ):
                raise CellError(
                    f"Token IDs have wrong type/count in {point} slot {slot}"
                )
            expected_hash = request.get("token_ids_sha256")
            actual_hash = token_ids_sha256(ids)
            if expected_hash != actual_hash:
                raise CellError(f"Token ID hash mismatch in {point} slot {slot}")
            answer = request.get("expected_answer")
            if not isinstance(answer, str) or not answer.strip():
                raise CellError(f"Missing expected answer in {point} slot {slot}")
            by_slot[slot] = ids
        if set(by_slot) != set(range(contract["concurrency"])):
            raise CellError(f"Slots are not contiguous in {point}")
        ordered = [by_slot[index] for index in range(contract["concurrency"])]
        for left in range(len(ordered)):
            for right in range(left + 1, len(ordered)):
                common = 0
                for a, b in zip(ordered[left], ordered[right]):
                    if a != b:
                        break
                    common += 1
                if common:
                    raise CellError(
                        f"Prompts share a {common}-token prefix in {point}"
                    )
    return doc, raw


def make_schedule() -> list[dict[str, Any]]:
    """Counterbalance k order within each of the four fixed load points."""
    order = (
        ("c2_16k", (2, 3)),
        ("c2_80k", (3, 2)),
        ("c4_16k", (3, 2)),
        ("c4_32k", (2, 3)),
    )
    return [
        {
            "point": point,
            "k": k,
            "concurrency": POINTS[point]["concurrency"],
            "prompt_tokens_per_request": POINTS[point]["prompt_tokens"],
            "max_completion_tokens": MAX_OUTPUT_TOKENS,
            "fresh_engine_required": True,
        }
        for point, ks in order
        for k in ks
    ]


def validate_runtime_proof(
    proof: dict[str, Any], manifest: dict[str, Any], point: str, k: int
) -> None:
    if proof.get("num_speculative_tokens") != k:
        raise CellError("Runtime proof k does not match requested k")
    runtime_expectations = manifest["runtime_expectations"]
    for field in (
        "vllm_version",
        "exllamav3_revision",
        "installed_exl3_source_sha256",
        "r0_source_commit",
        "driver_version",
        "cuda_version",
        "torch_version",
    ):
        expected = (
            manifest["provenance"][field]
            if (field.endswith("_sha256") or field == "r0_source_commit")
            and field in manifest["provenance"]
            else runtime_expectations.get(field)
        )
        if proof.get(field) != expected:
            raise CellError(f"Runtime proof {field} does not match frozen expectation")
    if not isinstance(proof.get("engine_id"), str) or not proof["engine_id"].strip():
        raise CellError("Runtime proof must identify a fresh engine_id")
    if proof.get("model") != manifest.get("model"):
        raise CellError("Runtime proof model does not match prompt manifest")
    for field in (
        "model_pack_sha256",
        "model_revision_sha256",
        "tokenizer_sha256",
    ):
        if proof.get(field) != manifest["provenance"][field]:
            raise CellError(f"Runtime proof {field} does not match manifest")
    if not PROMPT_SHA_RE.fullmatch(proof["installed_exl3_source_sha256"]):
        raise CellError("Malformed installed EXL3 source SHA-256")
    if not isinstance(proof["r0_source_commit"], str) or not re.fullmatch(
        r"[0-9a-f]{40}", proof["r0_source_commit"]
    ):
        raise CellError("Malformed R0 source commit")
    config = proof.get("config")
    if not isinstance(config, dict):
        raise CellError("Runtime proof has no observed config")
    for name, expected in FROZEN_CONFIG.items():
        if config.get(name) != expected:
            raise CellError(f"Observed runtime config mismatch: {name}")
    effective = config.get("effective_max_num_batched_tokens")
    if type(effective) is not int or effective <= 0:
        raise CellError("Runtime proof needs positive effective auto token budget")
    if effective != runtime_expectations["effective_max_num_batched_tokens"]:
        raise CellError("Effective auto token budget differs from frozen expectation")
    evidence = proof.get("effective_budget_evidence")
    if not isinstance(evidence, dict):
        raise CellError("Runtime proof lacks source-qualified budget evidence")
    if (
        evidence.get("kind") != "vllm_0.29.0_enginecore_source_inference"
        or evidence.get("scheduler_override_absent") is not True
        or evidence.get("scheduled_token_cli_arg_absent") is not True
        or evidence.get("batched_token_cli_arg_absent") is not True
        or evidence.get("engine_args_default_none_verified") is not True
        or evidence.get("source_default_relation")
        != "max_num_scheduled_tokens=None -> max_num_batched_tokens"
        or evidence.get("observed_enginecore_max_num_scheduled_tokens") != effective
        or evidence.get("log_line_count") != 1
    ):
        raise CellError("Runtime proof does not establish the auto budget inference")
    if evidence.get("log_line_source") != "EngineCore":
        raise CellError("Budget evidence must come from EngineCore, not APIServer")
    log_line = evidence.get("scheduled_tokens_log_line")
    if not isinstance(log_line, str) or not ENGINECORE_PREFIX.search(log_line):
        raise CellError("Budget evidence lacks the original EngineCore log line")
    match = SCHEDULED_TOKEN_LINE.search(log_line)
    if not match or int(match.group(1).replace(",", "")) != effective:
        raise CellError("EngineCore warning text does not support the inferred budget")
    kv_tokens = proof.get("kv_pool_tokens")
    required = math.ceil(
        POINTS[point]["concurrency"]
        * (POINTS[point]["prompt_tokens"] + MAX_OUTPUT_TOKENS)
        * 1.2
    )
    if type(kv_tokens) is not int or kv_tokens < required:
        raise CellError(f"KV pool below 20% headroom gate: need {required} tokens")
    if evidence.get("observed_enginecore_kv_pool_tokens") != kv_tokens:
        raise CellError("Runtime proof KV pool does not match EngineCore log evidence")
    kv_lines = evidence.get("kv_pool_log_lines")
    if not isinstance(kv_lines, list) or not kv_lines:
        raise CellError("Runtime proof lacks original EngineCore KV-pool log line")
    for line in kv_lines:
        if not isinstance(line, str) or not ENGINECORE_PREFIX.search(line):
            raise CellError("KV-pool evidence must come from EngineCore")
        match = KV_POOL_LINE.search(line)
        if not match or int(match.group(1).replace(",", "")) != kv_tokens:
            raise CellError("EngineCore KV-pool log text does not support the capacity value")


def _request_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 30):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def scrape_metrics(base: str) -> dict[str, Any]:
    raw = _request_json(f"{base}/metrics", timeout=15).decode("utf-8", "replace")
    values: dict[str, float] = {}
    cache_usage: dict[str, float] = {}
    per_position: dict[str, float] = {}
    wanted = {**COUNTERS, **GAUGES}
    for line in raw.splitlines():
        if not line or line.startswith("#") or " " not in line:
            continue
        sample, value_text = line.rsplit(None, 1)
        try:
            value = float(value_text)
        except ValueError:
            continue
        if not math.isfinite(value):
            raise CellError(f"Non-finite Prometheus sample: {sample}")
        for key, metric in wanted.items():
            if sample == metric or sample.startswith(metric + "{"):
                values[key] = values.get(key, 0.0) + value
        if "accepted_tokens_per_pos_total{" in sample:
            label = re.search(r'position="?([^",}]+)', sample)
            if label:
                key = label.group(1)
                per_position[key] = per_position.get(key, 0.0) + value
        if "cache_usage_perc" in sample:
            cache_usage[sample] = value
    result: dict[str, Any] = {key: values.get(key) for key in wanted}
    result["cache_usage"] = cache_usage
    result["accepted_per_position"] = per_position
    result["observed_monotonic_ns"] = time.monotonic_ns()
    return result


def _wait_idle(base: str, timeout_s: float = 120) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    latest: dict[str, Any] = {}
    while time.monotonic() < deadline:
        latest = scrape_metrics(base)
        if latest.get("running") is None or latest.get("waiting") is None:
            raise CellError("Engine metrics omit running/waiting request gauges")
        if latest["running"] == 0 and latest["waiting"] == 0:
            return latest
        time.sleep(0.25)
    raise CellError("Engine did not return to idle before timeout")


def _stream_request(
    base: str,
    model: str,
    request_spec: dict[str, Any],
    slot: int,
    barrier: threading.Barrier,
    released: dict[str, int],
) -> dict[str, Any]:
    barrier.wait(timeout=60)
    dispatched = time.monotonic_ns()
    payload = {
        "model": model,
        "prompt": request_spec["token_ids"],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0,
        "seed": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request_payload = json.dumps(payload, separators=(",", ":")).encode()
    req = urllib.request.Request(
        f"{base}/v1/completions",
        data=request_payload,
        headers={"Content-Type": "application/json"},
    )
    text_parts: list[str] = []
    first_token_ns: int | None = None
    usage: dict[str, Any] | None = None
    finish_reason = None
    with urllib.request.urlopen(req, timeout=7200) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if event.get("usage") is not None:
                usage = event["usage"]
            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("text") or ""
            if delta:
                if first_token_ns is None:
                    first_token_ns = time.monotonic_ns()
                text_parts.append(delta)
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
    finished = time.monotonic_ns()
    text = "".join(text_parts)
    return {
        "slot": slot,
        "request_payload_sha256": sha256_bytes(request_payload),
        "dispatch_monotonic_ns": dispatched,
        "first_token_monotonic_ns": first_token_ns,
        "finish_monotonic_ns": finished,
        "usage": usage,
        "finish_reason": finish_reason,
        "text": text,
        "text_sha256": sha256_bytes(text.encode("utf-8")),
        "answer_match": request_spec["expected_answer"] in text,
    }


def _run_wave(
    base: str,
    model: str,
    requests: list[dict[str, Any]],
    wave_index: int,
    monitor_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    idle_before = _wait_idle(base)
    before = scrape_metrics(base)
    if any(before.get(key) is None for key in COUNTERS):
        raise CellError("Metrics omit one or more required cell counters")
    concurrency = len(requests)
    release: dict[str, int] = {}
    barrier = threading.Barrier(
        concurrency, action=lambda: release.update({"ns": time.monotonic_ns()})
    )
    stop_monitor = threading.Event()
    monitor_failures: list[str] = []
    request_failures: list[str] = []

    def monitor() -> None:
        while not stop_monitor.wait(0.25):
            try:
                sample = scrape_metrics(base)
                sample["wave"] = wave_index
                monitor_samples.append(sample)
            except Exception as exc:
                monitor_failures.append(str(exc))

    thread = threading.Thread(target=monitor, name="r0-wave-metrics", daemon=True)
    thread.start()
    results: list[dict[str, Any]] = []
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(_stream_request, base, model, req, i, barrier, release)
                for i, req in enumerate(requests)
            ]
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    request_failures.append(f"{type(exc).__name__}: {exc}")
    finally:
        stop_monitor.set()
        thread.join(timeout=20)
    after = scrape_metrics(base)
    idle_after = _wait_idle(base)
    if any(after.get(key) is None for key in COUNTERS):
        raise CellError("Metrics omit one or more required cell counters")
    if monitor_failures:
        raise CellError(f"Metrics poll failed during wave: {monitor_failures[0]}")
    results.sort(key=lambda result: result["slot"])
    if request_failures:
        return {
            "wave": wave_index,
            "status": "FAILED_REQUEST",
            "errors": request_failures,
            "metrics_before": before,
            "metrics_after": after,
            "idle_before": idle_before,
            "idle_after": idle_after,
            "requests": results,
            "samples": monitor_samples,
        }
    dispatch_times = [result["dispatch_monotonic_ns"] for result in results]
    skew_ms = (max(dispatch_times) - min(dispatch_times)) / 1_000_000
    if skew_ms > 50:
        return {
            "wave": wave_index,
            "status": "DISCARDED_DISPATCH_SKEW",
            "dispatch_skew_ms": skew_ms,
            "requests": results,
        }
    if len(results) != concurrency:
        raise CellError("Wave did not return all request results")
    release_ns = release.get("ns")
    if release_ns is None:
        raise CellError("Barrier release timestamp missing")
    finish_ns = max(result["finish_monotonic_ns"] for result in results)
    used_tokens = sum(
        (result.get("usage") or {}).get("completion_tokens", 0)
        for result in results
    )
    span_s = (finish_ns - release_ns) / 1_000_000_000
    overlap_start = max(
        result["first_token_monotonic_ns"] or result["finish_monotonic_ns"]
        for result in results
    )
    overlap_end = min(result["finish_monotonic_ns"] for result in results)
    deltas = {
        key: after[key] - before[key]
        for key in COUNTERS
    }
    for key, delta in deltas.items():
        if delta < 0:
            raise CellError(f"Metric counter reset during wave: {key}")
    per_position_before = before.get("accepted_per_position", {})
    per_position_after = after.get("accepted_per_position", {})
    per_position_delta = {
        key: per_position_after[key] - per_position_before[key]
        for key in sorted(set(per_position_before) & set(per_position_after))
    }
    return {
        "wave": wave_index,
        "status": "VALID",
        "dispatch_skew_ms": skew_ms,
        "barrier_release_monotonic_ns": release_ns,
        "idle_before": idle_before,
        "metrics_before": before,
        "metrics_after": after,
        "idle_after": idle_after,
        "counter_deltas": deltas,
        "accepted_per_position_delta": per_position_delta,
        "requests": results,
        "aggregate_output_tokens": used_tokens,
        "aggregate_tok_s": used_tokens / span_s if span_s > 0 else None,
        "release_to_last_finish_s": span_s,
        "decode_overlap_s": max(0, overlap_end - overlap_start) / 1_000_000_000,
    }


def _percentile95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _summarize(waves: list[dict[str, Any]], k: int, concurrency: int) -> dict[str, Any]:
    requests = [request for wave in waves for request in wave["requests"]]
    ttft = [
        (r["first_token_monotonic_ns"] - r["dispatch_monotonic_ns"]) / 1e9
        for r in requests
        if r["first_token_monotonic_ns"] is not None
    ]
    e2e = [
        (r["finish_monotonic_ns"] - r["dispatch_monotonic_ns"]) / 1e9
        for r in requests
    ]
    tpot = [
        (r["finish_monotonic_ns"] - r["first_token_monotonic_ns"])
        / 1_000_000
        / (r["usage"]["completion_tokens"] - 1)
        for r in requests
        if r["first_token_monotonic_ns"] is not None
        and r.get("usage")
        and r["usage"].get("completion_tokens", 0) > 1
    ]
    deltas = {key: sum(w["counter_deltas"][key] for w in waves) for key in COUNTERS}
    drafts = deltas["drafts"]
    draft_width = deltas["draft_tokens"] / drafts if drafts else None
    samples = [sample for wave in waves for sample in wave["samples"]]
    cache_values = [
        value
        for sample in samples
        for value in (sample.get("cache_usage") or {}).values()
    ]
    simultaneous_running_s = 0.0
    for wave in waves:
        ordered_samples = sorted(
            wave["samples"], key=lambda sample: sample["observed_monotonic_ns"]
        )
        for current, following in zip(ordered_samples, ordered_samples[1:]):
            if (current.get("running") or 0) >= concurrency:
                simultaneous_running_s += (
                    following["observed_monotonic_ns"]
                    - current["observed_monotonic_ns"]
                ) / 1_000_000_000
    return {
        "request_count": len(requests),
        "wave_count": len(waves),
        "median_aggregate_tok_s": statistics.median(
            w["aggregate_tok_s"] for w in waves
        ),
        "p95_ttft_s": _percentile95(ttft),
        "median_ttft_s": statistics.median(ttft) if ttft else None,
        "p95_tpot_ms": _percentile95(tpot),
        "median_tpot_ms": statistics.median(tpot) if tpot else None,
        "p95_end_to_end_s": _percentile95(e2e),
        "median_end_to_end_s": statistics.median(e2e) if e2e else None,
        "counter_deltas": deltas,
        "draft_tokens_per_draft": draft_width,
        "accepted_per_draft": deltas["accepted"] / drafts if drafts else None,
        "accepted_per_output_token": (
            deltas["accepted"] / sum(r["usage"]["completion_tokens"] for r in requests)
        ),
        "requested_k": k,
        "preemptions": deltas["preemptions"],
        "peak_running_observed": max(
            (
                sample.get("running") or 0
                for wave in waves
                for sample in wave["samples"]
            ),
            default=0,
        ),
        "peak_waiting_observed": max(
            (
                sample.get("waiting") or 0
                for wave in waves
                for sample in wave["samples"]
            ),
            default=0,
        ),
        "all_concurrent_running_sample_count": sum(
            (sample.get("running") or 0) >= concurrency
            for wave in waves
            for sample in wave["samples"]
        ),
        "positive_decode_overlap_wave_count": sum(
            wave["decode_overlap_s"] > 0 for wave in waves
        ),
        "median_decode_overlap_s": statistics.median(
            [wave["decode_overlap_s"] for wave in waves if wave["decode_overlap_s"] > 0]
        )
        if any(wave["decode_overlap_s"] > 0 for wave in waves)
        else None,
        "estimated_all_concurrent_running_s": simultaneous_running_s,
        "peak_cache_usage_perc_observed": max(cache_values) if cache_values else None,
        "cache_usage_metric_observed": bool(cache_values),
        "accepted_per_position_delta": {
            position: sum(
                wave["accepted_per_position_delta"].get(position, 0.0)
                for wave in waves
            )
            for position in sorted(
                {
                    position
                    for wave in waves
                    for position in wave["accepted_per_position_delta"]
                }
            )
        },
    }


def run_cell(
    manifest: dict[str, Any],
    manifest_raw: bytes,
    proof: dict[str, Any],
    point: str,
    k: int,
    out_dir: Path,
    base: str,
) -> dict[str, Any]:
    if point not in POINTS or k not in (2, 3):
        raise CellError("Point/k is outside the frozen matrix")
    validate_runtime_proof(proof, manifest, point, k)
    if out_dir.exists():
        raise CellError(f"Output directory already exists: {out_dir}")
    parsed_base = urlsplit(base)
    if (
        parsed_base.scheme != "http"
        or parsed_base.hostname != "127.0.0.1"
        or parsed_base.port != PORT
        or parsed_base.path
        or parsed_base.query
        or parsed_base.fragment
        or parsed_base.username
        or parsed_base.password
    ):
        raise CellError("Execution is restricted to the local port 8002")
    base = base.rstrip("/")
    _request_json(f"{base}/health", timeout=10)
    model_response = json.loads(_request_json(f"{base}/v1/models", timeout=15))
    ids = [item.get("id") for item in model_response.get("data", [])]
    if manifest.get("model") not in ids:
        raise CellError("Healthy API does not expose the manifest model ID")

    point_spec = manifest["points"][point]
    requests = sorted(point_spec["requests"], key=lambda item: item["slot"])
    if not out_dir.parent.is_dir():
        raise CellError(f"Output parent directory must already exist: {out_dir.parent}")
    out_dir.mkdir(parents=True, mode=0o700)
    (out_dir / "prompt_manifest.json").write_bytes(manifest_raw)
    (out_dir / "runtime_proof.json").write_text(
        json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result: dict[str, Any] = {
        "status": "RUNNING",
        "point": point,
        "k": k,
        "engine_id": proof["engine_id"],
        "manifest_sha256": sha256_bytes(manifest_raw),
        "runtime_proof": proof,
        "fixed_config": FROZEN_CONFIG,
        "effective_max_num_batched_tokens": proof["config"][
            "effective_max_num_batched_tokens"
        ],
        "metric_samples": [],
        "discarded_waves": [],
        "warmup": None,
        "waves": [],
    }
    measured: list[dict[str, Any]] = []
    try:
        warm_samples: list[dict[str, Any]] = []
        warmup = _run_wave(base, manifest["model"], requests, 0, warm_samples)
        warmup["samples"] = warm_samples
        _validate_wave(warmup, requests, k)
        result["warmup"] = warmup
        accepted = 0
        attempts = 0
        while accepted < MEASURED_WAVES:
            attempts += 1
            if attempts > MEASURED_WAVES + MAX_SKEW_REPEATS:
                raise CellError("Exceeded dispatch-skew repeat allowance")
            samples: list[dict[str, Any]] = []
            wave = _run_wave(
                base, manifest["model"], requests, accepted + 1, samples
            )
            wave["samples"] = samples
            result["metric_samples"].extend(samples)
            if wave["status"] == "DISCARDED_DISPATCH_SKEW":
                result["discarded_waves"].append(wave)
                continue
            result["waves"].append(wave)
            _validate_wave(wave, requests, k)
            measured.append(wave)
            accepted += 1
        result["summary"] = _summarize(measured, k, len(requests))
        summary = result["summary"]
        drafts = summary["counter_deltas"]["drafts"]
        width = summary["draft_tokens_per_draft"]
        if (
            drafts <= 0
            or width is None
            or not (k - 1 <= width <= k)
            or summary["preemptions"] != 0
        ):
            raise CellError("MTP counter width or preemption gate failed")
        if not summary["cache_usage_metric_observed"]:
            raise CellError(
                "No KV/cache usage metric was observed during measured waves"
            )
        result["status"] = "PASS"
    except Exception as exc:
        result["status"] = "FAIL"
        result["error"] = f"{type(exc).__name__}: {exc}"
    (out_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _validate_wave(
    wave: dict[str, Any], requests: list[dict[str, Any]], k: int
) -> None:
    if wave.get("status") != "VALID":
        raise CellError("Wave was not valid")
    for measured, source in zip(wave["requests"], requests):
        usage = measured.get("usage") or {}
        if usage.get("prompt_tokens") != len(source["token_ids"]):
            raise CellError(f"Wrong API prompt token count for slot {source['slot']}")
        if usage.get("completion_tokens") != MAX_OUTPUT_TOKENS:
            raise CellError(
                f"Wrong API completion token count for slot {source['slot']}"
            )
        if not measured.get("answer_match"):
            raise CellError(f"Known target answer mismatch in slot {source['slot']}")
    deltas = wave["counter_deltas"]
    if deltas["preemptions"] != 0:
        raise CellError("Nonzero preemptions in wave")
    if deltas["drafts"] <= 0:
        raise CellError("No MTP drafts observed in wave")
    width = deltas["draft_tokens"] / deltas["drafts"]
    if not (k - 1 <= width <= k):
        raise CellError(
            f"Observed MTP draft width {width:g} is inconsistent with k={k}"
        )
    if wave["decode_overlap_s"] <= 0:
        raise CellError("Wave has no positive common decode-overlap window")
    if not any(
        (sample.get("running") or 0) >= len(requests)
        for sample in wave.get("samples", [])
    ):
        raise CellError("Metrics never observed all requests running simultaneously")


def _dry_run(manifest: dict[str, Any], raw: bytes) -> dict[str, Any]:
    return {
        "mode": "DRY_RUN",
        "network_used": False,
        "gpu_used": False,
        "manifest_sha256": sha256_bytes(raw),
        "provenance": manifest["provenance"],
        "fixed_config": FROZEN_CONFIG,
        "schedule": make_schedule(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-r0-source-commit")
    parser.add_argument(
        "--dry-run", action="store_true", help="validate and print plan"
    )
    parser.add_argument(
        "--execute", action="store_true", help="send local API requests"
    )
    parser.add_argument("--point", choices=sorted(POINTS))
    parser.add_argument("--k", type=int, choices=(2, 3))
    parser.add_argument("--runtime-proof", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--base-url", default=f"http://127.0.0.1:{PORT}")
    args = parser.parse_args()
    if args.dry_run and args.execute:
        parser.error("choose either --dry-run or --execute")
    if not args.dry_run and not args.execute:
        args.dry_run = True
    try:
        manifest, raw = load_prompt_manifest(args.manifest)
        if args.expected_r0_source_commit is not None:
            if not re.fullmatch(r"[0-9a-f]{40}", args.expected_r0_source_commit):
                raise CellError("Expected R0 source commit must be 40 lowercase hex chars")
            if manifest["provenance"]["r0_source_commit"] != args.expected_r0_source_commit:
                raise CellError("Manifest r0_source_commit differs from expected R0 source commit")
        if args.dry_run:
            print(json.dumps(_dry_run(manifest, raw), indent=2, sort_keys=True))
            return 0
        if args.expected_r0_source_commit is None:
            raise CellError("Execution requires --expected-r0-source-commit")
        if not all((args.point, args.k, args.runtime_proof, args.out_dir)):
            parser.error(
                "--execute requires --point, --k, --runtime-proof, and --out-dir"
            )
        proof, _ = _load_json(args.runtime_proof, 1024 * 1024)
        if not isinstance(proof, dict):
            raise CellError("Runtime proof must be a JSON object")
        result = run_cell(
            manifest, raw, proof, args.point, args.k, args.out_dir, args.base_url
        )
        print(
            json.dumps(
                {k: v for k, v in result.items() if k != "waves"}, sort_keys=True
            )
        )
        return 0 if result["status"] == "PASS" else 1
    except (OSError, ValueError, CellError, urllib.error.URLError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
