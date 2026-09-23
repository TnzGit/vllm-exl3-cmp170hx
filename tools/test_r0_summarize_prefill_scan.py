"""Synthetic artifact tests for the fail-closed prefill scan analyzer."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys_path = str(Path(__file__).resolve().parent)
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

import r0_summarize_prefill_scan as analyzer


RUN_ID = "synthetic-run"
CONFIGS = ("auto", "1024", "4096")
CONTEXTS = (15533, 79533, 159533)


def put_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def put_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def make_cell(context: int, repeat_count: int, *, parity: bool = True,
              prompt_hash: str | None = None, scale: float = 1.0) -> dict[str, object]:
    prompt_hash = prompt_hash or f"{context:064x}"
    tokens = context
    count = 1 if repeat_count == 1 else repeat_count
    samples = []
    for repeat in range(1, count + 1):
        ttft = 0.2 * scale + repeat * 0.01
        wall = ttft + 1.02
        output_text = "<think>done</think>\n\nalpha-code, beta-code"
        if not parity and repeat > 1:
            output_text += f"\nforced continuation {repeat}"
        samples.append({
            "repeat": repeat,
            "prompt_tokens": tokens,
            "completion_tokens": 256,
            "ttft_s": ttft,
            "wall_s": wall,
            "prefill_tok_s": tokens / ttft,
            "tpot_s": (wall - ttft) / 255,
            "preemption_delta": 0,
            "output_text": output_text,
            "output_text_sha256": hashlib.sha256(output_text.encode()).hexdigest(),
        })
    return {
        "schema": 1,
        "benchmark": "non_kvmem_c1_mtp_k3_prefill_scan",
        "context_requested": context,
        "repeats_requested": count,
        "max_tokens_requested": 256,
        "prompt_source": "token_ids",
        "prompt_sha256": prompt_hash,
        "warmup": {"requested": True, "max_tokens": 16, "prompt_tokens": tokens,
                   "completion_tokens": 16, "preemption_delta": 0},
        "samples": samples,
        "output_parity": parity,
        "status": "VALID" if parity else "VALID_PERFORMANCE_WITH_OUTPUT_VARIATION",
    }


def make_scan(root: Path, *, mismatch: bool = False, parity: bool = True) -> Path:
    identity = (
        f"run_id={RUN_ID}\nexpected_sha={'a' * 40}\nrepo_head={'a' * 40}\n"
        f"installed_plugin_sha256={'b' * 64}\n"
        "contexts=15533:ctx16000 79533:ctx80000 159533:ctx160000\n"
        f"candidates={' '.join(CONFIGS)}\nrepeats=3\nmax_tokens=256\n"
        "cell_contract=one_warmup_then_repeats_measured_requests_per_context\n"
    )
    put_text(root / "run_identity.txt", identity)
    put_text(root / "run_exit.txt", f"run_id={RUN_ID}\nexit_status=0\n")
    put_text(root / "xid_before.txt", "xid_count_before=4\n")
    put_text(root / "xid_after.txt", "xid_count_after=4\nxid_delta=0\n")
    put_text(root / "port_before.txt", "host=127.0.0.1\nport=8002\nstate=CLOSED\n")
    put_text(root / "gpu_final.csv", "timestamp_utc=now\nCMP 170HX, 14 MiB\ncompute_processes:\n")
    for context in CONTEXTS:
        put_json(root / "prompt_cases" / f"prompt_{context}.json", {
            "query_text": "Return marker-A and marker-B codes in order",
            "target_facts": [
                {"marker": "marker-A", "code": "alpha-code"},
                {"marker": "marker-B", "code": "beta-code"},
            ],
        })
    fingerprint = {"sha256_python_sources": "a" * 64, "python_file_count": 2,
                   "roots": {"vllm": ["/venv/vllm"], "vllm_exl3": ["/venv/exl3"]},
                   "versions": {"vllm": "test", "vllm_exl3": "test"}}
    put_json(root / "installed_source_before.json", fingerprint)

    events = [f"2026-09-23T00:00:00Z run={RUN_ID} RUN_START"]
    for config_index, config in enumerate(CONFIGS):
        base = root / f"batched_tokens_{config}"
        put_text(base / "startup.txt", f"config={config}\nhealthy_utc=now\ncompleted_utc=now\n")
        put_text(base / "xid.txt", "xid_count_before=4\nxid_count_after=4\nxid_delta=0\n")
        put_text(base / "port_healthy.txt", "host=127.0.0.1\nport=8002\nstate=OPEN\n")
        put_text(base / "port_after.txt", "host=127.0.0.1\nport=8002\nstate=CLOSED\n")
        put_json(base / "installed_source.json", fingerprint)
        put_json(base / "installed_source_after.json", fingerprint)
        put_text(root / f"processes_batched_tokens_{config}_after.txt",
                 "timestamp_utc=now\n    PID    PPID    PGID COMMAND\n")
        events.extend((f"2026-09-23T00:00:00Z run={RUN_ID} CONFIG_START config={config}",))
        for context in CONTEXTS:
            prompt_hash = f"{context:064x}"
            if mismatch and config == "4096" and context == 79533:
                prompt_hash = "f" * 64
            parity_here = parity and not (config == "1024" and context == 15533)
            cell = make_cell(context, 3, parity=parity_here, prompt_hash=prompt_hash,
                             scale=1.0 + config_index * 0.1)
            put_json(base / f"cell_{context}.json", cell)
            put_text(base / f"cell_{context}.status.txt", "cell_exit_status=0\n")
            events.append(f"2026-09-23T00:00:00Z run={RUN_ID} CELL_DONE config={config} context={context}")
        sentinel = make_cell(15533, 1, prompt_hash=f"{15533:064x}")
        put_json(base / "cell_15533_sentinel.json", sentinel)
        events.extend((f"2026-09-23T00:00:00Z run={RUN_ID} SENTINEL_DONE config={config} context=15533",
                       f"2026-09-23T00:00:00Z run={RUN_ID} CONFIG_DONE config={config}"))
    events.extend((f"2026-09-23T00:00:00Z run={RUN_ID} RUN_COMPLETE xid_delta=0",
                   f"2026-09-23T00:00:00Z run={RUN_ID} RUN_EXIT status=0"))
    put_text(root / "events.log", "\n".join(events) + "\n")
    return root


def make_partial_4096_boot_capacity(root: Path) -> Path:
    run_events = (root / "events.log").read_text(encoding="utf-8").splitlines()
    run_events = [line for line in run_events
                  if "config=4096" not in line and "RUN_COMPLETE " not in line
                  and "RUN_EXIT status=0" not in line]
    run_events.extend((
        f"2026-09-23T00:00:00Z run={RUN_ID} CONFIG_START config=4096",
        f"2026-09-23T00:00:00Z run={RUN_ID} STARTUP_FAILED config=4096 pid=465305",
        f"2026-09-23T00:00:00Z run={RUN_ID} STOP_REQUEST config=batched_tokens_4096 pid=465305 pgid=465305",
        f"2026-09-23T00:00:00Z run={RUN_ID} STOPPED config=batched_tokens_4096 pid=465305 pgid=465305",
        f"2026-09-23T00:00:00Z run={RUN_ID} ERROR engine failed to become healthy for MAX_NUM_BATCHED_TOKENS=4096",
        f"2026-09-23T00:00:00Z run={RUN_ID} RUN_EXIT status=2",
    ))
    put_text(root / "events.log", "\n".join(run_events) + "\n")
    put_text(root / "run_exit.txt", f"run_id={RUN_ID}\nexit_status=2\n")
    (root / "xid_after.txt").unlink()

    base = root / "batched_tokens_4096"
    put_text(base / "startup.txt",
             "config=4096\nstarted_utc=now\nstartup_failed_utc=now\n"
             "server_exit_status=1\nserver_exit_utc=now\n")
    put_text(base / "server.log",
             "ValueError: To serve at least one request with the model's max seq len (246000), "
             "(7.0 GiB KV cache is needed, which is larger than the available KV cache memory "
             "(6.63 GiB). Based on the available memory, the estimated maximum model length is 232000.\n")
    keep = {"startup.txt", "server.log", "port_after.txt", "xid.txt"}
    for path in base.iterdir():
        if path.is_file() and path.name not in keep:
            path.unlink()
    return root


class PrefillSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "OUT"
        make_scan(self.root)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_valid_scan_summarizes_ranges_and_comparisons(self) -> None:
        report = analyzer.analyze(self.root)
        self.assertEqual(report["status"], "VALID_PREFILL_PERFORMANCE_ONLY")
        self.assertFalse(report["decode_semantic_qualification"])
        stats = report["summary"]["auto"]["15533"]["metrics"]["ttft_s"]
        self.assertAlmostEqual(stats["median"], 0.22)
        self.assertAlmostEqual(stats["min"], 0.21)
        self.assertAlmostEqual(stats["max"], 0.23)
        self.assertGreater(report["comparisons_vs_auto"]["1024"]["15533"]["ttft_s"]["change_pct_vs_auto"], 0)

    def test_first_correct_final_answer_allows_later_synthetic_turn(self) -> None:
        path = self.root / "batched_tokens_1024" / "cell_79533.json"
        cell = json.loads(path.read_text(encoding="utf-8"))
        for sample in cell["samples"]:
            sample["output_text"] += "\n<think>synthetic continuation</think>another turn"
            sample["output_text_sha256"] = hashlib.sha256(sample["output_text"].encode()).hexdigest()
        put_json(path, cell)
        report = analyzer.analyze(self.root)
        self.assertTrue(report["summary"]["1024"]["79533"]["visible_final_answer_correct"])

    def test_partial_4096_capacity_failure_is_explicit_and_has_no_metrics(self) -> None:
        make_partial_4096_boot_capacity(self.root)
        with self.assertRaisesRegex(analyzer.EvidenceError, "exit status"):
            analyzer.analyze(self.root)
        report = analyzer.analyze(self.root, classify_partial_4096_boot_capacity=True)
        self.assertEqual(report["status"], "VALID_PARTIAL_4096_BOOT_CAPACITY_NO_GO")
        self.assertEqual(report["completed_configs"], ["auto", "1024"])
        self.assertNotIn("4096", report["summary"])
        self.assertNotIn("4096", report["comparisons_vs_auto"])
        self.assertEqual(report["failed_config"]["available_kv_cache"], "6.63 GiB")
        self.assertIn("(246000)", report["failed_config"]["raw_log_reason"])

    def test_partial_classification_rejects_other_startup_errors_and_live_processes(self) -> None:
        make_partial_4096_boot_capacity(self.root)
        server_log = self.root / "batched_tokens_4096" / "server.log"
        server_log.write_text("ValueError: unrelated startup failure\n", encoding="utf-8")
        with self.assertRaisesRegex(analyzer.EvidenceError, "expected 246000-token"):
            analyzer.analyze(self.root, classify_partial_4096_boot_capacity=True)

        make_scan(self.root)
        make_partial_4096_boot_capacity(self.root)
        cleanup = self.root / "processes_batched_tokens_4096_after.txt"
        cleanup.write_text("timestamp_utc=now\n    PID    PPID    PGID COMMAND\n 123 1 123 server\n",
                           encoding="utf-8")
        with self.assertRaisesRegex(analyzer.EvidenceError, "processes remain"):
            analyzer.analyze(self.root, classify_partial_4096_boot_capacity=True)

    def test_output_variation_is_prefill_only_and_explicit(self) -> None:
        report = analyzer.analyze(self.root)
        cell = report["summary"]["1024"]["15533"]
        self.assertFalse(cell["output_parity"])
        self.assertEqual(cell["classification"], "prefill_only_output_variation")
        self.assertFalse(report["decode_semantic_qualification"])

    def test_incomplete_cell_and_cross_config_prompt_mismatch_fail_closed(self) -> None:
        (self.root / "batched_tokens_1024" / "cell_79533.json").unlink()
        with self.assertRaisesRegex(analyzer.EvidenceError, "missing required evidence"):
            analyzer.analyze(self.root)
        make_scan(self.root, mismatch=True)
        with self.assertRaisesRegex(analyzer.EvidenceError, "prompt identity/count differs"):
            analyzer.analyze(self.root)

    def test_preemption_is_rejected(self) -> None:
        path = self.root / "batched_tokens_auto" / "cell_15533.json"
        cell = json.loads(path.read_text(encoding="utf-8"))
        cell["samples"][0]["preemption_delta"] = 1
        put_json(path, cell)
        with self.assertRaisesRegex(analyzer.EvidenceError, "preemption detected"):
            analyzer.analyze(self.root)

    def test_cli_refuses_to_overwrite_output(self) -> None:
        target = Path(self.temp.name) / "report.json"
        target.write_text("preserve me", encoding="utf-8")
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
            rc = analyzer.main(["--dir", str(self.root), "--output", str(target)])
        self.assertEqual(rc, 2)
        self.assertEqual(target.read_text(encoding="utf-8"), "preserve me")
        self.assertIn('"status": "VALID_PREFILL_PERFORMANCE_ONLY"', stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
