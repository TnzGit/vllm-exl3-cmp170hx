#!/usr/bin/env python3
"""Focused synthetic tests for the serial prefix-cache probe."""

import unittest
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import r0_prefix_cache_probe as probe


class PrefixCacheMetricsPropagationTests(unittest.TestCase):
    def test_waits_for_delayed_query_and_hit_counters(self) -> None:
        stale_metrics = {
            "queries": 0.0,
            "hits": 0.0,
            "running": 0.0,
            "waiting": 0.0,
        }
        updated_metrics = {
            "queries": 2048.0,
            "hits": 1984.0,
            "running": 0.0,
            "waiting": 0.0,
        }

        with (
            patch.object(
                probe,
                "_fetch_metrics",
                side_effect=[stale_metrics, stale_metrics, updated_metrics],
            ) as fetch_metrics,
            patch.object(probe.time, "sleep"),
        ):
            result = probe._wait_for_prefix_cache_metrics(
                "http://127.0.0.1:8002",
                timeout=1.0,
                wait_seconds=10.0,
                before=stale_metrics,
                require_hit=True,
                where="after request 2",
            )

        self.assertEqual(result, updated_metrics)
        self.assertEqual(fetch_metrics.call_count, 3)

    def test_times_out_when_required_hit_counter_never_propagates(self) -> None:
        metrics = {
            "queries": 2048.0,
            "hits": 0.0,
            "running": 0.0,
            "waiting": 0.0,
        }
        with (
            patch.object(probe, "_fetch_metrics", return_value=metrics),
            patch.object(probe.time, "monotonic", side_effect=[0.0, 2.0]),
            patch.object(probe.time, "sleep"),
        ):
            with self.assertRaisesRegex(
                probe.ProbeFailure,
                r"metrics did not propagate hit counter.*hits_delta=0",
            ):
                probe._wait_for_prefix_cache_metrics(
                    "http://127.0.0.1:8002",
                    timeout=1.0,
                    wait_seconds=1.0,
                    before={
                        "queries": 0.0,
                        "hits": 0.0,
                        "running": 0.0,
                        "waiting": 0.0,
                    },
                    require_hit=True,
                    where="after request 2",
                )


class PrefixCacheFailureEvidenceTests(unittest.TestCase):
    def test_output_mismatch_failure_keeps_bounded_request_evidence(self) -> None:
        before = {"queries": 100.0, "hits": 0.0, "running": 0.0, "waiting": 0.0}
        after_first = {
            "queries": 2200.0,
            "hits": 0.0,
            "running": 0.0,
            "waiting": 0.0,
        }
        after_second = {
            "queries": 4300.0,
            "hits": 2048.0,
            "running": 0.0,
            "waiting": 0.0,
        }
        completions = [
            {
                "ttft_seconds": 0.1,
                "prompt_tokens_usage": 2048,
                "completion_tokens_usage": 8,
                "output_text_sha256": "hash-one",
                "output_text_characters": 300,
                "output_text_preview": "A" * probe.OUTPUT_PREVIEW_CHARS,
                "output_text_preview_truncated": True,
            },
            {
                "ttft_seconds": 0.1,
                "prompt_tokens_usage": 2048,
                "completion_tokens_usage": 8,
                "output_text_sha256": "hash-two",
                "output_text_characters": 3,
                "output_text_preview": "B!\n",
                "output_text_preview_truncated": False,
            },
        ]
        args = SimpleNamespace(
            port=8002, timeout=1.0, repeats=2, idle_timeout=0.0, metrics_timeout=1.0
        )
        response = MagicMock()
        response.__enter__.return_value = response
        response.__exit__.return_value = False

        with (
            patch.object(probe.urllib.request, "urlopen", return_value=response),
            patch.object(probe.json, "load", return_value={"data": [{"id": "model"}]}),
            patch.object(probe, "_resolve_prompt", return_value=[1] * 2048),
            patch.object(probe, "_fetch_metrics", side_effect=[before, after_first]),
            patch.object(probe, "_read_sse_completion", side_effect=completions),
            patch.object(probe, "_wait_for_idle"),
            patch.object(
                probe,
                "_wait_for_prefix_cache_metrics",
                side_effect=[after_first, after_second],
            ),
        ):
            with self.assertRaisesRegex(
                probe.ProbeFailure, "greedy output changed"
            ) as caught:
                probe.run_probe(args)

        report = caught.exception.partial_report
        self.assertIsNotNone(report)
        assert report is not None
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(len(report["requests"]), 2)
        self.assertEqual(
            [record["prefix_cache_queries_delta"] for record in report["requests"]],
            [2100.0, 2100.0],
        )
        self.assertEqual(
            [record["prefix_cache_hits_delta"] for record in report["requests"]],
            [0.0, 2048.0],
        )
        self.assertEqual(
            [record["output_text_sha256"] for record in report["requests"]],
            ["hash-one", "hash-two"],
        )
        self.assertEqual(
            report["requests"][0]["output_text_preview"],
            "A" * probe.OUTPUT_PREVIEW_CHARS,
        )
        self.assertTrue(report["requests"][0]["output_text_preview_truncated"])
        self.assertEqual(report["requests"][1]["output_text_preview"], "B!\n")

    def test_main_writes_failure_evidence_and_returns_nonzero(self) -> None:
        report = {
            "status": "FAIL",
            "error": "greedy output changed across identical prompt requests",
            "requests": [{"output_text_preview": "short raw output"}],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "probe.json"
            with (
                patch.object(
                    sys,
                    "argv",
                    ["r0_prefix_cache_probe.py", "--deterministic-prompt", "x", "--out", str(out_path)],
                ),
                patch.object(
                    probe,
                    "run_probe",
                    side_effect=probe.ProbeFailure("failed", partial_report=report),
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = probe.main()

            self.assertEqual(result, 1)
            self.assertEqual(json.loads(out_path.read_text(encoding="utf-8")), report)

    def test_main_refuses_to_overwrite_an_existing_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "probe.json"
            out_path.write_text("keep this report\n", encoding="utf-8")
            with (
                patch.object(
                    sys,
                    "argv",
                    ["r0_prefix_cache_probe.py", "--deterministic-prompt", "x", "--out", str(out_path)],
                ),
                patch.object(probe, "run_probe", return_value={"status": "PASS"}),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                result = probe.main()

            self.assertEqual(result, 2)
            self.assertEqual(out_path.read_text(encoding="utf-8"), "keep this report\n")


if __name__ == "__main__":
    unittest.main()
