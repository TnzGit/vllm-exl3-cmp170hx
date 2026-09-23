#!/usr/bin/env python3
"""Focused synthetic tests for the serial prefix-cache probe."""

import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
