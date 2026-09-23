"""Static safety contract checks for the isolated APC lifecycle runner."""

from pathlib import Path
import unittest

RUNNER = Path(__file__).with_name("r0_run_prefix_cache_probe.sh")


class PrefixCacheRunnerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = RUNNER.read_text(encoding="utf-8")

    def test_fixed_probe_configuration(self):
        for item in (
            "readonly HOST=127.0.0.1 PORT=8002 GPU_MEM_UTIL=0.92",
            "0) MAX_MODEL_LEN=246000; CAPACITY_PROFILE=production-envelope",
            "1) MAX_MODEL_LEN=32768; CAPACITY_PROFILE=short-context-mechanism-only",
            "readonly NUM_SPEC_TOKENS=3 PROMPT_CASE_REL=ctx16000/turn_04_ask_d_e.json",
            "PREFIX_CACHING=1 NUM_SPEC_TOKENS=3",
            'env MODEL_DIR="$MODEL_DIR" GPU_MEM_UTIL="$GPU_MEM_UTIL"',
            "MAX_NUM_BATCHED_TOKENS=",
            "--min-prompt-tokens 15000",
        ):
            with self.subTest(item=item):
                self.assertIn(item, self.source)

    def test_provenance_resource_gates_and_owned_cleanup(self):
        for item in (
            '[[ "$ACTUAL_SHA" == "$EXPECTED_SHA" ]]',
            'git -C "$R0_REPO" status --porcelain=v1',
            '[[ "$REPO_PLUGIN_SHA" == "$INSTALLED_PLUGIN_SHA" ]]',
            'installed_source_fingerprint "$OUT/installed_source_before.json"',
            "port_listener_count)==0",
            "GPU already has compute processes",
            "setsid group does not match PID",
            'kill -TERM -- "-$ACTIVE_PGID"',
            'kill -KILL -- "-$ACTIVE_PGID"',
            'installed_source_fingerprint "$OUT/installed_source_after.json"',
            "Xid count changed by shutdown",
        ):
            with self.subTest(item=item):
                self.assertIn(item, self.source)

    def test_probe_artifacts_and_no_broad_or_remote_kills(self):
        for item in ('"$OUT/probe.json"', '"$OUT/probe.stdout.log"',
                     '"$OUT/metrics_before.prom"', '"$OUT/metrics_after_probe.prom"'):
            with self.subTest(item=item):
                self.assertIn(item, self.source)
        for forbidden in ("pkill", "killall", "ssh "):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, self.source)


if __name__ == "__main__":
    unittest.main()
