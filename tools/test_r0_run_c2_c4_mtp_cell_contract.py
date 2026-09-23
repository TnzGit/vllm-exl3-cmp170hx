"""Static, CPU-only contract checks for the owned one-cell shell runner."""

from pathlib import Path
import unittest


SCRIPT = Path(__file__).with_name("r0_run_c2_c4_mtp_cell.sh")


class OwnedCellRunnerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_one_cell_arguments_and_frozen_envelope_are_explicit(self):
        for text in (
            "POINT=\"$1\" K=\"$2\"",
            "MAX_MODEL_LEN=240000",
            "MAX_NUM_SEQS=4",
            "GPU_MEM_UTIL=0.92",
            "MAX_NUM_BATCHED_TOKENS=",
            "VLLM_EXL3_COOP=1",
            "VLLM_EXL3_NGRAM_TABLE=disk",
            "NUM_SPEC_TOKENS=\"$K\"",
            "PORT=8002",
        ):
            self.assertIn(text, self.text)
        self.assertNotIn("246000", self.text)
        self.assertIn('args[args.index("--max-model-len")+1] != "240000"', self.text)
        self.assertIn('"MAX_MODEL_LEN":"240000"', self.text)
        helper = SCRIPT.with_name("r0_c2_c4_mtp_matched_load.py").read_text(encoding="utf-8")
        self.assertIn('"max_model_len": 240_000', helper)

    def test_proof_gate_precedes_first_helper_execution(self):
        proof = self.text.index('"$OUT/runtime_proof_build.log" 2>&1 <<\'PY\'')
        execute = self.text.index('"$HELPER" --manifest "$MANIFEST" --expected-r0-source-commit "$EXPECTED_SHA" --execute')
        self.assertLess(proof, execute)
        helper = SCRIPT.with_name("r0_c2_c4_mtp_matched_load.py").read_text(encoding="utf-8")
        self.assertIn("parse_enginecore_startup_evidence", helper)
        self.assertIn("source_default_relation", helper)
        self.assertIn("EngineCore", self.text)

    def test_process_ownership_and_cleanup_are_narrow(self):
        self.assertIn("setsid bash \"$LAUNCHER\"", self.text)
        self.assertIn('kill -TERM -- "-$ACTIVE_PGID"', self.text)
        self.assertIn('kill -KILL -- "-$ACTIVE_PGID"', self.text)
        self.assertIn("port_listener_count", self.text)
        self.assertIn("gpu_compute_pids", self.text)
        self.assertIn("xid_count", self.text)
        self.assertNotIn("pkill", self.text)
        self.assertNotIn("ssh ", self.text)

    def test_manifest_r0_commit_is_matched_to_expected_checkout_commit(self):
        cache_helper = SCRIPT.with_name("r0_model_identity_cache.py").read_text(encoding="utf-8")
        self.assertIn('manifest["provenance"]["r0_source_commit"] != expected_commit', cache_helper)
        self.assertIn('--expected-r0-source-commit "$EXPECTED_SHA"', self.text)
        self.assertIn('"r0_source_commit":r0_commit', self.text)
        self.assertNotIn("r0_source_sha256", self.text)

    def test_model_identity_cache_preserves_per_cell_hash_evidence(self):
        self.assertIn('IDENTITY_CACHE="$R0_ROOT/cache/r0_c2_c4_model_identity.json"', self.text)
        self.assertIn('IDENTITY_CACHE_HELPER="$SCRIPT_DIR/r0_model_identity_cache.py"', self.text)
        self.assertIn('--output "$OUT/input_hashes.json"', self.text)
        self.assertIn('"$VENV/bin/python" "$IDENTITY_CACHE_HELPER"', self.text)
        self.assertTrue(SCRIPT.with_name("r0_model_identity_cache.py").is_file())

    def test_dry_run_is_explicit_and_no_refusal_tree_is_referenced(self):
        self.assertIn("if (( DRY_RUN )); then", self.text)
        self.assertIn("--dry-run", self.text)
        self.assertNotIn("refusal-edit", self.text)

    def test_c4_32k_is_rejected_before_runtime_environment_checks(self):
        blocked = self.text.index("c4_32k) echo 'BLOCKED:")
        required_env = self.text.index('EXPECTED_SHA="${EXPECTED_SHA:?')
        self.assertLess(blocked, required_env)
        self.assertNotIn("c4_32k; K:", self.text)

    def test_remote_runner_does_not_require_ripgrep(self):
        self.assertIn("for cmd in nvidia-smi curl setsid ss journalctl sha256sum git grep", self.text)
        self.assertIn("grep -nE -- '--max-num-scheduled-tokens|MAX_NUM_SCHEDULED_TOKENS'", self.text)
        self.assertNotIn("command -v rg", self.text)


if __name__ == "__main__":
    unittest.main()
