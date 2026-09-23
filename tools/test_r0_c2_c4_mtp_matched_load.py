"""CPU-only contract tests for the C2/C4 matched-load runner."""

from __future__ import annotations

import importlib.util
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).with_name("r0_c2_c4_mtp_matched_load.py")
SPEC = importlib.util.spec_from_file_location("r0_c2_c4_mtp_matched_load", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def valid_manifest() -> dict:
    points = {}
    for point_index, (point, geometry) in enumerate(runner.POINTS.items()):
        requests = []
        for slot in range(geometry["concurrency"]):
            ids = [1000 + point_index * 10 + slot] + [
                100 + point_index * 10 + slot
            ] * (geometry["prompt_tokens"] - 1)
            requests.append(
                {
                    "slot": slot,
                    "token_ids": ids,
                    "token_ids_sha256": runner.token_ids_sha256(ids),
                    "expected_answer": f"answer-{point}-{slot}",
                }
            )
        points[point] = {**geometry, "requests": requests}
    return {
        "schema": 1,
        "model": "qwen38-test-pack",
        "provenance": {
            "model_pack_sha256": "a" * 64,
            "model_revision_sha256": "b" * 64,
            "model_config_sha256": "9" * 64,
            "tokenizer_sha256": "c" * 64,
            "installed_exl3_source_sha256": "e" * 64,
            "r0_source_commit": "f" * 40,
        },
        "runtime_expectations": {
            "vllm_version": "0.29.0",
            "exllamav3_revision": "d" * 40,
            "driver_version": "test-driver",
            "cuda_version": "test-cuda",
            "torch_version": "test-torch",
            "effective_max_num_batched_tokens": 2048,
        },
        "points": points,
    }


class C2C4MatchedLoadTests(unittest.TestCase):
    def test_startup_proof_requires_enginecore_and_rejects_conflicts(self) -> None:
        api_only = (
            "(APIServer pid=1) WARNING [vllm.py:1924] max_num_scheduled_tokens "
            "is set to 2048 based on the speculative decoding settings.\n"
            "(EngineCore pid=2) INFO [kv_cache_utils.py:2032] "
            "GPU KV cache size: 279,087 tokens\n"
        )
        with self.assertRaisesRegex(runner.CellError, "warning"):
            runner.parse_enginecore_startup_evidence(api_only)

        valid = (
            "(APIServer pid=1) WARNING [vllm.py:1924] max_num_scheduled_tokens "
            "is set to 4096 based on the speculative decoding settings.\n"
            "(EngineCore pid=2) WARNING [vllm.py:1924] max_num_scheduled_tokens "
            "is set to 2048 based on the speculative decoding settings.\n"
            "(EngineCore pid=2) INFO [kv_cache_utils.py:2032] "
            "GPU KV cache size: 279,087 tokens\n"
        )
        observed = runner.parse_enginecore_startup_evidence(valid)
        self.assertEqual(observed["observed_enginecore_max_num_scheduled_tokens"], 2048)
        self.assertEqual(observed["observed_enginecore_kv_pool_tokens"], 279_087)

        conflict = valid + (
            "(EngineCore pid=2) INFO [kv_cache_utils.py:2032] "
            "GPU KV cache size: 260,000 tokens\n"
        )
        with self.assertRaisesRegex(runner.CellError, "conflict"):
            runner.parse_enginecore_startup_evidence(conflict)

    def test_schedule_has_all_eight_counterbalanced_fresh_engine_cells(self) -> None:
        schedule = runner.make_schedule()
        self.assertEqual(len(schedule), 8)
        for point in runner.POINTS:
            pair = [cell["k"] for cell in schedule if cell["point"] == point]
            self.assertEqual(sorted(pair), [2, 3])
        self.assertEqual(
            [cell["k"] for cell in schedule], [2, 3, 3, 2, 3, 2, 2, 3]
        )
        self.assertTrue(all(cell["fresh_engine_required"] for cell in schedule))

    def test_manifest_validates_exact_geometry_and_prompt_hashes(self) -> None:
        doc = valid_manifest()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "prompts.json"
            raw = json.dumps(doc, separators=(",", ":")).encode()
            path.write_bytes(raw)
            loaded, loaded_raw = runner.load_prompt_manifest(path)
        self.assertEqual(loaded["model"], doc["model"])
        self.assertEqual(loaded_raw, raw)
        dry_run = runner._dry_run(loaded, loaded_raw)
        self.assertFalse(dry_run["network_used"])
        self.assertFalse(dry_run["gpu_used"])

    def test_manifest_rejects_wrong_prompt_count_and_tampered_hash(self) -> None:
        doc = valid_manifest()
        doc["points"]["c2_16k"]["requests"][0]["token_ids"].pop()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "prompts.json"
            path.write_text(json.dumps(doc))
            with self.assertRaises(runner.CellError):
                runner.load_prompt_manifest(path)

        doc = valid_manifest()
        doc["points"]["c4_16k"]["requests"][0]["token_ids_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "prompts.json"
            path.write_text(json.dumps(doc))
            with self.assertRaisesRegex(runner.CellError, "hash mismatch"):
                runner.load_prompt_manifest(path)

    def test_manifest_requires_full_lowercase_r0_commit(self) -> None:
        for bad in ("f" * 64, "f" * 39, "F" * 40, "g" * 40):
            doc = valid_manifest()
            doc["provenance"]["r0_source_commit"] = bad
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "prompts.json"
                path.write_text(json.dumps(doc))
                with self.assertRaisesRegex(runner.CellError, "r0_source_commit"):
                    runner.load_prompt_manifest(path)

    def test_cli_compares_expected_commit_and_requires_it_for_execution(self) -> None:
        manifest = valid_manifest()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "prompts.json"
            path.write_text(json.dumps(manifest))
            with contextlib.redirect_stdout(io.StringIO()):
                with patch.object(sys, "argv", [str(SCRIPT), "--manifest", str(path),
                                                  "--expected-r0-source-commit", "f" * 40,
                                                  "--dry-run"]):
                    self.assertEqual(runner.main(), 0)
            with contextlib.redirect_stderr(io.StringIO()):
                with patch.object(sys, "argv", [str(SCRIPT), "--manifest", str(path),
                                                  "--expected-r0-source-commit", "0" * 40,
                                                  "--dry-run"]):
                    self.assertEqual(runner.main(), 2)
                with patch.object(sys, "argv", [str(SCRIPT), "--manifest", str(path), "--execute"]):
                    self.assertEqual(runner.main(), 2)

    def test_runtime_proof_requires_exact_frozen_r0_commit(self) -> None:
        manifest = valid_manifest()
        proof = {
            "num_speculative_tokens": 2,
            "vllm_version": "0.29.0",
            "exllamav3_revision": "d" * 40,
            "installed_exl3_source_sha256": "e" * 64,
            "r0_source_commit": "0" * 40,
            "driver_version": "test-driver",
            "cuda_version": "test-cuda",
            "torch_version": "test-torch",
        }
        with self.assertRaisesRegex(runner.CellError, "r0_source_commit"):
            runner.validate_runtime_proof(proof, manifest, "c2_16k", 2)

    def test_manifest_rejects_any_shared_prompt_prefix(self) -> None:
        doc = valid_manifest()
        requests = doc["points"]["c2_16k"]["requests"]
        requests[1]["token_ids"][0] = requests[0]["token_ids"][0]
        requests[1]["token_ids_sha256"] = runner.token_ids_sha256(
            requests[1]["token_ids"]
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "prompts.json"
            path.write_text(json.dumps(doc))
            with self.assertRaisesRegex(runner.CellError, "share a 1-token prefix"):
                runner.load_prompt_manifest(path)

    def test_runtime_proof_enforces_frozen_envelope_and_kv_headroom(self) -> None:
        manifest = valid_manifest()
        proof = {
            "engine_id": "fresh-engine-test-1",
            "model": manifest["model"],
            "num_speculative_tokens": 2,
            "vllm_version": "0.29.0",
            "model_pack_sha256": manifest["provenance"]["model_pack_sha256"],
            "model_revision_sha256": manifest["provenance"]["model_revision_sha256"],
            "tokenizer_sha256": manifest["provenance"]["tokenizer_sha256"],
            "exllamav3_revision": manifest["runtime_expectations"][
                "exllamav3_revision"
            ],
            "installed_exl3_source_sha256": manifest["provenance"][
                "installed_exl3_source_sha256"
            ],
            "r0_source_commit": manifest["provenance"]["r0_source_commit"],
            "driver_version": "test-driver",
            "cuda_version": "test-cuda",
            "torch_version": "test-torch",
            "kv_pool_tokens": 200_000,
            "config": {
                **runner.FROZEN_CONFIG,
                "effective_max_num_batched_tokens": manifest[
                    "runtime_expectations"
                ]["effective_max_num_batched_tokens"],
            },
            "effective_budget_evidence": {
                "kind": "vllm_0.29.0_enginecore_source_inference",
                "log_line_source": "EngineCore",
                "observed_enginecore_max_num_scheduled_tokens": 2048,
                "scheduled_tokens_log_line": "(EngineCore pid=2) WARNING [vllm.py:1924] max_num_scheduled_tokens is set to 2048 based on the speculative decoding settings.",
                "observed_enginecore_kv_pool_tokens": 200_000,
                "kv_pool_log_lines": ["(EngineCore pid=2) INFO [kv_cache_utils.py:2032] GPU KV cache size: 200,000 tokens"],
                "source_default_relation": "max_num_scheduled_tokens=None -> max_num_batched_tokens",
                "scheduler_override_absent": True,
                "scheduled_token_cli_arg_absent": True,
                "batched_token_cli_arg_absent": True,
                "engine_args_default_none_verified": True,
                "log_line_count": 1,
            },
        }
        runner.validate_runtime_proof(proof, manifest, "c2_80k", 2)
        proof["config"]["prefix_caching"] = True
        with self.assertRaisesRegex(runner.CellError, "prefix_caching"):
            runner.validate_runtime_proof(proof, manifest, "c2_80k", 2)

        proof["config"]["prefix_caching"] = False
        proof["effective_budget_evidence"]["log_line_source"] = "APIServer"
        with self.assertRaisesRegex(runner.CellError, "EngineCore"):
            runner.validate_runtime_proof(proof, manifest, "c2_80k", 2)


if __name__ == "__main__":
    unittest.main()
