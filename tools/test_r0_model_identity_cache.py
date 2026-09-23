"""CPU-only tests for trusted-filesystem model identity hash reuse."""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).with_name("r0_model_identity_cache.py")
SPEC = importlib.util.spec_from_file_location("r0_model_identity_cache", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
cache = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cache)


class ModelIdentityCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.model = self.base / "model"
        self.model.mkdir()
        (self.model / "config.json").write_bytes(b'{"model_type":"tiny"}\n')
        (self.model / "tokenizer.json").write_bytes(b'{"tokenizer":"tiny"}\n')
        (self.model / "weights.safetensors").write_bytes(b"tiny synthetic weights\n")
        self.manifest_path = self.base / "manifest.json"
        self.cache_path = self.base / "cache" / "identity.json"
        self.output_path = self.base / "first" / "input_hashes.json"
        identity = cache.snapshot(self.model, b"")
        hashes = cache._full_hashes(self.model.resolve(), identity)
        points = {}
        for point_index, (point, geometry) in enumerate(cache.runner.POINTS.items()):
            requests = []
            for slot in range(geometry["concurrency"]):
                ids = [100 + point_index * 10 + slot] + [7] * (
                    geometry["prompt_tokens"] - 1
                )
                requests.append({
                    "slot": slot,
                    "token_ids": ids,
                    "token_ids_sha256": cache.runner.token_ids_sha256(ids),
                    "expected_answer": f"answer-{point}-{slot}",
                })
            points[point] = {**geometry, "requests": requests}
        document = {
            "schema": 1,
            "model": "tiny-model",
            "provenance": {
                "model_pack_sha256": hashes["model_tree_sha256"],
                "model_revision_sha256": "a" * 64,
                "model_config_sha256": hashes["model_config_sha256"],
                "tokenizer_sha256": hashes["tokenizer_tree_sha256"],
                "installed_exl3_source_sha256": "b" * 64,
                "r0_source_commit": "c" * 40,
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
        self.manifest_path.write_text(json.dumps(document), encoding="utf-8")
        cache.runner.load_prompt_manifest(self.manifest_path)

    def tearDown(self):
        self.temp.cleanup()

    def verify(self, output: Path | None = None) -> str:
        return cache.verify_or_reuse(
            self.model,
            self.manifest_path,
            self.cache_path,
            output or self.output_path,
            "c" * 40,
        )

    def test_full_hash_then_reuses_only_matching_identity(self):
        self.assertEqual(self.verify(), "full_hash")
        proof = json.loads(self.cache_path.read_text(encoding="utf-8"))
        self.assertEqual(proof["proof"], "full_model_tree_hashes_verified")
        first_output = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.assertEqual(first_output["model_file_count"], 3)

        with patch.object(cache, "_full_hashes", side_effect=AssertionError("unexpected rehash")):
            mode = self.verify(self.base / "second" / "input_hashes.json")
        self.assertEqual(mode, "cached")

    def test_metadata_only_change_causes_a_fresh_full_hash(self):
        self.verify()
        asset = self.model / "weights.safetensors"
        before = asset.stat().st_mtime_ns
        os.utime(asset, ns=(asset.stat().st_atime_ns, before + 2_000_000_000))
        original = cache._full_hashes
        with patch.object(cache, "_full_hashes", wraps=original) as hasher:
            self.assertEqual(self.verify(self.base / "after-touch" / "input_hashes.json"), "full_hash")
            self.assertEqual(hasher.call_count, 1)

    def test_malformed_cached_summary_falls_back_to_full_hash(self):
        self.verify()
        proof = json.loads(self.cache_path.read_text(encoding="utf-8"))
        proof["hashes"]["model_file_count"] = 999
        self.cache_path.write_text(json.dumps(proof), encoding="utf-8")
        with patch.object(cache, "_full_hashes", wraps=cache._full_hashes) as hasher:
            self.assertEqual(self.verify(self.base / "repaired" / "input_hashes.json"), "full_hash")
            self.assertEqual(hasher.call_count, 1)
        repaired = json.loads(self.cache_path.read_text(encoding="utf-8"))
        self.assertEqual(repaired["hashes"]["model_file_count"], 3)

    def test_metadata_change_during_full_hash_prevents_proof_write(self):
        original = cache._full_hashes

        def hash_then_touch(root, identity):
            hashes = original(root, identity)
            asset = self.model / "weights.safetensors"
            info = asset.stat()
            os.utime(asset, ns=(info.st_atime_ns, info.st_mtime_ns + 3_000_000_000))
            return hashes

        with patch.object(cache, "_full_hashes", side_effect=hash_then_touch):
            with self.assertRaisesRegex(cache.CacheError, "changed during full hash"):
                self.verify()
        self.assertFalse(self.cache_path.exists())
        self.assertFalse(self.output_path.exists())

    def test_content_tampering_rehashes_and_fails_manifest_comparison(self):
        self.verify()
        (self.model / "weights.safetensors").write_bytes(b"modified weights\n")
        original = cache._full_hashes
        with patch.object(cache, "_full_hashes", wraps=original) as hasher:
            with self.assertRaisesRegex(cache.CacheError, "model_pack_sha256"):
                self.verify(self.base / "tampered" / "input_hashes.json")
            self.assertEqual(hasher.call_count, 1)
        self.assertFalse((self.base / "tampered" / "input_hashes.json").exists())

    def test_added_file_changes_exact_set_and_rehashes(self):
        self.verify()
        (self.model / "extra.bin").write_bytes(b"new")
        with patch.object(cache, "_full_hashes", wraps=cache._full_hashes) as hasher:
            with self.assertRaisesRegex(cache.CacheError, "model_pack_sha256"):
                self.verify(self.base / "added" / "input_hashes.json")
            self.assertEqual(hasher.call_count, 1)

    def test_model_symlink_and_hardlink_assets_fail_closed(self):
        outside = self.base / "outside.bin"
        outside.write_bytes(b"outside")
        (self.model / "linked.bin").symlink_to(outside)
        with self.assertRaisesRegex(cache.CacheError, "Symlink"):
            self.verify()
        (self.model / "linked.bin").unlink()

        os.link(self.model / "weights.safetensors", self.base / "hardlink.bin")
        with self.assertRaisesRegex(cache.CacheError, "Hard-linked"):
            self.verify()

    def test_manifest_bytes_are_part_of_cache_identity(self):
        self.verify()
        document = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        document["runtime_expectations"]["driver_version"] = "test-driver-updated"
        self.manifest_path.write_text(json.dumps(document), encoding="utf-8")
        with patch.object(cache, "_full_hashes", wraps=cache._full_hashes) as hasher:
            self.assertEqual(self.verify(self.base / "new-manifest" / "input_hashes.json"), "full_hash")
            self.assertEqual(hasher.call_count, 1)


if __name__ == "__main__":
    unittest.main()
