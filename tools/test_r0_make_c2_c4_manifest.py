"""CPU-only tests for transparent archived-turn prompt derivation."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).with_name("r0_make_c2_c4_manifest.py")
SPEC = importlib.util.spec_from_file_location("r0_make_c2_c4_manifest", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


class TinyTokenizer:
    pieces = {
        10: "historical prefix ",
        11: "KVMEM_FACT_A_104729 stores cobalt-lantern-47. ",
        12: "What is the recovery code for KVMEM_FACT_A_104729?",
        20: "Amber",
        21: "Birch",
        30: " neutral",
    }

    def encode(self, text, add_special_tokens=False):
        reverse = {value: key for key, value in self.pieces.items()}
        if text in reverse:
            return [reverse[text]]
        raise ValueError("test tokenizer accepts only its fixture token strings")

    def decode(self, ids, **kwargs):
        return "".join(self.pieces[token] for token in ids)


class FixtureTokenizer:
    all_special_ids = [0]

    def __init__(self):
        self.pieces = {100 + code: chr(code) for code in range(32, 128)}
        self.pieces.update({500 + i: word for i, word in enumerate(generator.LEAD_WORDS)})
        self.pieces[700] = " neutral"
        self.reverse = {piece: token for token, piece in self.pieces.items()}

    def __len__(self):
        return 1000

    def get_vocab(self):
        return self.reverse

    def encode(self, text, add_special_tokens=False):
        if text in self.reverse:
            return [self.reverse[text]]
        ids = []
        position = 0
        while position < len(text):
            if text.startswith(" neutral", position):
                ids.append(700)
                position += len(" neutral")
            else:
                ids.append(self.reverse[text[position]])
                position += 1
        return ids

    def decode(self, ids, **kwargs):
        return "".join(self.pieces[token] for token in ids)


class ArchivedTurnDerivationTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TinyTokenizer()
        self.source = {
            "prompt_tokens": 3,
            "prompt_token_ids": [10, 11, 12],
            "query_span": [2, 3],
            "query_text": self.tokenizer.pieces[12],
            "target_facts": [{
                "marker": "KVMEM_FACT_A_104729",
                "code": "cobalt-lantern-47",
            }],
        }

    def test_derivation_preserves_archived_ids_and_inserts_filler_before_query(self):
        with tempfile.TemporaryDirectory() as temp:
            source_file = Path(temp) / "turn.json"
            source_file.write_text("archived parent fixture", encoding="utf-8")
            ids, answer, proof = generator.derive_archived_request(
                self.tokenizer, self.source, source_file, "ctx16000/turn_00_ask_a.json",
                target_tokens=8, lead_token=20, filler_id=30,
            )
            self.assertEqual(len(ids), 8)
            self.assertEqual(ids[0], 20)
            self.assertEqual(ids[1], 11)
            self.assertEqual(ids[2:7], [30] * 5)
            self.assertEqual(ids[7:], [12])
            self.assertEqual(self.tokenizer.decode(ids[7:]), self.source["query_text"])
            self.assertIn(answer, self.tokenizer.decode(ids))
            self.assertEqual(answer, "cobalt-lantern-47")
            self.assertEqual(proof["filler_count"], 5)
            self.assertEqual(proof["source_token_ids_sha256"],
                             generator.runner.token_ids_sha256([10, 11, 12]))
            self.assertEqual(proof["input_classification"],
                             "derived; not historical byte-identical input")

    def test_derivation_fails_if_query_text_or_single_recovery_fact_is_ambiguous(self):
        with tempfile.TemporaryDirectory() as temp:
            source_file = Path(temp) / "turn.json"
            source_file.write_text("parent", encoding="utf-8")
            bad_query = {**self.source, "query_text": "different"}
            with self.assertRaisesRegex(generator.ManifestError, "Decoded archived query"):
                generator.derive_archived_request(
                    self.tokenizer, bad_query, source_file, "ctx16000/turn_00_ask_a.json", 8, 20, 30
                )
            bad_facts = {**self.source, "target_facts": self.source["target_facts"] * 2}
            with self.assertRaisesRegex(generator.ManifestError, "exactly one"):
                generator.derive_archived_request(
                    self.tokenizer, bad_facts, source_file, "ctx16000/turn_00_ask_a.json", 8, 20, 30
                )

    def test_derivation_fails_if_source_exceeds_requested_length(self):
        with tempfile.TemporaryDirectory() as temp:
            source_file = Path(temp) / "turn.json"
            source_file.write_text("parent", encoding="utf-8")
            with self.assertRaisesRegex(generator.ManifestError, "exceeds target"):
                generator.derive_archived_request(
                    self.tokenizer, self.source, source_file, "ctx16000/turn_00_ask_a.json", 2, 20, 30
                )

    def test_manifest_builder_uses_only_archived_turns_and_marks_c4_32k_blocked(self):
        tokenizer = FixtureTokenizer()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model"
            tokenizer_path = model / "tokenizer"
            tokenizer_path.mkdir(parents=True)
            (model / "config.json").write_text("{}", encoding="utf-8")
            (tokenizer_path / "tokenizer.json").write_text("{}", encoding="utf-8")
            sources = root / "sources"
            for context in (16000, 80000):
                folder = sources / f"ctx{context}"
                folder.mkdir(parents=True)
                target_tokens = 15_533 if context == 16000 else 79_533
                for slot, filename in enumerate(generator.TURN_FILES):
                    marker = f"KVMEM_FACT_{slot}_{context}"
                    code = f"recovery-{slot}-{context}"
                    query = f"What is the code for {marker}?"
                    query_ids = tokenizer.encode(query)
                    history_ids = tokenizer.encode(f"{marker}={code};")
                    source_length = target_tokens - 1
                    padding = source_length - 1 - len(history_ids) - len(query_ids)
                    source_ids = [tokenizer.reverse["x"]] + history_ids + [
                        tokenizer.reverse["x"]
                    ] * padding + query_ids
                    query_start = len(source_ids) - len(query_ids)
                    source = {
                        "context_limit": context,
                        "prompt_tokens": len(source_ids),
                        "prompt_token_ids": source_ids,
                        "query_span": [query_start, len(source_ids)],
                        "query_text": query,
                        "target_facts": [{"marker": marker, "code": code}],
                    }
                    (folder / filename).write_text(json.dumps(source), encoding="utf-8")
            args = SimpleNamespace(
                model_path=model,
                tokenizer_path=tokenizer_path,
                source_prompts=sources,
                model="fixture-model",
                model_revision_sha256="a" * 64,
                installed_exl3_source_sha256="b" * 64,
                r0_source_commit="c" * 40,
                vllm_version="0.29.0",
                exllamav3_revision="d" * 40,
                driver_version="fixture-driver",
                cuda_version="fixture-cuda",
                torch_version="fixture-torch",
                effective_max_num_batched_tokens=2048,
            )
            manifest = generator.make_manifest(args, lambda _: tokenizer)
        self.assertEqual(set(manifest["points"]), {"c2_16k", "c2_80k", "c4_16k"})
        self.assertEqual(manifest["blocked_points"], generator.runner.BLOCKED_POINTS)
        self.assertEqual(manifest["input_classification"],
                         "derived_from_archived_turns_not_historical_byte_identical")
        for point, spec in manifest["points"].items():
            for request in spec["requests"]:
                self.assertEqual(len(request["token_ids"]), spec["prompt_tokens"])
                self.assertEqual(request["derivation"]["source_file_sha256"].__len__(), 64)
                self.assertEqual(request["derivation"]["expected_single_recovery_code"],
                                 request["expected_answer"])


if __name__ == "__main__":
    unittest.main()
