"""CPU-only deterministic manifest generator tests."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).with_name("r0_make_c2_c4_manifest.py")
SPEC = importlib.util.spec_from_file_location("r0_make_c2_c4_manifest", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


class FakeTokenizer:
    all_special_ids = [0]

    def __init__(self):
        alphabet = set(
            generator.PROMPT_PREFIX.format(lead="Amber")
            + generator.PROMPT_SUFFIX.format(answer="r0-c2-c4-c2_16k-0")
            + "".join(generator.LEAD_WORDS)
        )
        alphabet.update("0123456789_")
        self.char_to_id = {char: index + 1 for index, char in enumerate(sorted(alphabet))}
        self.id_to_piece = {value: key for key, value in self.char_to_id.items()}
        self.filler_id = len(self.char_to_id) + 1
        self.id_to_piece[self.filler_id] = " neutral"

    def __len__(self):
        return self.filler_id + 1

    def get_vocab(self):
        return {piece: token for token, piece in self.id_to_piece.items()}

    def encode(self, text, add_special_tokens=False):
        ids = []
        index = 0
        while index < len(text):
            if text.startswith(" neutral", index):
                ids.append(self.filler_id)
                index += len(" neutral")
            else:
                ids.append(self.char_to_id[text[index]])
                index += 1
        return ids

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(self.id_to_piece[token] for token in ids)


def args_for(root: Path, tokenizer_path: Path | None = None):
    return SimpleNamespace(
        model_path=root,
        tokenizer_path=tokenizer_path or root,
        model="local-test-model",
        model_revision_sha256="a" * 64,
        installed_exl3_source_sha256="b" * 64,
        r0_source_commit="c" * 40,
        vllm_version="0.29.0",
        exllamav3_revision="d" * 40,
        driver_version="test-driver",
        cuda_version="test-cuda",
        torch_version="test-torch",
        effective_max_num_batched_tokens=2048,
    )


class C2C4ManifestGeneratorTests(unittest.TestCase):
    @staticmethod
    def make_assets(root: Path):
        (root / "config.json").write_text('{"model_type":"test"}', encoding="utf-8")
        (root / "tokenizer.json").write_text('{"version":"1.0"}', encoding="utf-8")

    def test_output_is_deterministic_exact_length_and_disjoint_by_lead(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_assets(root)
            args = args_for(root)
            tokenizer = FakeTokenizer()
            first = generator.make_manifest(args, lambda _: tokenizer)
            second = generator.make_manifest(args, lambda _: tokenizer)
        self.assertEqual(first, second)
        self.assertEqual(first["provenance"]["r0_source_commit"], "c" * 40)
        self.assertNotIn("r0_source_sha256", first["provenance"])
        all_leads = []
        for point, geometry in generator.runner.POINTS.items():
            requests = first["points"][point]["requests"]
            self.assertEqual(
                [len(item["token_ids"]) for item in requests],
                [geometry["prompt_tokens"]] * geometry["concurrency"],
            )
            self.assertEqual(
                len({item["token_ids"][0] for item in requests}),
                geometry["concurrency"],
            )
            all_leads.extend(item["token_ids"][0] for item in requests)
            for item in requests:
                self.assertEqual(
                    item["token_ids_sha256"],
                    generator.runner.token_ids_sha256(item["token_ids"]),
                )
                prompt_text = tokenizer.decode(item["token_ids"])
                self.assertIn("Reply with exactly the answer code", prompt_text)
                self.assertTrue(prompt_text.endswith(item["expected_answer"]))
                self.assertEqual(
                    tokenizer.encode(prompt_text, add_special_tokens=False),
                    item["token_ids"],
                )
        self.assertEqual(len(set(all_leads)), len(all_leads))

    def test_fails_closed_when_tokenizer_cannot_supply_distinct_leads(self):
        class TinyTokenizer:
            all_special_ids = [0, 2]

            def __len__(self):
                return 4

            def get_vocab(self):
                return {"only": 1}

            def encode(self, text, add_special_tokens=False):
                return [1]

            def decode(self, ids, **kwargs):
                return " neutral"

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_assets(root)
            with self.assertRaisesRegex(generator.ManifestError, "distinct ordinary lead"):
                generator.make_manifest(args_for(root), lambda _: TinyTokenizer())

    def test_fails_closed_for_tokenizer_outside_model_tree(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model, tokenizer = root / "model", root / "tokenizer"
            model.mkdir()
            tokenizer.mkdir()
            self.make_assets(model)
            with self.assertRaisesRegex(generator.ManifestError, "inside model path"):
                generator.make_manifest(
                    args_for(model, tokenizer), lambda _: FakeTokenizer()
                )


if __name__ == "__main__":
    unittest.main()
