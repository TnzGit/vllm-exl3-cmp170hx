"""Patch Qwen4Exp so EXL3 quantization reaches lm_head and the PLE n-gram table.

vLLM has used two PLE constructor shapes across the Qwen4Exp integration:

* older builds construct ``PLEVocabParallelEmbedding`` without a model-specific
  ``quant_method``;
* vLLM 0.29.x preselects an FP8-only PLE method with
  ``_get_ple_embedding_quant_method(...)``.

For the 0.29.x shape we must keep that preselected method *and* pass
``quant_config``. ``VocabParallelEmbedding`` only consults ``quant_config`` when
``quant_method`` is ``None``. This preserves the Qwen FP8 special case while
allowing EXL3's ``Exl3Config.get_quant_method`` to return
``Exl3EmbeddingMethod`` for the row-wise n-gram table.

The main lm_head still needs ``quant_config=self.quant_config``.

Backs up each changed file to <file>.orig (kept if present), is idempotent, and
compile-checks the result.

usage: python3 patch_vllm_qwen4_ple.py <site-packages/vllm>
"""
import os
import shutil
import sys

MODEL_OLD = """        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
"""
MODEL_NEW = """        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
"""

# Older Qwen4Exp PLE constructor.
PLE_OLD_LEGACY = """        self.ngram_embedding = PLEVocabParallelEmbedding(
            padded_vocab_size,
            self.head_dim,
            params_dtype=params_dtype,
            padding_size=divisor,
            prefix=f"{prefix}.ngram_embedding",
"""
PLE_NEW_LEGACY = """        self.ngram_embedding = PLEVocabParallelEmbedding(
            padded_vocab_size,
            self.head_dim,
            params_dtype=params_dtype,
            padding_size=divisor,
            quant_config=quant_config,
            prefix=f"{prefix}.ngram_embedding",
"""

# vLLM 0.29.x: retain the model-specific FP8 method, but also pass the full
# quant config. When the helper returns None for EXL3, VocabParallelEmbedding
# falls through to quant_config.get_quant_method(self, prefix=...).
PLE_OLD_029 = """        self.ngram_embedding = PLEVocabParallelEmbedding(
            padded_vocab_size,
            self.head_dim,
            params_dtype=params_dtype,
            padding_size=divisor,
            prefix=f"{prefix}.ngram_embedding",
            quant_method=_get_ple_embedding_quant_method(
                quant_config, f"{prefix}.ngram_embedding"
            ),
"""
PLE_NEW_029 = """        self.ngram_embedding = PLEVocabParallelEmbedding(
            padded_vocab_size,
            self.head_dim,
            params_dtype=params_dtype,
            padding_size=divisor,
            quant_config=quant_config,
            prefix=f"{prefix}.ngram_embedding",
            quant_method=_get_ple_embedding_quant_method(
                quant_config, f"{prefix}.ngram_embedding"
            ),
"""


def _write_checked(path: str, out: str) -> bool:
    try:
        compile(out, path, "exec")
    except SyntaxError as e:
        print(f"ERROR: patched source does not compile: {e}")
        return False
    backup = path + ".orig"
    if not os.path.exists(backup):
        shutil.copyfile(path, backup)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(out)
    print(f"patched {path} (backup {backup})")
    return True


def patch_one(path: str, old: str, new: str) -> bool:
    if not os.path.exists(path):
        print(f"ERROR: {path} not found")
        return False
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    if new in src:
        print(f"already patched: {path}")
        return True
    n = src.count(old)
    if n != 1:
        print(f"ERROR: anchor found {n} times in {path} (need 1)")
        return False
    return _write_checked(path, src.replace(old, new))


def patch_ple(path: str) -> bool:
    if not os.path.exists(path):
        print(f"ERROR: {path} not found")
        return False
    with open(path, encoding="utf-8") as fh:
        src = fh.read()

    if PLE_NEW_029 in src or PLE_NEW_LEGACY in src:
        print(f"already patched: {path}")
        return True

    matches = []
    for old, new, name in (
        (PLE_OLD_029, PLE_NEW_029, "vLLM 0.29.x"),
        (PLE_OLD_LEGACY, PLE_NEW_LEGACY, "legacy"),
    ):
        n = src.count(old)
        if n:
            matches.append((old, new, name, n))

    if len(matches) != 1 or matches[0][3] != 1:
        detail = ", ".join(f"{name}={n}" for _, _, name, n in matches) or "none"
        print(
            "ERROR: PLE constructor did not match exactly one supported layout "
            f"in {path} ({detail})"
        )
        return False

    old, new, name, _ = matches[0]
    print(f"detected PLE layout: {name}")
    return _write_checked(path, src.replace(old, new))


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    root = os.path.join(sys.argv[1], "models", "qwen4_exp", "nvidia")
    ok = patch_one(os.path.join(root, "model.py"), MODEL_OLD, MODEL_NEW)
    ok = patch_ple(os.path.join(root, "ple_layer.py")) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
