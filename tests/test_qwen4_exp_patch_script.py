import subprocess
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "patch_vllm_qwen4_exp"
    / "patch_vllm_qwen4_ple.py"
)


MODEL = """class Model:
    def build(self, config, prefix):
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
"""


PLE_029 = """class PLE:
    def build(self, padded_vocab_size, divisor, params_dtype, prefix, quant_config):
        self.ngram_embedding = PLEVocabParallelEmbedding(
            padded_vocab_size,
            self.head_dim,
            params_dtype=params_dtype,
            padding_size=divisor,
            prefix=f"{prefix}.ngram_embedding",
            quant_method=_get_ple_embedding_quant_method(
                quant_config, f"{prefix}.ngram_embedding"
            ),
        )
"""


PLE_LEGACY = """class PLE:
    def build(self, padded_vocab_size, divisor, params_dtype, prefix, quant_config):
        self.ngram_embedding = PLEVocabParallelEmbedding(
            padded_vocab_size,
            self.head_dim,
            params_dtype=params_dtype,
            padding_size=divisor,
            prefix=f"{prefix}.ngram_embedding",
        )
"""


def _tree(tmp_path: Path, ple: str) -> Path:
    root = tmp_path / "vllm"
    qwen = root / "models" / "qwen4_exp" / "nvidia"
    qwen.mkdir(parents=True)
    (qwen / "model.py").write_text(MODEL)
    (qwen / "ple_layer.py").write_text(ple)
    return root


def _run(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(root)],
        text=True,
        capture_output=True,
        check=False,
    )


def test_patch_vllm_029_ple_keeps_preselected_method_and_adds_quant_config(tmp_path):
    root = _tree(tmp_path, PLE_029)
    first = _run(root)
    assert first.returncode == 0, first.stdout + first.stderr

    qwen = root / "models" / "qwen4_exp" / "nvidia"
    model = (qwen / "model.py").read_text()
    ple = (qwen / "ple_layer.py").read_text()
    assert "quant_config=self.quant_config" in model
    assert "quant_config=quant_config" in ple
    assert "quant_method=_get_ple_embedding_quant_method(" in ple
    assert "detected PLE layout: vLLM 0.29.x" in first.stdout

    second = _run(root)
    assert second.returncode == 0, second.stdout + second.stderr
    assert second.stdout.count("already patched:") == 2


def test_patch_legacy_ple_adds_quant_config(tmp_path):
    root = _tree(tmp_path, PLE_LEGACY)
    run = _run(root)
    assert run.returncode == 0, run.stdout + run.stderr
    ple = (
        root / "models" / "qwen4_exp" / "nvidia" / "ple_layer.py"
    ).read_text()
    assert "quant_config=quant_config" in ple
    assert "detected PLE layout: legacy" in run.stdout
