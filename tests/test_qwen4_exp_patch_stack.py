import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "tools" / "apply_qwen4_exp_patches.py"


MODEL = """class Model:
    def __init__(self, config, prefix):
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )

    def load_weights(self):
        mapper = WeightsMapper(
            orig_to_new_substr={"mtp.": None},
            orig_to_new_prefix={"visual.": None} if self.language_model_only else {},
        )
"""

PLE = """class PLE:
    def __init__(self, padded_vocab_size, divisor, params_dtype, prefix, quant_config):
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

MTP = """class MTP:
    def __init__(self, config, prefix):
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
"""


def _tree(tmp_path: Path) -> Path:
    root = tmp_path / "vllm"
    qwen = root / "models" / "qwen4_exp" / "nvidia"
    qwen.mkdir(parents=True)
    (qwen / "model.py").write_text(MODEL)
    (qwen / "ple_layer.py").write_text(PLE)
    (qwen / "mtp.py").write_text(MTP)
    return root


def _run(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(WRAPPER), str(root)],
        text=True,
        capture_output=True,
        check=False,
    )


def test_qwen_patch_wrapper_applies_complete_stack_and_is_idempotent(tmp_path):
    root = _tree(tmp_path)
    first = _run(root)
    assert first.returncode == 0, first.stdout + first.stderr
    assert "PASS: Qwen4Exp EXL3 patch stack applied" in first.stdout

    qwen = root / "models" / "qwen4_exp" / "nvidia"
    model = (qwen / "model.py").read_text()
    ple = (qwen / "ple_layer.py").read_text()
    mtp = (qwen / "mtp.py").read_text()

    assert "quant_config=self.quant_config" in model
    assert "quant_config=quant_config" in ple
    assert "quant_method=_get_ple_embedding_quant_method(" in ple
    assert "quant_config=self.quant_config" in mtp
    assert '".attn.q_proj.": None' in model
    assert '".attn.k_proj.": None' in model
    assert '".attn.v_proj.": None' in model

    second = _run(root)
    assert second.returncode == 0, second.stdout + second.stderr
    assert "PASS: Qwen4Exp EXL3 patch stack applied" in second.stdout


def test_qwen_patch_wrapper_refuses_partial_source_contract(tmp_path):
    root = _tree(tmp_path)
    qwen = root / "models" / "qwen4_exp" / "nvidia"
    # Break the MTP anchor so the second patch fails. The wrapper must stop
    # before claiming the stack is applied.
    (qwen / "mtp.py").write_text("class MTP: pass\n")

    run = _run(root)
    assert run.returncode != 0
    assert "patch_vllm_mtp_lmhead.py failed" in run.stderr
    assert "PASS: Qwen4Exp EXL3 patch stack applied" not in run.stdout
