import subprocess
import sys
from pathlib import Path

import pytest


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


def _run(root: Path, profile: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(WRAPPER),
            str(root),
            "--profile",
            profile,
        ],
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("profile", "expect_mtp", "expect_vision"),
    [
        ("text-no-draft", False, False),
        ("text-mtp", True, False),
        ("multimodal-no-draft", False, True),
        ("multimodal-mtp", True, True),
    ],
)
def test_qwen_patch_wrapper_applies_minimum_profile_and_is_idempotent(
    tmp_path, profile, expect_mtp, expect_vision
):
    root = _tree(tmp_path)
    first = _run(root, profile)
    assert first.returncode == 0, first.stdout + first.stderr
    assert f"profile={profile}" in first.stdout

    qwen = root / "models" / "qwen4_exp" / "nvidia"
    model = (qwen / "model.py").read_text()
    ple = (qwen / "ple_layer.py").read_text()
    mtp = (qwen / "mtp.py").read_text()

    assert "quant_config=self.quant_config" in model
    assert "quant_config=quant_config" in ple
    assert "quant_method=_get_ple_embedding_quant_method(" in ple
    assert ("quant_config=self.quant_config" in mtp) is expect_mtp
    for token in (
        '".attn.q_proj.": None',
        '".attn.k_proj.": None',
        '".attn.v_proj.": None',
    ):
        assert (token in model) is expect_vision

    second = _run(root, profile)
    assert second.returncode == 0, second.stdout + second.stderr
    assert "PASS: Qwen4Exp EXL3 patch stack applied" in second.stdout


def test_text_no_draft_does_not_require_mtp_or_vision_anchors(tmp_path):
    root = _tree(tmp_path)
    qwen = root / "models" / "qwen4_exp" / "nvidia"
    (qwen / "mtp.py").write_text("class MTP: pass\n")

    run = _run(root, "text-no-draft")
    assert run.returncode == 0, run.stdout + run.stderr
    model = (qwen / "model.py").read_text()
    assert '".attn.k_proj.": None' not in model


def test_text_mtp_refuses_broken_mtp_anchor(tmp_path):
    root = _tree(tmp_path)
    qwen = root / "models" / "qwen4_exp" / "nvidia"
    (qwen / "mtp.py").write_text("class MTP: pass\n")

    run = _run(root, "text-mtp")
    assert run.returncode != 0
    assert "patch_vllm_mtp_lmhead.py failed" in run.stderr
    assert "PASS: Qwen4Exp EXL3 patch stack applied" not in run.stdout


def test_multimodal_refuses_broken_vision_anchor(tmp_path):
    root = _tree(tmp_path)
    qwen = root / "models" / "qwen4_exp" / "nvidia"
    model = (qwen / "model.py").read_text()
    model = model.replace(
        'orig_to_new_prefix={"visual.": None} if self.language_model_only else {},',
        "orig_to_new_prefix={},",
    )
    (qwen / "model.py").write_text(model)

    run = _run(root, "multimodal-no-draft")
    assert run.returncode != 0
    assert "patch_vllm_vision_split.py failed" in run.stderr
