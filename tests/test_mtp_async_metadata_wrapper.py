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
        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()
"""

SHORT_CONV = """from typing import Any
import torch
from vllm.utils.torch_utils import async_tensor_h2d


class Builder:
    def build(self, m, query_start_loc, num_accepted_tokens):
        spec_req_idx_cpu = spec_sequence_masks_cpu.nonzero(as_tuple=True)[0]
        decode_req_idx_cpu = decode_mask_cpu.nonzero(as_tuple=True)[0]
        prefill_req_idx_cpu = prefill_mask_cpu.nonzero(as_tuple=True)[0]
        non_spec_req_idx_cpu = torch.cat((decode_req_idx_cpu, prefill_req_idx_cpu))
        spec_req_idx = spec_req_idx_cpu.to(query_start_loc.device)
        non_spec_req_idx = non_spec_req_idx_cpu.to(query_start_loc.device)

        if mixed:
            req_group = torch.full(
                (m.num_reqs,),
                2,
                dtype=torch.int32,
                device=query_start_loc.device,
            )
            req_group[spec_req_idx] = 0
            req_group[decode_req_idx_cpu.to(query_start_loc.device)] = 1

        assert num_accepted_tokens is not None
        num_accepted_tokens = num_accepted_tokens[
            spec_req_idx_cpu.to(num_accepted_tokens.device)
        ]

        if num_decodes > 0 or num_prefills > 0:
            num_computed_tokens = m.compute_num_computed_tokens()
            if non_spec_req_idx_cpu is not None:
                non_spec_req_idx = non_spec_req_idx_cpu.to(num_computed_tokens.device)
                num_computed_tokens = num_computed_tokens[non_spec_req_idx]
        return num_accepted_tokens
"""


def _tree(tmp_path: Path) -> Path:
    root = tmp_path / "vllm"
    qwen = root / "models" / "qwen4_exp" / "nvidia"
    qwen.mkdir(parents=True)
    (qwen / "model.py").write_text(MODEL)
    (qwen / "ple_layer.py").write_text(PLE)
    (qwen / "mtp.py").write_text(MTP)

    short_conv = root / "v1" / "attention" / "backends" / "short_conv_attn.py"
    short_conv.parent.mkdir(parents=True)
    short_conv.write_text(SHORT_CONV)
    return root


def _run(root: Path, profile: str, async_metadata: bool) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(WRAPPER),
        str(root),
        "--profile",
        profile,
    ]
    if async_metadata:
        cmd.append("--mtp-async-metadata")
    return subprocess.run(cmd, text=True, capture_output=True, check=False)


def test_wrapper_applies_async_metadata_only_when_explicitly_requested(tmp_path):
    root = _tree(tmp_path)
    run = _run(root, "text-mtp", True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "mtp_async_metadata=1" in run.stdout

    qwen = root / "models" / "qwen4_exp" / "nvidia"
    assert "quant_config=self.quant_config" in (qwen / "model.py").read_text()
    assert "quant_config=self.quant_config" in (qwen / "mtp.py").read_text()

    short_conv = (
        root / "v1" / "attention" / "backends" / "short_conv_attn.py"
    ).read_text()
    assert "spec_req_idx = async_tensor_h2d(" in short_conv
    assert "num_accepted_tokens = num_accepted_tokens[spec_req_idx]" in short_conv

    second = _run(root, "text-mtp", True)
    assert second.returncode == 0, second.stdout + second.stderr
    assert "already patched:" in second.stdout


def test_wrapper_rejects_async_metadata_for_no_draft_profile(tmp_path):
    root = _tree(tmp_path)
    run = _run(root, "text-no-draft", True)
    assert run.returncode == 2
    assert "valid only with an MTP profile" in run.stderr
