import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "tools" / "cmp170hx_qwen_preflight.py"

spec = importlib.util.spec_from_file_location("cmp170hx_qwen_preflight", PREFLIGHT)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


MODEL_BASE = """class Model:
    def __init__(self, config, prefix):
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )

    def load_weights(self):
        mapper = WeightsMapper(
            orig_to_new_substr={"mtp.": None},
            orig_to_new_prefix={"visual.": None} if self.language_model_only else {},
        )
"""

MODEL_VISION = MODEL_BASE.replace(
    'orig_to_new_substr={"mtp.": None},',
    '''orig_to_new_substr={
                "mtp.": None,
                ".attn.q_proj.": None,
                ".attn.k_proj.": None,
                ".attn.v_proj.": None,
            },''',
)

PLE = """class PLE:
    def __init__(self, padded_vocab_size, divisor, params_dtype, prefix, quant_config):
        self.ngram_embedding = PLEVocabParallelEmbedding(
            padded_vocab_size,
            self.head_dim,
            params_dtype=params_dtype,
            padding_size=divisor,
            quant_config=quant_config,
            prefix=f"{prefix}.ngram_embedding",
            quant_method=_get_ple_embedding_quant_method(
                quant_config, f"{prefix}.ngram_embedding"
            ),
        )
"""

MTP_PATCHED = """class MTP:
    def __init__(self, config, prefix):
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
"""

MTP_UNPATCHED = """class MTP:
    def __init__(self, config, prefix):
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
"""


def _tree(tmp_path: Path, *, vision: bool, mtp: bool) -> Path:
    root = tmp_path / "vllm"
    qwen = root / "models" / "qwen4_exp" / "nvidia"
    qwen.mkdir(parents=True, exist_ok=True)
    (qwen / "model.py").write_text(MODEL_VISION if vision else MODEL_BASE)
    (qwen / "ple_layer.py").write_text(PLE)
    (qwen / "mtp.py").write_text(MTP_PATCHED if mtp else MTP_UNPATCHED)
    return root


def _by_name(checks):
    return {c.name: c for c in checks}


def test_text_no_draft_skips_mtp_and_vision(tmp_path):
    checks = _by_name(
        mod.check_vllm_patches(_tree(tmp_path, vision=False, mtp=False), "text-no-draft")
    )
    assert checks["main lm_head quant"].status == "PASS"
    assert checks["PLE quant plumbing"].status == "PASS"
    assert checks["MTP lm_head quant"].status == "SKIP"
    assert checks["vision split qkv"].status == "SKIP"


def test_text_mtp_requires_only_mtp(tmp_path):
    bad = _by_name(
        mod.check_vllm_patches(_tree(tmp_path, vision=False, mtp=False), "text-mtp")
    )
    assert bad["MTP lm_head quant"].status == "FAIL"
    assert bad["MTP hidden-buffer device"].status == "WARN"
    assert bad["vision split qkv"].status == "SKIP"

    good = _by_name(
        mod.check_vllm_patches(_tree(tmp_path, vision=False, mtp=True), "text-mtp")
    )
    assert good["MTP lm_head quant"].status == "PASS"
    assert good["MTP hidden-buffer device"].status == "WARN"


def test_multimodal_profile_requires_vision_filter(tmp_path):
    bad = _by_name(
        mod.check_vllm_patches(
            _tree(tmp_path, vision=False, mtp=False), "multimodal-no-draft"
        )
    )
    assert bad["vision split qkv"].status == "FAIL"
    assert bad["MTP lm_head quant"].status == "SKIP"

    good = _by_name(
        mod.check_vllm_patches(
            _tree(tmp_path, vision=True, mtp=False), "multimodal-no-draft"
        )
    )
    assert good["vision split qkv"].status == "PASS"


def test_multimodal_mtp_requires_both(tmp_path):
    checks = _by_name(
        mod.check_vllm_patches(
            _tree(tmp_path, vision=True, mtp=True), "multimodal-mtp"
        )
    )
    assert checks["MTP lm_head quant"].status == "PASS"
    assert checks["vision split qkv"].status == "PASS"
