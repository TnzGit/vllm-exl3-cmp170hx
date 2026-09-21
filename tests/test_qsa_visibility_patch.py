import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "patch_vllm_qwen4_exp" / "patch_vllm_qsa_visibility.py"


def _load():
    spec = importlib.util.spec_from_file_location("q1_patch", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def _source():
    return '''from __future__ import annotations

import torch

from .indexer_qsa import QSAIndexer

class Owner:
    def run(self, selected, num_tokens, side_metadata):
        if selected.shape != (
            num_tokens,
            self.indexer.output_width,
        ):
            raise RuntimeError("QSA indexer returned an invalid selection shape")
        return selected
'''


def test_visibility_patch_is_default_off_and_has_expected_call(tmp_path):
    mod = _load()
    target = tmp_path / "qsa.py"
    target.write_text(_source())

    status = mod.patch(target)
    assert status == "patched"
    out = target.read_text()
    assert mod.MARKER in out
    assert 'VLLM_QWEN_KVMEM_RESIDENT_PLAN' in out
    assert 'VLLM_QWEN_KVMEM_VISIBILITY_STATS_PATH' in out
    assert 'selected.masked_fill_(drop, -1)' in out
    assert 'logical_positions=side_metadata.logical_positions[:num_tokens]' in out
    assert 'torch.isin' in out
    compile(out, str(target), "exec")


def test_visibility_check_only_does_not_modify_source(tmp_path):
    mod = _load()
    target = tmp_path / "qsa.py"
    before = _source()
    target.write_text(before)
    status = mod.patch(target, check_only=True)
    assert "no files changed" in status
    assert target.read_text() == before


def test_visibility_patch_never_changes_qsa_scoring_call():
    src = SCRIPT.read_text()
    assert "self.indexer(" not in src
    assert "persistent_topk" not in src
    assert "qsa_sparse_paged_attention" not in src
