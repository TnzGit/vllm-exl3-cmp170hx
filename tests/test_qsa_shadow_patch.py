import subprocess
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "patch_vllm_qwen4_exp"
    / "patch_vllm_qsa_shadow.py"
)


QSA_029 = '''from __future__ import annotations

from typing import ClassVar, cast

import torch

from .indexer_qsa import QSAIndexer

class Owner:
    def _run_qsa(self, hidden_states, positions):
        num_tokens = 4
        side_metadata = type("M", (), {"logical_positions": torch.arange(4)})()
        selected = self.indexer(
            hidden_states,
            positions,
            self.topk_indices_buffer[:num_tokens],
        )
        if selected.shape != (
            num_tokens,
            self.indexer.output_width,
        ):
            raise RuntimeError("QSA indexer returned an invalid selection shape")
        return selected
'''


def _tree(tmp_path: Path, text: str = QSA_029) -> Path:
    root = tmp_path / "vllm"
    qsa = root / "models" / "qwen4_exp" / "nvidia" / "qsa.py"
    qsa.parent.mkdir(parents=True)
    qsa.write_text(text, encoding="utf-8")
    return root


def _run(root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(root), *extra],
        text=True,
        capture_output=True,
        check=False,
    )


def test_qsa_shadow_patch_exact_layout_and_idempotence(tmp_path):
    root = _tree(tmp_path)
    qsa = root / "models" / "qwen4_exp" / "nvidia" / "qsa.py"

    first = _run(root)
    assert first.returncode == 0, first.stdout + first.stderr
    src = qsa.read_text(encoding="utf-8")
    assert "# KVMEM_QSA_SHADOW_V1" in src
    assert "VLLM_QWEN_KVMEM_SHADOW_PATH" in src
    assert "VLLM_QWEN_KVMEM_SHADOW_MIN_POS" in src
    assert "torch.cuda.is_current_stream_capturing()" in src
    assert "selected=selected" in src
    assert "logical_positions=side_metadata.logical_positions[:num_tokens]" in src

    backup = qsa.with_suffix(".py.kvmem_qsa_shadow.orig")
    assert backup.read_text(encoding="utf-8") == QSA_029

    second = _run(root)
    assert second.returncode == 0, second.stdout + second.stderr
    assert "already patched" in second.stdout
    assert qsa.read_text(encoding="utf-8") == src
    assert backup.read_text(encoding="utf-8") == QSA_029


def test_qsa_shadow_patch_fails_closed_on_layout_drift(tmp_path):
    root = _tree(tmp_path, QSA_029.replace("QSA indexer returned an invalid selection shape", "changed"))
    run = _run(root)
    assert run.returncode != 0
    assert "anchor count" in run.stderr
    qsa = root / "models" / "qwen4_exp" / "nvidia" / "qsa.py"
    assert "# KVMEM_QSA_SHADOW_V1" not in qsa.read_text(encoding="utf-8")


def test_qsa_shadow_check_only_does_not_modify_source(tmp_path):
    root = _tree(tmp_path)
    qsa = root / "models" / "qwen4_exp" / "nvidia" / "qsa.py"
    before = qsa.read_bytes()

    run = _run(root, "--check-only")
    assert run.returncode == 0, run.stdout + run.stderr
    assert "no files changed" in run.stdout
    assert qsa.read_bytes() == before
    assert not qsa.with_suffix(".py.kvmem_qsa_shadow.orig").exists()
