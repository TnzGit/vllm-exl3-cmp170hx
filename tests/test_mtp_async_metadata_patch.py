import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "tools"
    / "patch_vllm_qwen4_exp"
    / "patch_vllm_mtp_async_metadata.py"
)


SHORT_CONV_029 = """from typing import Any
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
                dtype=torch.int64,
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


def _tree(tmp_path: Path, text: str = SHORT_CONV_029) -> tuple[Path, Path]:
    root = tmp_path / "vllm"
    target = root / "v1" / "attention" / "backends" / "short_conv_attn.py"
    target.parent.mkdir(parents=True)
    target.write_text(text)
    return root, target


def _run(root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(root), *extra],
        text=True,
        capture_output=True,
        check=False,
    )


def test_async_metadata_backport_applies_exact_029_delta_and_is_idempotent(tmp_path):
    root, target = _tree(tmp_path)

    first = _run(root)
    assert first.returncode == 0, first.stdout + first.stderr
    src = target.read_text()

    assert "spec_req_idx = async_tensor_h2d(spec_req_idx_cpu, device=query_start_loc.device)" in src
    assert "non_spec_req_idx: torch.Tensor | None = None" in src
    assert "decode_req_idx = async_tensor_h2d(" in src
    assert "req_group[decode_req_idx] = 1" in src
    assert "num_accepted_tokens = num_accepted_tokens[spec_req_idx]" in src
    assert "assert non_spec_req_idx is not None" in src

    assert "spec_req_idx_cpu.to(query_start_loc.device)" not in src
    assert "decode_req_idx_cpu.to(query_start_loc.device)" not in src
    assert "spec_req_idx_cpu.to(num_accepted_tokens.device)" not in src
    assert "non_spec_req_idx_cpu.to(num_computed_tokens.device)" not in src

    backup = target.with_suffix(target.suffix + ".orig")
    assert backup.is_file()
    assert backup.read_text() == SHORT_CONV_029

    second = _run(root)
    assert second.returncode == 0, second.stdout + second.stderr
    assert "already patched:" in second.stdout


def test_async_metadata_backport_fails_closed_on_layout_drift(tmp_path):
    broken = SHORT_CONV_029.replace(
        "spec_req_idx = spec_req_idx_cpu.to(query_start_loc.device)",
        "spec_req_idx = some_future_helper(spec_req_idx_cpu)",
    )
    root, target = _tree(tmp_path, broken)
    before = target.read_text()

    run = _run(root)
    assert run.returncode != 0
    assert "anchor 1 found 0 times" in run.stderr
    assert target.read_text() == before
    assert not target.with_suffix(target.suffix + ".orig").exists()


def test_async_metadata_backport_requires_existing_async_helper_import(tmp_path):
    broken = SHORT_CONV_029.replace(
        "from vllm.utils.torch_utils import async_tensor_h2d\n", ""
    )
    root, target = _tree(tmp_path, broken)
    before = target.read_text()

    run = _run(root)
    assert run.returncode != 0
    assert "lacks async_tensor_h2d import" in run.stderr
    assert target.read_text() == before


def test_async_metadata_backport_preserves_req_group_dtype(tmp_path):
    # Upstream vLLM v0.29.0 uses int64 here. The backport must not rewrite
    # this unrelated dtype while inserting the async request-index transfers.
    root, target = _tree(tmp_path, SHORT_CONV_029)
    run = _run(root)
    assert run.returncode == 0, run.stdout + run.stderr
    src = target.read_text()
    assert "dtype=torch.int64" in src
    assert "dtype=torch.int32" not in src


def test_async_metadata_backport_also_accepts_legacy_int32_layout(tmp_path):
    legacy = SHORT_CONV_029.replace("dtype=torch.int64", "dtype=torch.int32")
    root, target = _tree(tmp_path, legacy)
    run = _run(root)
    assert run.returncode == 0, run.stdout + run.stderr
    src = target.read_text()
    assert "dtype=torch.int32" in src
    assert "req_group[decode_req_idx] = 1" in src


def test_async_metadata_backport_check_only_validates_without_writing(tmp_path):
    root, target = _tree(tmp_path, SHORT_CONV_029)
    before = target.read_text()
    run = _run(root, "--check-only")
    assert run.returncode == 0, run.stdout + run.stderr
    assert "no files changed" in run.stdout
    assert target.read_text() == before
    assert not target.with_suffix(target.suffix + ".orig").exists()
