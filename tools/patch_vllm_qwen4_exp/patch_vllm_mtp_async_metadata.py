"""Backport vLLM #55054: async Qwen4Exp PLE MTP metadata transfers.

Upstream PR:
  https://github.com/vllm-project/vllm/pull/55054

The vLLM 0.29.0 Qwen4Exp short-conv metadata builder already imports
``async_tensor_h2d``, but still uses synchronous CPU->GPU ``.to(device)``
for speculative/non-speculative request indices. In MTP this creates two
cudaStreamSynchronize calls per draft step.

This patch applies only the upstream metadata-transfer delta. It changes no
model math, request ordering, EXL3 dispatch, or sampling semantics.

Usage:
  python3 patch_vllm_mtp_async_metadata.py <site-packages/vllm>

The patch is fail-fast, idempotent, compile-checked, and preserves
<target>.orig on the first application.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


OLD_NEW = (
    (
        """        spec_req_idx = spec_req_idx_cpu.to(query_start_loc.device)
        non_spec_req_idx = non_spec_req_idx_cpu.to(query_start_loc.device)
""",
        """        spec_req_idx = async_tensor_h2d(spec_req_idx_cpu, device=query_start_loc.device)
        non_spec_req_idx: torch.Tensor | None = None
""",
    ),
    (
        """            req_group = torch.full(
                (m.num_reqs,),
                2,
                dtype=torch.int32,
                device=query_start_loc.device,
            )
            req_group[spec_req_idx] = 0
            req_group[decode_req_idx_cpu.to(query_start_loc.device)] = 1
""",
        """            non_spec_req_idx = async_tensor_h2d(
                non_spec_req_idx_cpu, device=query_start_loc.device
            )
            decode_req_idx = async_tensor_h2d(
                decode_req_idx_cpu, device=query_start_loc.device
            )
            req_group = torch.full(
                (m.num_reqs,),
                2,
                dtype=torch.int32,
                device=query_start_loc.device,
            )
            req_group[spec_req_idx] = 0
            req_group[decode_req_idx] = 1
""",
    ),
    (
        """        num_accepted_tokens = num_accepted_tokens[
            spec_req_idx_cpu.to(num_accepted_tokens.device)
        ]
""",
        """        num_accepted_tokens = num_accepted_tokens[spec_req_idx]
""",
    ),
    (
        """            if non_spec_req_idx_cpu is not None:
                non_spec_req_idx = non_spec_req_idx_cpu.to(num_computed_tokens.device)
                num_computed_tokens = num_computed_tokens[non_spec_req_idx]
""",
        """            assert non_spec_req_idx is not None
            num_computed_tokens = num_computed_tokens[non_spec_req_idx]
""",
    ),
)

PATCHED_MARKERS = (
    "spec_req_idx = async_tensor_h2d(spec_req_idx_cpu, device=query_start_loc.device)",
    "non_spec_req_idx: torch.Tensor | None = None",
    "decode_req_idx = async_tensor_h2d(",
    "num_accepted_tokens = num_accepted_tokens[spec_req_idx]",
    "assert non_spec_req_idx is not None",
)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: patch_vllm_mtp_async_metadata.py <site-packages/vllm>", file=sys.stderr)
        return 2

    root = Path(sys.argv[1]).resolve()
    path = root / "v1" / "attention" / "backends" / "short_conv_attn.py"
    if not path.is_file():
        print(f"ERROR: target not found: {path}", file=sys.stderr)
        return 2

    src = path.read_text(encoding="utf-8")

    if all(marker in src for marker in PATCHED_MARKERS):
        print(f"already patched: {path}")
        return 0

    if "from vllm.utils.torch_utils import async_tensor_h2d" not in src:
        print(
            "ERROR: vLLM source lacks async_tensor_h2d import; "
            "this backport is only validated against the v0.29 Qwen4Exp layout",
            file=sys.stderr,
        )
        return 1

    out = src
    for idx, (old, new) in enumerate(OLD_NEW, start=1):
        count = out.count(old)
        if count != 1:
            print(
                f"ERROR: #55054 anchor {idx} found {count} times in {path}; "
                "refusing a partial/ambiguous patch",
                file=sys.stderr,
            )
            return 1
        out = out.replace(old, new, 1)

    missing = [marker for marker in PATCHED_MARKERS if marker not in out]
    if missing:
        print(f"ERROR: postcondition markers missing: {missing}", file=sys.stderr)
        return 1

    compile(out, str(path), "exec")

    backup = path.with_suffix(path.suffix + ".orig")
    if not backup.exists():
        shutil.copyfile(path, backup)
    path.write_text(out, encoding="utf-8")
    print(f"patched {path} (upstream #55054 metadata-transfer delta; backup {backup})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
