#!/usr/bin/env python3
"""Apply the minimum Qwen4Exp EXL3 vLLM patch stack for one service profile.

usage:
  python3 tools/apply_qwen4_exp_patches.py <site-packages/vllm> --profile PROFILE

Profiles:
  text-no-draft     main lm_head + PLE only
  text-mtp          main lm_head + PLE + MTP lm_head
  multimodal-no-draft
                    main lm_head + PLE + vision split-qkv filtering
  multimodal-mtp    all three patch scripts

The wrapper is fail-fast and verifies postconditions. It never claims a
partially applied patch stack succeeded.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROFILE_PATCHES = {
    "text-no-draft": ("patch_vllm_qwen4_ple.py",),
    "text-mtp": (
        "patch_vllm_qwen4_ple.py",
        "patch_vllm_mtp_lmhead.py",
    ),
    "multimodal-no-draft": (
        "patch_vllm_qwen4_ple.py",
        "patch_vllm_vision_split.py",
    ),
    "multimodal-mtp": (
        "patch_vllm_qwen4_ple.py",
        "patch_vllm_mtp_lmhead.py",
        "patch_vllm_vision_split.py",
    ),
}


def _block_has(text: str, start: str, required: tuple[str, ...], span: int = 1200) -> bool:
    i = text.find(start)
    if i < 0:
        return False
    block = text[i : i + span]
    return all(token in block for token in required)


def _verify(vllm_root: Path, profile: str) -> list[str]:
    qwen = vllm_root / "models" / "qwen4_exp" / "nvidia"
    model = (qwen / "model.py").read_text(encoding="utf-8")
    ple = (qwen / "ple_layer.py").read_text(encoding="utf-8")
    mtp = (qwen / "mtp.py").read_text(encoding="utf-8")
    errors: list[str] = []

    if not _block_has(
        model,
        "self.lm_head = ParallelLMHead(",
        ("quant_config=self.quant_config",),
    ):
        errors.append("main lm_head does not receive quant_config=self.quant_config")

    if not _block_has(
        ple,
        "self.ngram_embedding = PLEVocabParallelEmbedding(",
        (
            "quant_config=quant_config",
            "quant_method=_get_ple_embedding_quant_method(",
        ),
    ):
        errors.append(
            "PLE n-gram constructor does not preserve quant_method and pass quant_config"
        )

    if profile.endswith("-mtp") and not _block_has(
        mtp,
        "self.lm_head = ParallelLMHead(",
        ("quant_config=self.quant_config",),
    ):
        errors.append("MTP lm_head does not receive quant_config=self.quant_config")

    if profile.startswith("multimodal-"):
        for token in (
            '".attn.q_proj.": None',
            '".attn.k_proj.": None',
            '".attn.v_proj.": None',
        ):
            if token not in model:
                errors.append(f"vision split-qkv drop rule missing: {token}")

    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("vllm_root", type=Path)
    ap.add_argument("--profile", choices=sorted(PROFILE_PATCHES), required=True)
    args = ap.parse_args()

    vllm_root = args.vllm_root.resolve()
    qwen = vllm_root / "models" / "qwen4_exp" / "nvidia"
    if not qwen.is_dir():
        print(f"ERROR: Qwen4Exp vLLM source not found: {qwen}", file=sys.stderr)
        return 2

    patch_dir = Path(__file__).resolve().parent / "patch_vllm_qwen4_exp"
    for name in PROFILE_PATCHES[args.profile]:
        script = patch_dir / name
        if not script.is_file():
            print(f"ERROR: patch script missing: {script}", file=sys.stderr)
            return 2
        print(f"[patch:{args.profile}] {name}", flush=True)
        result = subprocess.run(
            [sys.executable, str(script), str(vllm_root)],
            text=True,
            check=False,
        )
        if result.returncode != 0:
            print(
                f"ERROR: {name} failed with rc={result.returncode}; "
                "refusing to continue the patch stack",
                file=sys.stderr,
            )
            return result.returncode or 1

    errors = _verify(vllm_root, args.profile)
    if errors:
        print("ERROR: patch scripts returned success but postconditions failed:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(
        "PASS: Qwen4Exp EXL3 patch stack applied and postconditions verified "
        f"for profile={args.profile}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
