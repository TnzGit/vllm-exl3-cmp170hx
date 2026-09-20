#!/usr/bin/env python3
"""Read-only preflight for the CMP170HX Qwen3.8-Flash-Next EXL3 runtime.

This script does not load a model. It records the installed runtime identity,
checks the Qwen4Exp EXL3 plumbing in the installed vLLM tree, registers the
plugin explicitly, and prints plugin diagnostics.

Exit status is non-zero only for contracts that must be fixed before a model
boot. Missing/other GPUs are warnings so the script can also run in CI.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple


class Check(NamedTuple):
    status: str
    name: str
    detail: str


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "<not-installed>"


def _git_head(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
    except Exception:
        return "<not-a-git-checkout>"


def _find_vllm_dir() -> Path | None:
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).resolve().parent


def _block_has(
    text: str, start: str, required: tuple[str, ...], span: int = 900
) -> bool:
    i = text.find(start)
    if i < 0:
        return False
    block = text[i : i + span]
    return all(token in block for token in required)


def check_vllm_patches(vllm_dir: Path, profile: str) -> list[Check]:
    root = vllm_dir / "models" / "qwen4_exp" / "nvidia"
    model = root / "model.py"
    ple = root / "ple_layer.py"
    mtp = root / "mtp.py"
    missing = [str(p) for p in (model, ple, mtp) if not p.exists()]
    if missing:
        return [Check("FAIL", "qwen4_exp source", f"missing: {missing}")]

    model_src = model.read_text()
    ple_src = ple.read_text()
    mtp_src = mtp.read_text()

    checks = [
        Check(
            "PASS"
            if _block_has(
                model_src,
                "self.lm_head = ParallelLMHead(",
                ("quant_config=self.quant_config",),
            )
            else "FAIL",
            "main lm_head quant",
            "ParallelLMHead must receive quant_config=self.quant_config",
        ),
        Check(
            "PASS"
            if _block_has(
                ple_src,
                "self.ngram_embedding = PLEVocabParallelEmbedding(",
                (
                    "quant_config=quant_config",
                    "quant_method=_get_ple_embedding_quant_method(",
                ),
            )
            else "FAIL",
            "PLE quant plumbing",
            (
                "vLLM 0.29 must keep Qwen's preselected FP8 quant_method and also "
                "pass quant_config so EXL3 can fall through to Exl3EmbeddingMethod"
            ),
        ),
        Check(
            (
                "PASS"
                if _block_has(
                    mtp_src,
                    "self.lm_head = ParallelLMHead(",
                    ("quant_config=self.quant_config",),
                    span=1200,
                )
                else "FAIL"
            )
            if profile.endswith("-mtp")
            else "SKIP",
            "MTP lm_head quant",
            (
                "required for MTP profiles"
                if profile.endswith("-mtp")
                else "not required for no-draft profile"
            ),
        ),
        Check(
            (
                "PASS"
                if all(
                    token in model_src
                    for token in (
                        '.attn.q_proj.": None',
                        '.attn.k_proj.": None',
                        '.attn.v_proj.": None',
                    )
                )
                else "FAIL"
            )
            if profile.startswith("multimodal-")
            else "SKIP",
            "vision split qkv",
            (
                "required for multimodal profiles"
                if profile.startswith("multimodal-")
                else "not required for language-model-only profile"
            ),
        ),
    ]
    return checks


def check_gpu() -> Check:
    try:
        import torch
    except Exception as exc:
        return Check("WARN", "GPU", f"torch unavailable: {exc}")

    if not torch.cuda.is_available():
        return Check(
            "WARN",
            "GPU",
            f"torch {torch.__version__}, CUDA {torch.version.cuda}, CUDA device unavailable",
        )

    try:
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        cc = torch.cuda.get_device_capability(idx)
        total = torch.cuda.get_device_properties(idx).total_memory / 2**30
        status = "PASS" if cc == (8, 0) else "WARN"
        return Check(
            status,
            "GPU",
            (
                f"{name}; cc={cc[0]}.{cc[1]}; VRAM={total:.2f} GiB; "
                f"torch={torch.__version__}; CUDA={torch.version.cuda}"
            ),
        )
    except Exception as exc:
        return Check("WARN", "GPU", f"CUDA probe failed: {exc}")


def plugin_diagnostics() -> tuple[Check, dict]:
    try:
        import vllm_exl3

        vllm_exl3.register()
        diag = vllm_exl3.runtime_diagnostics()
        return (
            Check(
                "PASS",
                "vllm-exl3",
                "registered; runtime_diagnostics available",
            ),
            diag,
        )
    except Exception as exc:
        return Check(
            "FAIL", "vllm-exl3", f"{type(exc).__name__}: {exc}"
        ), {}


def exllamav3_check() -> Check:
    try:
        import exllamav3_ext
    except Exception as exc:
        return Check("FAIL", "exllamav3_ext", f"not importable: {exc}")

    symbols = {
        name: hasattr(exllamav3_ext, name)
        for name in (
            "exl3_moe",
            "exl3_moe_coop",
            "ngram_dequant",
            "reconstruct",
        )
    }
    must = (
        symbols["exl3_moe"]
        and symbols["ngram_dequant"]
        and symbols["reconstruct"]
    )
    return Check(
        "PASS" if must else "FAIL",
        "exllamav3_ext",
        f"version={_version('exllamav3')}; symbols={json.dumps(symbols, sort_keys=True)}",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--profile",
        choices=(
            "text-no-draft",
            "text-mtp",
            "multimodal-no-draft",
            "multimodal-mtp",
        ),
        default="text-no-draft",
    )
    args = ap.parse_args()
    profile = args.profile

    print("CMP170HX Qwen3.8-Flash-Next EXL3 preflight")
    print(f"profile={profile}")
    print(f"python={sys.version.split()[0]} platform={platform.platform()}")
    print(
        "packages="
        + json.dumps(
            {
                "vllm": _version("vllm"),
                "vllm-exl3": _version("vllm-exl3"),
                "exllamav3": _version("exllamav3"),
                "torch": _version("torch"),
            },
            sort_keys=True,
        )
    )

    checks: list[Check] = [check_gpu(), exllamav3_check()]

    vllm_dir = _find_vllm_dir()
    if vllm_dir is None:
        checks.append(Check("FAIL", "vLLM", "package is not importable"))
    else:
        checks.append(
            Check(
                "PASS",
                "vLLM",
                f"path={vllm_dir}; version={_version('vllm')}",
            )
        )
        checks.extend(check_vllm_patches(vllm_dir, profile))

    plugin_check, diag = plugin_diagnostics()
    checks.append(plugin_check)

    repo_root = Path(__file__).resolve().parents[1]
    print(f"plugin_checkout_head={_git_head(repo_root)}")
    print(
        "env="
        + json.dumps(
            {
                key: os.environ.get(key)
                for key in (
                    "VLLM_EXL3_NGRAM_TABLE",
                    "VLLM_EXL3_NGRAM_KERNEL",
                    "VLLM_EXL3_COOP",
                    "VLLM_EXL3_MOE_KERNEL",
                    "VLLM_EXL3_FAT_THRESHOLD",
                )
            },
            sort_keys=True,
        )
    )

    for c in checks:
        print(f"{c.status:4}  {c.name:22}  {c.detail}")

    if diag:
        print("runtime_diagnostics=")
        print(json.dumps(diag, indent=2, sort_keys=True, default=str))

    table_mode = os.environ.get("VLLM_EXL3_NGRAM_TABLE", "resident")
    if table_mode == "disk":
        print(
            "NOTE  disk n-gram requires PIECEWISE CUDA graphs and "
            "vllm::exl3_ngram_lookup_out in splitting_ops; verify the worker-side "
            "effective compilation config after the server starts."
        )

    hard_fail = any(c.status == "FAIL" for c in checks)
    return 1 if hard_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
