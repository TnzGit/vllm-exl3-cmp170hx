#!/usr/bin/env python3
"""Record the R0 runtime identity as machine-readable JSON.

Prints python/torch/CUDA/driver/vLLM/ExLlamaV3/plugin versions, GPU geometry,
host RAM and filesystem facts. Read-only; writes nothing.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def _ver(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "<not-installed>"


def _out(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "<unavailable>"


def _git_head(path: str | Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except Exception:
        return "<not-a-git-checkout>"


def main() -> int:
    ident: dict = {
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
        },
        "platform": {
            "machine": platform.machine(),
            "release": platform.release(),
            "libc": platform.libc_ver()[0],
        },
    }

    try:
        import torch

        ident["torch"] = {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": str(torch.backends.cudnn.version()),
            "device_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            ident["gpu"] = {
                "name": props.name,
                "compute_capability": [props.major, props.minor],
                "vram_bytes": props.total_memory,
                "vram_gib": props.total_memory / 2**30,
                "sm_count": props.multi_processor_count,
                "reserved_gib": torch.cuda.memory_reserved(idx) / 2**30,
            }
    except Exception as exc:  # pragma: no cover
        ident["torch"] = {"error": repr(exc)}

    ident["driver"] = _out(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    ident["nvidia_smi_version"] = _out(
        ["bash", "-lc", "nvidia-smi --version | head -1"]
    )

    ident["vllm"] = {"version": _ver("vllm")}
    try:
        import vllm

        ident["vllm"]["file"] = vllm.__file__
    except Exception as exc:
        ident["vllm"]["error"] = repr(exc)

    ident["exllamav3"] = {"version": _ver("exllamav3")}
    try:
        import exllamav3
        from exllamav3.version import __version__ as ev3

        ident["exllamav3"]["module_version"] = ev3
        ident["exllamav3"]["file"] = exllamav3.__file__
    except Exception as exc:
        ident["exllamav3"]["error"] = repr(exc)

    # Extension ABI / cooperative-path symbols come from the compiled ext.
    try:
        import torch  # noqa: F401
        import exllamav3_ext

        ident["exllamav3_ext"] = {
            "file": exllamav3_ext.__file__,
            "has_exl3_moe_coop": hasattr(exllamav3_ext, "exl3_moe_coop"),
            "has_exl3_moe": hasattr(exllamav3_ext, "exl3_moe"),
        }
    except Exception as exc:
        ident["exllamav3_ext"] = {"error": repr(exc)}

    ident["vllm_exl3"] = {"version": _ver("vllm-exl3")}
    for env in ("VLLM_EXL3_PLUGIN_SRC", "HOME"):
        pass
    try:
        import vllm_exl3

        ident["vllm_exl3"]["file"] = vllm_exl3.__file__
    except Exception as exc:
        ident["vllm_exl3"]["error"] = repr(exc)
    try:
        import vllm_exl3_c

        ident["vllm_exl3_c"] = {"file": vllm_exl3_c.__file__}
    except Exception as exc:
        ident["vllm_exl3_c"] = {"error": repr(exc)}

    here = Path(__file__).resolve()
    for label, cand in (
        ("plugin", here.parents[1]),
        ("exllamav3_source", Path.home() / "src" / "exllamav3-r0"),
    ):
        if cand.is_dir():
            ident.setdefault("git", {})[label] = {
                "path": str(cand),
                "head": _git_head(cand),
            }

    ident["host"] = {
        "ram_gib": round(
            int(_out(["bash", "-lc", "free -b | awk '/Mem:/ {print $2}'"])) / 2**30, 2
        )
        if _out(["bash", "-lc", "free -b | awk '/Mem:/ {print $2}'"]).isdigit()
        else None,
        "nproc": os.cpu_count(),
    }
    for mount in ("/", "/tmp", str(Path.home())):
        try:
            st = shutil.disk_usage(mount)
            ident.setdefault("filesystem", {})[mount] = {
                "total_gib": round(st.total / 2**30, 1),
                "free_gib": round(st.free / 2**30, 1),
            }
        except Exception:
            pass

    print(json.dumps(ident, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
