#!/usr/bin/env python3
"""Cache full model identity hashes while trusted filesystem metadata is stable.

The metadata shortcut assumes a trusted local filesystem. It is not intended to
withstand a privileged actor who can forge inode timestamps or alter the proof.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
MATCHED_LOAD = HERE / "r0_c2_c4_mtp_matched_load.py"
SPEC = importlib.util.spec_from_file_location("r0_matched_load", MATCHED_LOAD)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

TOKENIZER_FILES = {
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "tokenizer.model", "spiece.model", "vocab.json",
    "vocab.txt", "merges.txt", "chat_template.jinja",
}
HASH_FIELDS = (
    "model_tree_sha256", "model_file_count", "model_bytes_hashed",
    "model_config_sha256", "tokenizer_tree_sha256", "tokenizer_file_count",
    "tokenizer_bytes_hashed", "manifest_sha256",
)


class CacheError(RuntimeError):
    """An unsafe asset, invalid proof, or failed manifest comparison."""


def _manifest_raw(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise CacheError(f"Manifest must be a regular non-symlink file: {path}")
    info = path.stat(follow_symlinks=False)
    if info.st_nlink != 1:
        raise CacheError(f"Manifest must not be hard-linked: {path}")
    doc, raw = runner._load_json(path, runner.MAX_MANIFEST_BYTES)
    if not isinstance(doc, dict):
        raise CacheError("Manifest root must be an object")
    return raw


def _metadata(path: Path) -> dict[str, int]:
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode):
        raise CacheError(f"Non-regular file in model tree: {path}")
    if info.st_nlink != 1:
        raise CacheError(f"Hard-linked model file is not allowed: {path}")
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
    }


def snapshot(model_path: Path, manifest_raw: bytes) -> dict[str, Any]:
    if model_path.is_symlink():
        raise CacheError(f"Model directory must not be a symlink: {model_path}")
    root = model_path.resolve(strict=True)
    if not root.is_dir():
        raise CacheError(f"Model path is not a directory: {model_path}")

    def walk_error(error: OSError) -> None:
        raise CacheError(f"Cannot enumerate model tree: {error}") from error

    files: dict[str, dict[str, int]] = {}
    for current, directories, names in os.walk(
        root, topdown=True, onerror=walk_error, followlinks=False
    ):
        parent = Path(current)
        for name in directories:
            item = parent / name
            if item.is_symlink() or not item.is_dir():
                raise CacheError(f"Unsafe directory in model tree: {item}")
        for name in names:
            item = parent / name
            if item.is_symlink():
                raise CacheError(f"Symlink in model tree: {item}")
            relative = item.relative_to(root).as_posix()
            files[relative] = _metadata(item)

    if "config.json" not in files:
        raise CacheError("Model tree has no root config.json")
    if not any(Path(name).name in TOKENIZER_FILES for name in files):
        raise CacheError("Model tree has no recognized tokenizer assets")
    return {
        "model_path": str(root),
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "files": files,
    }


def _hash_files(root: Path, relative_names: list[str]) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    count = 0
    total = 0
    for relative in sorted(relative_names):
        path = root / relative
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                total += len(chunk)
        count += 1
    return digest.hexdigest(), count, total


def _full_hashes(root: Path, identity: dict[str, Any]) -> dict[str, Any]:
    files = identity["files"]
    all_names = sorted(files)
    tokenizer_names = sorted(
        name for name in all_names if Path(name).name in TOKENIZER_FILES
    )
    model_hash, model_count, model_bytes = _hash_files(root, all_names)
    tokenizer_hash, tokenizer_count, tokenizer_bytes = _hash_files(root, tokenizer_names)
    config_hash = hashlib.sha256()
    with (root / "config.json").open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            config_hash.update(chunk)
    return {
        "model_tree_sha256": model_hash,
        "model_file_count": model_count,
        "model_bytes_hashed": model_bytes,
        "model_config_sha256": config_hash.hexdigest(),
        "tokenizer_tree_sha256": tokenizer_hash,
        "tokenizer_file_count": tokenizer_count,
        "tokenizer_bytes_hashed": tokenizer_bytes,
        "manifest_sha256": identity["manifest_sha256"],
    }


def _compare_manifest(hashes: dict[str, Any], manifest: dict[str, Any]) -> None:
    expected = manifest["provenance"]
    for name, actual in (
        ("model_pack_sha256", hashes["model_tree_sha256"]),
        ("tokenizer_sha256", hashes["tokenizer_tree_sha256"]),
        ("model_config_sha256", hashes["model_config_sha256"]),
    ):
        if expected.get(name) != actual:
            raise CacheError(f"Actual {name} differs from frozen manifest")


def _cached_hashes_valid(hashes: Any, identity: dict[str, Any]) -> bool:
    if not isinstance(hashes, dict) or set(hashes) != set(HASH_FIELDS):
        return False
    for name in (
        "model_tree_sha256", "model_config_sha256", "tokenizer_tree_sha256",
        "manifest_sha256",
    ):
        value = hashes.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(
            char not in "0123456789abcdef" for char in value
        ):
            return False
    if hashes["manifest_sha256"] != identity["manifest_sha256"]:
        return False
    for name in (
        "model_file_count", "model_bytes_hashed", "tokenizer_file_count",
        "tokenizer_bytes_hashed",
    ):
        if type(hashes.get(name)) is not int or hashes[name] < 0:
            return False
    files = identity["files"]
    tokenizer_names = [
        name for name in files if Path(name).name in TOKENIZER_FILES
    ]
    return (
        hashes["model_file_count"] == len(files)
        and hashes["model_bytes_hashed"] == sum(item["size"] for item in files.values())
        and hashes["tokenizer_file_count"] == len(tokenizer_names)
        and hashes["tokenizer_bytes_hashed"]
        == sum(files[name]["size"] for name in tokenizer_names)
    )


def _read_proof(path: Path) -> dict[str, Any] | None:
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        raise CacheError(f"Cache proof must be a regular non-symlink file: {path}")
    info = path.stat(follow_symlinks=False)
    if info.st_nlink != 1:
        raise CacheError(f"Cache proof must not be hard-linked: {path}")
    try:
        size = info.st_size
        if size <= 0 or size > 4 * 1024 * 1024:
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temp_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _write_output(path: Path, hashes: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(hashes, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def verify_or_reuse(
    model_path: Path,
    manifest_path: Path,
    cache_path: Path,
    output_path: Path,
    expected_commit: str,
) -> str:
    manifest, raw = runner.load_prompt_manifest(manifest_path)
    if manifest["provenance"]["r0_source_commit"] != expected_commit:
        raise CacheError("Manifest r0_source_commit differs from EXPECTED_SHA")
    if _manifest_raw(manifest_path) != raw:
        raise CacheError("Manifest changed while it was being loaded")

    current = snapshot(model_path, raw)
    root = Path(current["model_path"])
    cached = _read_proof(cache_path)
    if (
        cached is not None
        and cached.get("schema") == 1
        and cached.get("proof") == "full_model_tree_hashes_verified"
        and cached.get("identity") == current
        and _cached_hashes_valid(cached.get("hashes"), current)
    ):
        hashes = cached["hashes"]
        _compare_manifest(hashes, manifest)
        mode = "cached"
    else:
        before = current
        hashes = _full_hashes(root, before)
        after_raw = _manifest_raw(manifest_path)
        after = snapshot(root, after_raw)
        if after_raw != raw or after != before:
            raise CacheError("Model metadata or manifest changed during full hash verification")
        _compare_manifest(hashes, manifest)
        _atomic_json(cache_path, {
            "schema": 1,
            "proof": "full_model_tree_hashes_verified",
            "verified_utc": datetime.now(timezone.utc).isoformat(),
            "identity": after,
            "hashes": hashes,
        })
        mode = "full_hash"

    _write_output(output_path, hashes)
    return mode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-r0-source-commit", required=True)
    args = parser.parse_args(argv)
    try:
        if len(args.expected_r0_source_commit) != 40 or any(
            char not in "0123456789abcdef" for char in args.expected_r0_source_commit
        ):
            raise CacheError("EXPECTED_SHA must be 40 lowercase hexadecimal characters")
        mode = verify_or_reuse(
            args.model,
            args.manifest,
            args.cache,
            args.output,
            args.expected_r0_source_commit,
        )
        print(f"model_identity_{mode}_sha256_proof_valid")
        return 0
    except (OSError, ValueError, TypeError, KeyError, CacheError) as exc:
        print(f"model identity cache: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
