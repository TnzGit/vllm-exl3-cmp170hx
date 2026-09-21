#!/usr/bin/env python3
"""Submit one exact-token K0 shadow-retrieval case to the OpenAI endpoint."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path


def _post_json(url: str, payload: dict, timeout: int = 7200) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--case", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    case = json.loads(args.case.read_text())
    token_ids = case["prompt_token_ids"]
    base = f"http://127.0.0.1:{args.port}"
    model = json.load(
        urllib.request.urlopen(f"{base}/v1/models", timeout=30)
    )["data"][0]["id"]

    payload = {
        "model": model,
        "prompt": token_ids,
        "max_tokens": 1,
        "temperature": 0.0,
        "seed": 0,
        "stream": False,
    }
    t0 = time.time()
    response = _post_json(f"{base}/v1/completions", payload)
    wall = time.time() - t0
    usage = response.get("usage") or {}
    choice = (response.get("choices") or [{}])[0]

    targets = case.get("needles")
    if targets is None:
        targets = case.get("target_facts", [])
    target_tokens = case.get("target_tokens")
    if target_tokens is None:
        target_tokens = case.get("prompt_tokens")

    out = {
        "schema": 1,
        "case": case["name"],
        "target_tokens": target_tokens,
        "query_span": case["query_span"],
        "needles": targets,
        "wall_s": round(wall, 6),
        "usage": usage,
        "finish_reason": choice.get("finish_reason"),
        "text": choice.get("text"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
