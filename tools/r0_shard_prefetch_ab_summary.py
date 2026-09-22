#!/usr/bin/env python3
"""Summarize one-boot safetensors shard-prefetch A/B evidence."""

from __future__ import annotations

import argparse, json
from pathlib import Path

GIB=1024**3

def load_rows(path: Path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

def agg(rows):
    out={"count":len(rows)}
    out["file_bytes"]=sum(int(r["file_bytes"]) for r in rows)
    out["file_gib"]=out["file_bytes"]/GIB
    out["consume_wall_s"]=sum(float(r["consume_wall_s"]) for r in rows)
    out["total_shard_wall_s"]=sum(float(r["total_shard_wall_s"]) for r in rows)
    out["prefetch_wall_s"]=sum(float(r["prefetch_wall_s"]) for r in rows)
    out["prefetch_join_wall_s"]=sum(float(r["prefetch_join_wall_s"]) for r in rows)
    out["read_bytes"]=sum(int((r.get("delta") or {}).get("io",{}).get("read_bytes",0)) for r in rows)
    out["read_gib"]=out["read_bytes"]/GIB
    out["major_faults"]=sum(int((r.get("delta") or {}).get("majflt",0)) for r in rows)
    out["minor_faults"]=sum(int((r.get("delta") or {}).get("minflt",0)) for r in rows)
    out["user_cpu_s"]=sum(float((r.get("delta") or {}).get("utime_s",0.0)) for r in rows)
    out["system_cpu_s"]=sum(float((r.get("delta") or {}).get("stime_s",0.0)) for r in rows)
    out["consume_s_per_gib"]=out["consume_wall_s"]/out["file_gib"] if out["file_gib"] else None
    out["total_s_per_gib"]=out["total_shard_wall_s"]/out["file_gib"] if out["file_gib"] else None
    out["major_faults_per_gib"]=out["major_faults"]/out["file_gib"] if out["file_gib"] else None
    out["read_gib_per_file_gib"]=out["read_gib"]/out["file_gib"] if out["file_gib"] else None
    return out

def summarize(rows, main_weights_s):
    control=[r for r in rows if r["arm"]=="control"]
    prefetch=[r for r in rows if r["arm"]=="prefetch"]
    c=agg(control); p=agg(prefetch)
    total_bytes=c["file_bytes"]+p["file_bytes"]
    share=p["file_bytes"]/total_bytes if total_bytes else None
    valid=bool(c["count"]>=2 and p["count"]>=2 and share is not None and 0.40<=share<=0.60)
    speedup=(c["total_s_per_gib"]/p["total_s_per_gib"] if c["total_s_per_gib"] and p["total_s_per_gib"] else None)
    return {
      "schema":1,
      "shard_prefetch_ab_valid":valid,
      "main_weights_s":main_weights_s,
      "control":c, "prefetch":p,
      "balance":{"prefetch_byte_share":share,"byte_balance_ok": bool(share is not None and 0.40<=share<=0.60)},
      "comparison":{
        "prefetch_vs_control_speedup":speedup,
        "prefetch_total_s_per_gib_ratio": (p["total_s_per_gib"]/c["total_s_per_gib"] if c["total_s_per_gib"] and p["total_s_per_gib"] is not None else None),
        "major_faults_per_gib_ratio": (p["major_faults_per_gib"]/c["major_faults_per_gib"] if c["major_faults_per_gib"] and p["major_faults_per_gib"] is not None else None),
        "read_gib_per_file_gib_ratio": (p["read_gib_per_file_gib"]/c["read_gib_per_file_gib"] if c["read_gib_per_file_gib"] and p["read_gib_per_file_gib"] is not None else None),
      },
      "per_shard":rows,
      "interpretation_contract":{
        "single_boot_interleaved":True,
        "prefetch_primitive":"vllm._prefetch_checkpoint",
        "prefetch_scope":"same shard only; thread joined before next shard",
        "projection_is_not_qualification":True,
      },
    }

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--stats",type=Path,required=True); ap.add_argument("--startup",type=Path,required=True); ap.add_argument("--out",type=Path,required=True); a=ap.parse_args()
    startup=json.loads(a.startup.read_text()); out=summarize(load_rows(a.stats), startup["timings"]["main_weights_s"])
    a.out.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2)); return 0 if out["shard_prefetch_ab_valid"] else 3

if __name__=="__main__": raise SystemExit(main())
