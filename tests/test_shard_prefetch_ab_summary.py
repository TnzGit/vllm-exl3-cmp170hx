import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/"tools"/"r0_shard_prefetch_ab_summary.py"

def load():
    spec=importlib.util.spec_from_file_location("s",SCRIPT); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def row(i,arm,gib,wall,read_gib,faults):
    B=1024**3
    return {"index":i,"arm":arm,"file":"x","file_bytes":int(gib*B),"consume_wall_s":wall,"prefetch_wall_s":0.0,"prefetch_join_wall_s":0.0,"total_shard_wall_s":wall,"delta":{"majflt":faults,"minflt":10,"utime_s":1.0,"stime_s":0.2,"io":{"read_bytes":int(read_gib*B)}}}

def test_summary_compares_normalized_wall_faults_and_reads():
    m=load(); rows=[row(0,'control',2,4,1.5,200),row(1,'prefetch',2,2,1.0,50),row(2,'control',2,4,1.5,200),row(3,'prefetch',2,2,1.0,50)]
    out=m.summarize(rows,100.0)
    assert out["shard_prefetch_ab_valid"] is True
    assert out["balance"]["prefetch_byte_share"] == 0.5
    assert out["comparison"]["prefetch_vs_control_speedup"] == 2.0
    assert out["comparison"]["major_faults_per_gib_ratio"] == 0.25
    assert out["comparison"]["read_gib_per_file_gib_ratio"] == 2/3

def test_summary_rejects_unbalanced_bytes():
    m=load(); rows=[row(0,'control',9,9,1,10),row(1,'control',9,9,1,10),row(2,'prefetch',1,1,1,10),row(3,'prefetch',1,1,1,10)]
    out=m.summarize(rows,100.0)
    assert out["shard_prefetch_ab_valid"] is False
    assert out["balance"]["byte_balance_ok"] is False
