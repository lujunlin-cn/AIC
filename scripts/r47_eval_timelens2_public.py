#!/usr/bin/env python3
"""Evaluate fixed TimeLens2 span output against the public human proxy.

This is a source-level diagnostic only.  The model spans are converted into a
binary per-second score, then the same exact 85% budget and a centre rank
tie-break used by R26 are applied.  No official labels are read.
"""
from __future__ import annotations
import argparse, csv, json, re
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.r44_local_state_residual import GRADE, proxy_f, select

def rows(paths):
    out=[]
    for stream,p in enumerate(paths):
        with p.open(newline="") as f: out.extend(dict(x) for x in csv.DictReader(f))
        for x in out:
            if "_stream" not in x: x["_stream"]=stream
    return out

def parse(text):
    m=re.search(r"\[\s*\[.*?\]\s*\]",text)
    if not m: return []
    try:
        a=json.loads(m.group(0)); return [(float(x[0]),float(x[1])) for x in a if len(x)>=2]
    except Exception: return []

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--labels",nargs=2,type=Path,required=True); ap.add_argument("--responses",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); args=ap.parse_args()
    rs=rows(args.labels); resp=json.loads(args.responses.read_text()); spans={x["video_id"]:parse(x["response"]) for x in resp["results"]}
    by={}
    for r in rs: by.setdefault(r["video_id"],[]).append(r)
    source=[]
    for v,items in sorted(by.items()):
        streams=sorted(set(int(x["_stream"]) for x in items))
        for stream in streams:
            ss=[x for x in items if int(x["_stream"])==stream]
            # The task ids are contiguous per exported stream; fall back to
            # the row order if a future export changes that numbering.
            times=np.array(sorted({int(float(x["time_sec"])) for x in ss}),dtype=int)
            y=np.array([GRADE[next(x["generic_highlight"] for x in ss if int(float(x["time_sec"]))==t)] for t in times],float)
            center=1-np.abs(times-14.5)/14.5
            inside=np.array([any(a<=t<=b for a,b in spans.get(v,[])) for t in times],float)
            k=int(round(.85*len(times)))
            # Binary span score plus deterministic centre tie-break.
            fused=inside*2.0+center*1e-3
            kl=select(fused,k); kc=select(center,k)
            source.append({"video_id":v,"stream":stream,"n":len(times),"center":proxy_f(kc,y,True),"timelens":proxy_f(kl,y,True),"delta":proxy_f(kl,y,True)-proxy_f(kc,y,True),"span_seconds":spans.get(v,[])})
    out={"experiment_id":"R47_TIMELENS2_PUBLIC_PROXY","status":"PUBLIC_SOURCE_DIAGNOSTIC","sources":len(by),"views":len(source),"macro_center":float(np.mean([x["center"] for x in source])),"macro_timelens":float(np.mean([x["timelens"] for x in source])),"macro_delta":float(np.mean([x["delta"] for x in source])),"positive_views":int(sum(x["delta"]>0 for x in source)),"source":source}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n"); print(json.dumps({k:out[k] for k in ("sources","views","macro_center","macro_timelens","macro_delta","positive_views")},ensure_ascii=False,indent=2))
if __name__=="__main__": main()
