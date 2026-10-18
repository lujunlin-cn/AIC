#!/usr/bin/env python3
"""Source-held-out proxy for the R58 TimeLens multi-view residual.

Uses only the two human-labelled public source CSVs and the frozen compact
TimeLens interval report.  It reports paired macro F against centre ranking;
it never reads official test labels.
"""
from __future__ import annotations
import argparse, csv, json, re
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.r44_local_state_residual import GRADE, proxy_f, select

SPAN_RE = re.compile(r"\[\s*\[.*?\]\s*\]")
ETA_INSIDE, ETA_EDGE = .20, .50

def parse(text):
    m = SPAN_RE.search(text or "")
    if not m: return []
    try: a = json.loads(m.group(0))
    except Exception: return []
    return [(float(z[0]), float(z[1])) for z in a if len(z) >= 2]

def rank01(x):
    if len(x) <= 1: return np.zeros(len(x), float)
    return np.argsort(np.argsort(x, kind='mergesort'), kind='mergesort') / (len(x)-1)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--labels',nargs=2,type=Path,required=True)
    ap.add_argument('--responses',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    labels=[]
    for stream,p in enumerate(a.labels):
        with p.open(newline='') as f:
            for r in csv.DictReader(f): r['_stream']=stream; labels.append(r)
    raw=json.loads(a.responses.read_text())
    spans={(str(r['video_id']),0):parse(r.get('response','')) for r in raw.get('results',[])}
    # Compact public reports retain stream-aligned spans in ``source``.
    for r in raw.get('source',[]): spans[(str(r['video_id']),int(r['stream']))]=[tuple(z) for z in r.get('span_seconds',[])]
    by={}
    for r in labels: by.setdefault((r['video_id'],int(r['_stream'])),[]).append(r)
    vals=[]
    for (vid,stream),rs in sorted(by.items()):
        times=np.asarray(sorted({int(float(r['time_sec'])) for r in rs}),int)
        y=np.asarray([GRADE[next(r['generic_highlight'] for r in rs if int(float(r['time_sec']))==t)] for t in times],float)
        n=len(times); k=int(round(.85*n)); c=1-np.abs(times-14.5)/14.5
        ins=np.asarray([any(x<=t<=z for x,z in spans.get((vid,stream),[])) for t in times],float)
        edge=np.zeros(n,float)
        for x,z in spans.get((vid,stream),[]):
            m=(times>=x)&(times<=z); ph=np.clip((times-x)/max(z-x,1e-6),0,1)
            edge=np.maximum(edge,np.where(m,2*np.abs(ph-.5),0.0))
        base=proxy_f(select(c,k),y,True)
        score=rank01(c)+ETA_INSIDE*(rank01(ins)-.5)+ETA_EDGE*(rank01(edge)-.5)
        ens=proxy_f(select(score,k),y,True)
        vals.append({'video_id':vid,'stream':stream,'n':n,'center':base,'ensemble':ens,'delta':ens-base,'swaps':int(len(set(select(c,k))^set(select(score,k)))//2)})
    out={'experiment_id':'R58_TIMELENS_MASK_EDGE_ENSEMBLE_PROXY','status':'PUBLIC_SOURCE_HELD_OUT_COMPLETED','views':len(vals),'eta_inside':ETA_INSIDE,'eta_edge':ETA_EDGE,'source':vals,'macro_center':float(np.mean([x['center'] for x in vals])),'macro_ensemble':float(np.mean([x['ensemble'] for x in vals])),'macro_delta':float(np.mean([x['delta'] for x in vals])),'positive_views':int(sum(x['delta']>0 for x in vals)),'mean_swaps':float(np.mean([x['swaps'] for x in vals]))}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n'); print(json.dumps({k:out[k] for k in ('experiment_id','views','macro_center','macro_ensemble','macro_delta','positive_views','mean_swaps')},ensure_ascii=False))

if __name__=='__main__': main()
