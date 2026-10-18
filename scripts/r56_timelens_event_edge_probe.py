#!/usr/bin/env python3
"""Public source-held-out probe for the TimeLens2 event-edge residual."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from scripts.r44_local_state_residual import GRADE, proxy_f, select

def rank01(x):
    x=np.asarray(x,float)
    return np.argsort(np.argsort(x,kind="mergesort"),kind="mergesort").astype(float)/max(len(x)-1,1)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--labels',nargs=2,type=Path,required=True); ap.add_argument('--responses',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    resp=json.loads(a.responses.read_text()); spans={str(x['video_id']):[tuple(z) for z in x.get('span_seconds',[])] for x in resp['source']}
    rows=[]
    for stream,p in enumerate(a.labels):
        with p.open(newline='') as f:
            for x in csv.DictReader(f): x['_stream']=stream; rows.append(x)
    by={}
    for r in rows: by.setdefault(r['video_id'],[]).append(r)
    out=[]
    for vid,items in sorted(by.items()):
        for stream in sorted({int(x['_stream']) for x in items}):
            rs=[x for x in items if int(x['_stream'])==stream]
            times=np.asarray(sorted({int(float(x['time_sec'])) for x in rs}),float)
            y=np.asarray([GRADE[next(x['generic_highlight'] for x in rs if int(float(x['time_sec']))==t)] for t in times],float)
            n=len(times); k=int(round(.85*n)); c=1-np.abs(times-14.5)/14.5
            edge=np.zeros(n,float)
            for aa,bb in spans.get(vid,[]):
                inside=(times>=aa-1e-6)&(times<=bb+1e-6); phase=np.clip((times-aa)/max(bb-aa,1e-6),0,1)
                edge=np.maximum(edge,np.where(inside,2*np.abs(phase-.5),0))
            kc=select(c,k); kr=select(rank01(c)+.2*(rank01(np.asarray([any(aa<=t<=bb for aa,bb in spans.get(vid,[])) for t in times],float))-.5),k); ke=select(rank01(c)+.2*(rank01(edge)-.5),k)
            out.append({'video_id':vid,'stream':stream,'center':proxy_f(kc,y,True),'r54':proxy_f(kr,y,True),'edge':proxy_f(ke,y,True),'edge_delta_r54':proxy_f(ke,y,True)-proxy_f(kr,y,True),'center_binary':proxy_f(kc,y,False),'r54_binary':proxy_f(kr,y,False),'edge_binary':proxy_f(ke,y,False),'edge_binary_delta_r54':proxy_f(ke,y,False)-proxy_f(kr,y,False)})
    result={'experiment_id':'R56_TIMELENS_EVENT_EDGE_PUBLIC_PROBE','status':'PUBLIC_SOURCE_DIAGNOSTIC','views':len(out),'macro':{k:float(np.mean([x[k] for x in out])) for k in ['center','r54','edge','edge_delta_r54','center_binary','r54_binary','edge_binary','edge_binary_delta_r54']},'positive_views':int(sum(x['edge_delta_r54']>0 for x in out)),'negative_views':int(sum(x['edge_delta_r54']<0 for x in out)),'per_view':out}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n'); print(json.dumps({k:result[k] for k in ['experiment_id','views','macro','positive_views','negative_views']},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
