#!/usr/bin/env python3
"""R44 frozen InternVideo2 embedding readout and R26 candidate builder."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.r44_local_state_residual import GRADE, _rank01, proxy_f, select


def read_rows(paths):
    rows=[]
    for stream,path in enumerate(paths):
        with path.open(newline="") as f:
            for r in csv.DictReader(f): r["_stream"]=stream; rows.append(r)
    return rows


def load_public(rows, emb_root):
    cache={}
    for r in rows:
        v=r["video_id"]
        if v not in cache:
            z=np.load(emb_root/f"{v}.npz")
            cache[v]=np.asarray(z["embeddings"],np.float32)
    X=[];Y=[];G=[];T=[]
    for r in rows:
        v=r["video_id"]; t=int(float(r["time_sec"]))
        e=cache[v][min(t,len(cache[v])-1)]
        X.append(e);Y.append(GRADE[r["generic_highlight"]]);G.append(v);T.append(t)
    return np.asarray(X,np.float32),np.asarray(Y,float),np.asarray(G),np.asarray(T),cache


def fit(x,y,c,alpha):
    sx=StandardScaler().fit(x)
    A=np.c_[np.ones(len(c)),c]; coef,*_=np.linalg.lstsq(A,y,rcond=None)
    base=A@coef
    head=Ridge(alpha=alpha).fit(sx.transform(x),y-base)
    return sx,head


def proxy(rows, pred, eta, keep):
    by={}
    for r in rows: by.setdefault(r["video_id"],[]).append(r)
    out=[]
    for v,rs in by.items():
        for stream in sorted(set(int(r["_stream"]) for r in rs)):
            rr=[r for r in rs if int(r["_stream"])==stream]
            ymap={int(float(r["time_sec"])):GRADE[r["generic_highlight"]] for r in rr}
            ts=np.array(sorted(ymap),int); y=np.array([ymap[t] for t in ts],float)
            c=1-np.abs(ts-14.5)/14.5; z=np.array([pred[(v,int(t))] for t in ts])
            k=round(keep*len(ts)); kc=select(c,k); kr=select(_rank01(c)+eta*(_rank01(z)-.5),k)
            out.append({"video_id":v,"stream":stream,"center":proxy_f(kc,y,True),"residual":proxy_f(kr,y,True),"delta":proxy_f(kr,y,True)-proxy_f(kc,y,True),"binary_delta":proxy_f(kr,y,False)-proxy_f(kc,y,False)})
    return out


def probe(args):
    rows=read_rows(args.labels); X,Y,G,T,_=load_public(rows,args.public_embeddings)
    c=np.array([1-abs(t-14.5)/14.5 for t in T],float)
    arms={"D0_CENTER":None,"D6_IV2_EMBED":X}
    result={"experiment_id":"R44_RESIDUAL_003","rows":len(rows),"sources":len(set(G)),"alpha":args.alpha,"keep":args.keep,"arms":{},"eta_proxy":{}}
    pred_all={}
    for name,x in arms.items():
        p=np.zeros(len(Y)); folds=[]
        for v in sorted(set(G)):
            tr,te=G!=v,G==v
            if x is None:
                p[te]=0
                mae=float(np.mean(np.abs(Y[te]-np.mean(Y[tr]))))
            else:
                sx,head=fit(x[tr],Y[tr],c[tr],args.alpha);p[te]=head.predict(sx.transform(x[te]));mae=float(np.mean(np.abs(Y[te]-(np.mean(Y[tr])+p[te]))))
            folds.append({"video_id":v,"mae":mae,"rho":float(spearmanr(Y[te],p[te]).statistic) if len(np.unique(Y[te]))>1 and len(np.unique(p[te]))>1 else None})
        result["arms"][name]={"macro_mae":float(np.mean([z["mae"] for z in folds])),"folds":folds}
        if name=="D6_IV2_EMBED": pred_all={(v,int(t)):p[i] for i,(v,t) in enumerate(zip(G,T))}
    for eta in args.etas:
        rs=proxy(rows,pred_all,eta,args.keep)
        result["eta_proxy"][str(eta)]={"macro_delta":float(np.mean([x["delta"] for x in rs])),"positive_views":int(sum(x["delta"]>0 for x in rs)),"n_views":len(rs),"macro_binary_delta":float(np.mean([x["binary_delta"] for x in rs]))}
    result["deployment_eta"]=args.eta; result["feature_contract"]="512D pooled InternVideo2 visual embedding, four-frame one-second windows, center-frozen Ridge residual"
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False))


def official(args):
    rows=read_rows(args.labels); X,Y,G,T,_=load_public(rows,args.public_embeddings);c=np.array([1-abs(t-14.5)/14.5 for t in T],float)
    sx,head=fit(X,Y,c,args.alpha)
    out=[];changed=0;total=0
    for r in [json.loads(x) for x in args.parent.read_text().splitlines() if x.strip()]:
        v=str(r["video_id"]);z=np.load(args.official_embeddings/f"{v}.npz");emb=np.asarray(z["embeddings"],np.float32); anchors=np.asarray(z["anchors"],float);fps=float(z["fps"]);n=int(z["frame_count"])
        rr=head.predict(sx.transform(emb));times=np.arange(n)/max(fps,1e-6);anchor_times=anchors
        frame_r=np.interp(times,anchor_times,rr);center=1-np.abs(times/max(times[-1],1e-6)-.5)*2
        parent={int(p["frame"]):p for p in r["predictions"]};ids=np.array(sorted(parent),int);fused=_rank01(center[ids])+args.eta*(_rank01(frame_r[ids])-.5);sel=ids[select(fused,len(ids))];preds=[parent[int(i)] for i in sorted(sel)]
        if [p["frame"] for p in preds]!=[p["frame"] for p in r["predictions"]]:changed+=1
        total+=len(preds);q=dict(r);q["predictions"]=preds;out.append(q)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w") as f:
        for r in out:f.write(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n")
    print(json.dumps({"experiment_id":"R44_RESIDUAL_003_OFFICIAL","videos":len(out),"predictions":total,"changed_videos":changed,"eta":args.eta,"alpha":args.alpha,"parent":"R26_POS_CENTER_EXACT085_B3","boxes":"byte copied","output":str(args.output)}))


def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest="cmd",required=True)
    p=sub.add_parser("probe");p.add_argument("--labels",nargs=2,type=Path,required=True);p.add_argument("--public-embeddings",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--alpha",type=float,default=100.0);p.add_argument("--keep",type=float,default=.85);p.add_argument("--etas",type=float,nargs="+",default=[.05,.1,.15,.2,.25]);p.add_argument("--eta",type=float,default=.2)
    o=sub.add_parser("official");o.add_argument("--labels",nargs=2,type=Path,required=True);o.add_argument("--public-embeddings",type=Path,required=True);o.add_argument("--official-embeddings",type=Path,required=True);o.add_argument("--parent",type=Path,required=True);o.add_argument("--output",type=Path,required=True);o.add_argument("--alpha",type=float,default=100.0);o.add_argument("--eta",type=float,default=.2)
    a=ap.parse_args();probe(a) if a.cmd=="probe" else official(a)


if __name__=="__main__":main()
