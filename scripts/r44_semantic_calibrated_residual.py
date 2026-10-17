#!/usr/bin/env python3
"""R44 human-label calibration of the audited InternVideo2 score.

The six-billion-parameter model is used only to create public-source and
already completed R43 anchor scores.  The deployed candidate is a tiny frozen
linear residual plus OpenCV descriptors and the existing B3 spatial parent.
No official labels are read.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.r44_local_state_residual import (
    GRADE, _rank01, center_rank, frame_descriptor, select, proxy_f,
)


def read_rows(paths):
    rows = []
    for stream, path in enumerate(paths):
        with path.open(newline="") as f:
            for r in csv.DictReader(f):
                r["_stream"] = stream
                rows.append(r)
    return rows


def public_features(root: Path, vid: str) -> np.ndarray:
    ims = []
    for i in range(30):
        p = root / vid / f"{i:02d}.jpg"
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(p)
        ims.append(im)
    base = np.stack([frame_descriptor(im) for im in ims])
    gray = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) for im in ims]
    prev = np.zeros(30, np.float32); nxt = np.zeros(30, np.float32)
    for i in range(30):
        if i: prev[i] = np.mean(np.abs(gray[i]-gray[i-1])) / 255.0
        if i+1 < 30: nxt[i] = np.mean(np.abs(gray[i+1]-gray[i])) / 255.0
    return np.c_[base, prev, nxt, np.abs(prev-nxt)].astype(np.float32)


def fit(x, y, c, alpha):
    sx = StandardScaler().fit(x)
    A = np.c_[np.ones(len(c)), c]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    base = A @ coef
    head = Ridge(alpha=alpha).fit(sx.transform(x), y-base)
    return sx, head


def score_proxy(rows_by_vid, preds, eta, keep):
    vals = []
    for vid, items in rows_by_vid.items():
        # One view per stream.  Overlapping labels stay separate views so a
        # disagreement is visible; this is not a consensus GT.
        streams = sorted(set(int(x["_stream"]) for x in items))
        for stream in streams:
            rs = [x for x in items if int(x["_stream"]) == stream]
            by_t = {int(float(x["time_sec"])): GRADE[x["generic_highlight"]] for x in rs}
            times = np.array(sorted(by_t), dtype=int)
            y = np.array([by_t[t] for t in times], dtype=float)
            c = 1.0 - np.abs(times - 14.5) / 14.5
            rr = np.array([preds[(vid, int(t))] for t in times], dtype=float)
            k = int(round(keep * len(times)))
            kc = select(c, k)
            fused = _rank01(c) + eta * (_rank01(rr)-0.5)
            kr = select(fused, k)
            vals.append({"video_id": vid, "stream": stream,
                         "center": proxy_f(kc, y, True),
                         "residual": proxy_f(kr, y, True),
                         "delta": proxy_f(kr, y, True)-proxy_f(kc, y, True),
                         "center_binary": proxy_f(kc, y, False),
                         "residual_binary": proxy_f(kr, y, False),
                         "delta_binary": proxy_f(kr, y, False)-proxy_f(kc, y, False)})
    return vals


def probe(args):
    rows = read_rows(args.labels)
    sem = json.loads(args.public_scores.read_text())
    feats = {}; sem_by_vid = {}
    for vid, d in sem["videos"].items():
        feats[vid] = public_features(args.frames, vid)
        sem_by_vid[vid] = np.asarray(d["event_score"], dtype=np.float32)
    X, Y, G, T = [], [], [], []
    for r in rows:
        vid = r["video_id"]; t = int(float(r["time_sec"]))
        X.append(feats[vid][t]); Y.append(GRADE[r["generic_highlight"]]); G.append(vid); T.append(t)
    X=np.asarray(X,np.float32); Y=np.asarray(Y,float); G=np.asarray(G); T=np.asarray(T)
    c=np.asarray([1.0-abs(t-14.5)/14.5 for t in T],float)
    semantic=np.asarray([sem_by_vid[v][min(t,len(sem_by_vid[v])-1)] for v,t in zip(G,T)],float)
    arms={"D0_CENTER":np.zeros((len(X),1),np.float32),
          "D2_LOCAL":X,
          "D4_IV2":semantic[:,None],
          "D5_LOCAL_IV2":np.c_[X,semantic]}
    pred_by_arm={}; fold_metrics={}
    for name,x in arms.items():
        p=np.full(len(Y),np.nan); folds=[]
        for vid in sorted(set(G)):
            tr,te=G!=vid,G==vid
            if name=="D0_CENTER":
                p[te]=0.0
                mae=float(np.mean(np.abs(Y[te]-(np.mean(Y[tr])+0.0*c[te]))))
            else:
                sx,head=fit(x[tr],Y[tr],c[tr],args.alpha)
                p[te]=head.predict(sx.transform(x[te]))
                mae=float(np.mean(np.abs(Y[te]-(np.mean(Y[tr])+p[te]))))
            folds.append({"video_id":vid,"mae":mae,"rho":float(spearmanr(Y[te],p[te]).statistic) if len(np.unique(Y[te]))>1 else None})
        pred_by_arm[name]={(v,int(t)):p[i] for i,(v,t) in enumerate(zip(G,T))}
        fold_metrics[name]={"macro_mae":float(np.mean([z["mae"] for z in folds])),"folds":folds}
    # Candidate residual uses the combined semantic+local arm, refit on all
    # labels.  The deployment coefficient is frozen at eta=.10 before looking
    # at official frames; eta rows are robustness diagnostics only.
    sx,head=fit(arms["D5_LOCAL_IV2"],Y,c,args.alpha)
    rhat=head.predict(sx.transform(arms["D5_LOCAL_IV2"]))
    pred={(v,int(t)):rhat[i] for i,(v,t) in enumerate(zip(G,T))}
    rows_by={}
    for r in rows: rows_by.setdefault(r["video_id"],[]).append(r)
    eta_rows={str(e):score_proxy(rows_by,pred,e,args.keep) for e in args.etas}
    result={"experiment_id":"R44_RESIDUAL_002","status":"PUBLIC_SOURCE_HELD_OUT_COMPLETED",
            "rows":len(rows),"sources":len(rows_by),"alpha":args.alpha,"keep":args.keep,
            "arms":fold_metrics,"candidate_arm":"D5_LOCAL_IV2","deployment_eta":0.10,
            "eta_proxy":{e:{"macro_delta":float(np.mean([x["delta"] for x in rs])),
                            "positive_views":int(sum(x["delta"]>0 for x in rs)),
                            "n_views":len(rs),
                            "macro_binary_delta":float(np.mean([x["delta_binary"] for x in rs]))}
                           for e,rs in eta_rows.items()},
            "views":eta_rows["0.1"] if "0.1" in eta_rows else eta_rows[str(0.1)],
            "feature_contract":"9 static descriptors + prev/next state + audited IV2 event score; source-held-out Ridge residual"}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"output":str(args.output),"sources":len(rows_by),"arms":fold_metrics},ensure_ascii=False))


def decode_anchors(video: Path, anchor_count: int) -> tuple[np.ndarray, float, int]:
    cap=cv2.VideoCapture(str(video)); fps=float(cap.get(cv2.CAP_PROP_FPS) or 30.0); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration=max((n-1)/fps,0.0)
    anchors=np.arange(0.0,duration+1e-6,1.0,dtype=float)
    if len(anchors)==0 or anchors[-1]<duration: anchors=np.r_[anchors,duration]
    if len(anchors)!=anchor_count:
        anchors=np.linspace(0.0,duration,anchor_count)
    # Decode once in ordinal order.  Repeated random seeks are much slower on
    # the long H.264 files and can silently fail near the tail.
    targets=np.clip(np.rint(anchors*fps).astype(int),0,max(n-1,0))
    wanted={int(t):[] for t in targets}
    for j,t in enumerate(targets): wanted[int(t)].append(j)
    ims=[None]*len(targets); idx=0
    while idx<=int(targets.max(initial=0)):
        ok,im=cap.read()
        if not ok: break
        if idx in wanted:
            for j in wanted[idx]: ims[j]=im.copy()
        idx += 1
    cap.release()
    fallback=next((im for im in ims if im is not None), np.zeros((224,224,3),np.uint8))
    ims=[im if im is not None else fallback.copy() for im in ims]
    base=np.stack([frame_descriptor(im) for im in ims]); gray=[cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32) for im in ims]
    prev=np.zeros(anchor_count,np.float32); nxt=np.zeros(anchor_count,np.float32)
    for i in range(anchor_count):
        if i: prev[i]=np.mean(np.abs(gray[i]-gray[i-1]))/255.0
        if i+1<anchor_count: nxt[i]=np.mean(np.abs(gray[i+1]-gray[i]))/255.0
    return np.c_[base,prev,nxt,np.abs(prev-nxt)].astype(np.float32),fps,n


def official(args):
    rows=[json.loads(x) for x in args.parent.read_text().splitlines() if x.strip()]
    public=read_rows(args.labels); ps=json.loads(args.public_scores.read_text())
    X=[];Y=[];C=[];S=[]
    cache={}
    for r in public:
        v=r["video_id"]; t=int(float(r["time_sec"]));
        if v not in cache: cache[v]=public_features(args.frames,v)
        X.append(cache[v][t]);Y.append(GRADE[r["generic_highlight"]]);C.append(1.0-abs(t-14.5)/14.5)
        d=ps["videos"][v];S.append(d["event_score"][min(t,len(d["event_score"])-1)])
    X=np.asarray(X,np.float32);Y=np.asarray(Y,float);C=np.asarray(C,float);S=np.asarray(S,float)
    sx,head=fit(np.c_[X,S],Y,C,args.alpha)
    out=[]; changed=0; total=0
    report=json.loads(args.r43_report.read_text()); diag={str(x["video_id"]):x for x in report["diagnostics"]}
    for r in rows:
        vid=str(r["video_id"]); d=diag[vid]; anchors=np.asarray(d["event_score"],float)
        feat,fps,n=decode_anchors(args.video_root/f"{vid}.mp4",len(anchors))
        # R43 anchor score is the semantic feature; interpolate it to every
        # original frame, then map candidate scores only over the parent frames.
        local=np.asarray(feat,np.float32); sem_anchor=anchors
        model_res=head.predict(sx.transform(np.c_[local,sem_anchor]))
        times=np.arange(n,dtype=float)/max(fps,1e-6); anchor_times=np.linspace(0.0,max((n-1)/max(fps,1e-6),0.0),len(anchors))
        rr=np.interp(times,anchor_times,model_res)
        center=1.0-np.abs(times/max(times[-1],1e-6)-0.5)*2.0
        parent={int(p["frame"]):p for p in r["predictions"]}; ids=np.asarray(sorted(parent),int)
        if len(ids) != n:
            raise RuntimeError(f"full B3 candidate coverage mismatch for {vid}: {len(ids)} != decoded {n}")
        fused=_rank01(center[ids])+args.eta*(_rank01(rr[ids])-0.5)
        k=int(round(0.85*n))
        sel=ids[select(fused,k)]; preds=[parent[int(i)] for i in sorted(sel)]
        if [p["frame"] for p in preds]!=[p["frame"] for p in r["predictions"]]: changed+=1
        total+=len(preds); q=dict(r); q["model_size_mb"]=450.283377; q["model_params_b"]=0.450283377; q["predictions"]=preds; out.append(q)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w") as f:
        for r in out:f.write(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n")
    ans={"experiment_id":"R44_RESIDUAL_002_OFFICIAL","videos":len(out),"predictions":total,"changed_videos":changed,"eta":args.eta,"alpha":args.alpha,"parent":"R26_POS_CENTER_EXACT085_B3","boxes":"byte copied from parent","output":str(args.output)}
    print(json.dumps(ans,ensure_ascii=False))


def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="cmd",required=True)
    p=sub.add_parser("probe");p.add_argument("--labels",nargs=2,type=Path,required=True);p.add_argument("--frames",type=Path,required=True);p.add_argument("--public-scores",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--alpha",type=float,default=10.0);p.add_argument("--keep",type=float,default=.85);p.add_argument("--etas",type=float,nargs="+",default=[.05,.1,.15,.2,.25])
    o=sub.add_parser("official");o.add_argument("--labels",nargs=2,type=Path,required=True);o.add_argument("--frames",type=Path,required=True);o.add_argument("--public-scores",type=Path,required=True);o.add_argument("--r43-report",type=Path,required=True);o.add_argument("--video-root",type=Path,required=True);o.add_argument("--parent",type=Path,required=True);o.add_argument("--output",type=Path,required=True);o.add_argument("--eta",type=float,default=.1);o.add_argument("--alpha",type=float,default=10.0)
    a=ap.parse_args()
    if a.cmd=="probe":probe(a)
    else:official(a)


if __name__=="__main__":main()
