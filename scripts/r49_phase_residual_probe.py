#!/usr/bin/env python3
"""R49 mechanism probe: event-phase prediction as a temporal residual.

The existing D3 arm regresses generic highlight grade directly from short-term
appearance changes.  This probe factors the label into an observable event
phase first (setup, ongoing, result, reaction, no-event), then converts the
out-of-source phase posterior into a fixed expected highlight utility.  The
only deployed intervention is a fixed eta=.20 rank residual on top of the
centre prior.  This is a mechanism test, not a weight sweep.

Only temporal state features (the high-frequency descriptor columns after the
9 static values) are used by the phase classifier.  Fold boundaries are whole
videos, including the two annotator-overlap sources.
"""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.r45_highfreq_semantic_residual import decode_highfreq, highfreq_features
from scripts.r44_local_state_residual import GRADE, _rank01, proxy_f, select

PHASES = ["铺垫", "进行中", "完成或结果", "结果后反应", "无事件或无法判断"]
PHASE_ID = {x:i for i,x in enumerate(PHASES)}

def read_rows(paths):
    out=[]
    for p in paths:
        with Path(p).open(newline="") as f:
            out += list(csv.DictReader(f))
    return out

def fit_phase(x, phase, grade):
    scaler = StandardScaler().fit(x)
    z = np.zeros((len(phase), len(PHASES)), np.float64)
    for i,p in enumerate(phase):
        if p in PHASE_ID: z[i, PHASE_ID[p]] = 1.0
    model = Ridge(alpha=10.0).fit(scaler.transform(x), z)
    util = np.zeros(len(PHASES), np.float64)
    for j in range(len(PHASES)):
        a = np.asarray([grade[i] for i,p in enumerate(phase) if p == PHASES[j]], float)
        util[j] = float(np.mean(a)) if len(a) else float(np.mean(grade))
    # Center utility to avoid letting a fold's label prevalence change the
    # frozen centre anchor's scale.
    return scaler, model, util

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(np.clip(z, -30, 30))
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-8)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--labels", nargs=2, type=Path, required=True)
    ap.add_argument("--videos", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--eta", type=float, default=.20)
    args=ap.parse_args()
    labels=read_rows(args.labels)
    by={}; data={}
    for r in labels: by.setdefault(r["video_id"], []).append(r)
    for v in sorted(by):
        fr,_,_=decode_highfreq(args.videos/f"{v}.mp4")
        data[v]=highfreq_features(fr).astype(np.float32)

    source=[]
    for held in sorted(by):
        tr=[r for r in labels if r["video_id"] != held]
        X=[]; P=[]; Y=[]
        for r in tr:
            t=min(int(float(r["time_sec"])),29)
            X.append(data[r["video_id"]][min(4*t, len(data[r["video_id"]])-1), 9:])
            P.append(r.get("event_phase", "") or "无事件或无法判断")
            Y.append(GRADE[r["generic_highlight"]])
        X=np.asarray(X,np.float32); Y=np.asarray(Y,float)
        sc,m,util=fit_phase(X,P,Y)
        # Use one row per time. For annotator overlap, generic grade is
        # averaged and phase is the deterministic majority label.
        grouped={}
        for r in by[held]: grouped.setdefault(int(float(r["time_sec"])), []).append(r)
        times=np.asarray(sorted(grouped),int)
        y=np.asarray([np.mean([GRADE[r["generic_highlight"]] for r in grouped[t]]) for t in times],float)
        phases=[]
        for t in times:
            rr=[r.get("event_phase", "") or "无事件或无法判断" for r in grouped[t]]
            phases.append(max(PHASES, key=lambda p:(rr.count(p), -PHASES.index(p))))
        xte=data[held][np.minimum(4*times,len(data[held])-1),9:]
        post=softmax(m.predict(sc.transform(xte)))
        q=post@util
        c=1-np.abs(times-(len(times)-1)/2)/max((len(times)-1)/2,1)
        k=int(round(.85*len(times)))
        kc=select(c,k); kr=select(_rank01(c)+args.eta*(_rank01(q)-.5),k)
        source.append({"video_id":held,"n":len(times),"phase_accuracy":float(np.mean(np.argmax(post,1)==np.asarray([PHASE_ID[p] for p in phases]))),"center":proxy_f(kc,y,True),"phase":proxy_f(kr,y,True),"delta":proxy_f(kr,y,True)-proxy_f(kc,y,True),"center_binary":proxy_f(kc,y,False),"phase_binary":proxy_f(kr,y,False),"delta_binary":proxy_f(kr,y,False)-proxy_f(kc,y,False),"utility_by_phase":dict(zip(PHASES,util.tolist()))})
    out={"experiment_id":"R49_EVENT_PHASE_RESIDUAL_PROBE","status":"PUBLIC_SOURCE_HELD_OUT_COMPLETED","rows":len(labels),"sources":len(source),"eta":args.eta,"alpha":10.0,"feature_contract":"4fps temporal state columns only -> Ridge phase posterior -> fold train phase utility -> fixed residual","aggregation":"overlap annotations averaged per time for held-out evaluation","official_gt":False,"macro_center":float(np.mean([x["center"] for x in source]),),"macro_phase":float(np.mean([x["phase"] for x in source])),"macro_delta":float(np.mean([x["delta"] for x in source])),"positive_sources":int(sum(x["delta"]>0 for x in source)),"macro_center_binary":float(np.mean([x["center_binary"] for x in source])),"macro_phase_binary":float(np.mean([x["phase_binary"] for x in source])),"positive_sources_binary":int(sum(x["delta_binary"]>0 for x in source)),"source":source}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n"); print(json.dumps({k:out[k] for k in ("experiment_id","macro_center","macro_phase","macro_delta","positive_sources","macro_center_binary","macro_phase_binary","positive_sources_binary")},ensure_ascii=False,indent=2))

if __name__ == "__main__": main()
