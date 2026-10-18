#!/usr/bin/env python3
"""R53b: can a deployable phase proxy recover the human phase interaction?

The previous R53 diagnostic found that observed event_phase fields improve a
local residual, but those fields are unavailable on official videos.  This
script trains a tiny phase readout from the same RGB/state descriptors, with
source-held-out and inner source-held-out predictions, then uses only the
predicted phase scores as residual features.  It is a falsifiable test of
whether the semantic interaction is recoverable without adding a VLM.
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import cv2, numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.r53_quality_interaction_probe import (GRADE, read_rows, features,
    center_rank, select, proxy_f)


def onehot(vals, classes):
    ix = {v:i for i,v in enumerate(classes)}; z = np.zeros((len(vals),len(classes)), np.float32)
    for i,v in enumerate(vals):
        if v in ix: z[i,ix[v]] = 1.
    return z


def phase_scores(xtr, ytr, xte, classes):
    sy = StandardScaler().fit(xtr)
    oh = onehot(ytr, classes)
    model = Ridge(alpha=10.).fit(sy.transform(xtr), oh)
    return model.predict(sy.transform(xte))


def residual_fit(xtr, ytr, ctr, xte, cte):
    sx = StandardScaler().fit(xtr)
    A = np.c_[np.ones(len(ctr)), ctr]; coef, *_ = np.linalg.lstsq(A,ytr,rcond=None)
    model = Ridge(alpha=10.).fit(sx.transform(xtr), ytr - A@coef)
    p = model.predict(sx.transform(xte)); target=np.c_[np.ones(len(cte)),cte]@coef+p
    return p,target


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--labels',nargs=2,type=Path,required=True); ap.add_argument('--frames',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    rows=read_rows(args.labels); vids=sorted({r['video_id'] for r in rows}); cache={v:features(args.frames,v) for v in vids}
    X=np.stack([cache[r['video_id']][int(float(r['time_sec']))] for r in rows]); y=np.asarray([GRADE[r['generic_highlight']] for r in rows],float); g=np.asarray([r['video_id'] for r in rows]); t=np.asarray([int(float(r['time_sec'])) for r in rows]); c=1.-np.abs(t-14.5)/14.5
    phase=np.asarray([r.get('event_phase','') or '<MISSING>' for r in rows]); classes=sorted(set(phase))
    pred=np.full(len(rows),np.nan); target=np.full(len(rows),np.nan); phase_pred=np.full((len(rows),len(classes)),np.nan); folds=[]
    # Outer source holdout.  Cross-fit the phase readout on each training
    # source before fitting the value residual to avoid same-source label leak.
    for v in vids:
        tr,te=g!=v,g==v; train_vids=sorted(set(g[tr])); train_phase=np.full((tr.sum(),len(classes)),np.nan)
        tr_positions=np.flatnonzero(tr)
        for u in train_vids:
            a=np.asarray([i for i in tr_positions if g[i]!=u]); b=np.asarray([i for i in tr_positions if g[i]==u])
            if len(a)==0 or len(b)==0: continue
            train_phase[np.isin(tr_positions,b)] = phase_scores(X[a],phase[a],X[b],classes)
        # In the unlikely case a fold is missing due to a singleton source,
        # use an all-training model only for that row; all current sources are
        # complete 30-frame blocks so this branch is not expected.
        missing=np.isnan(train_phase).any(1)
        if missing.any(): train_phase[missing]=phase_scores(X[tr],phase[tr],X[tr_positions[missing]],classes)
        test_phase=phase_scores(X[tr],phase[tr],X[te],classes)
        # Predicted phase scores interact with local state; all are observable
        # at deployment after running this frozen tiny phase readout.
        xxtr=np.c_[X[tr],train_phase,(X[tr,:,None]*train_phase[:,None,:]).reshape(tr.sum(),-1)]
        xxte=np.c_[X[te],test_phase,(X[te,:,None]*test_phase[:,None,:]).reshape(te.sum(),-1)]
        p,yp=residual_fit(xxtr,y[tr],c[tr],xxte,c[te]); pred[te]=p; target[te]=yp; phase_pred[te]=test_phase
        phase_acc=float(np.mean(np.argmax(test_phase,axis=1)==np.argmax(onehot(phase[te],classes),axis=1)))
        folds.append({'video_id':v,'mae':float(np.mean(np.abs(y[te]-yp))), 'rho':float(spearmanr(y[te],p).statistic) if len(np.unique(y[te]))>1 else None,'phase_accuracy':phase_acc})
    deltas=[]; views=[]
    for v in vids:
        for stream in sorted({int(r['_stream']) for r in rows if r['video_id']==v}):
            ix=np.asarray([i for i,r in enumerate(rows) if r['video_id']==v and int(r['_stream'])==stream]); ix=ix[np.argsort(t[ix])]; _,keep=np.unique(t[ix],return_index=True);ix=ix[np.sort(keep)]
            yy=y[ix]; cc=center_rank(len(ix)); k=int(round(.85*len(ix))); base=proxy_f(select(cc,k),yy); rb=np.argsort(np.argsort(pred[ix],kind='mergesort'),kind='mergesort')/max(len(ix)-1,1); sc=cc+.20*(rb-.5); val=proxy_f(select(sc,k),yy); deltas.append(val-base)
            views.append({'video_id':v,'stream':stream,'base':base,'phase_pred':val,'delta':val-base,'base_binary':proxy_f(select(cc,k),yy,True),'phase_pred_binary':proxy_f(select(sc,k),yy,True)})
    out={'experiment_id':'R53_PHASE_PREDICTED_READOUT','status':'PUBLIC_SOURCE_HELD_OUT_COMPLETED','rows':len(rows),'sources':len(vids),'views':len(views),'fixed_eta':.20,'fixed_ridge_alpha':10.,'classes':classes,'hypothesis':'human event-phase interaction remains useful when phase is predicted from observable RGB/state features','macro_mae':float(np.mean([z['mae'] for z in folds])),'macro_rho':float(np.nanmean([z['rho'] for z in folds])),'macro_phase_accuracy':float(np.mean([z['phase_accuracy'] for z in folds])),'macro_proxy_delta':float(np.mean(deltas)),'positive_views':int(sum(x>0 for x in deltas)),'folds':folds,'views_detail':views,'feature_contract':'D2 local descriptors + source-held-out inner-cross-fit Ridge phase score and local x phase interactions'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:out[k] for k in ('experiment_id','macro_mae','macro_rho','macro_proxy_delta','positive_views')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
