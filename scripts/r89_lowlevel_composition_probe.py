#!/usr/bin/env python3
"""R89: low-level motion/saliency/composition probe on native crop GT.

This is deliberately a small, auditable alternative to the V9 utility head.
It uses only decoded public keyframes and parameter-free image statistics:
gradient/local-contrast saliency, frame difference, sharpness and legal-window
geometry.  A source-disjoint ridge readout ranks the 129 legal crop windows by
their native IoU target.  No official videos or labels are read.

The output is a diagnostic artifact first.  It can later be used to generate
official point observations, but this script never touches the official set.
"""
from __future__ import annotations

import argparse, json, pickle, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

NC = 129


def geometry(W, H, ratio):
    target = float(ratio[0]) / float(ratio[1])
    if W / H > target:
        h, w, axis = float(H), float(H) * target, 0
    elif W / H < target:
        w, h, axis = float(W), float(W) / target, 1
    else:
        w, h, axis = float(W), float(H), None
    return w, h, axis


def candidates(W, H, ratio):
    w, h, axis = geometry(W, H, ratio)
    if axis == 0:
        off = np.linspace(0., W - w, NC, dtype=np.float32)
    elif axis == 1:
        off = np.linspace(0., H - h, NC, dtype=np.float32)
    else:
        off = np.zeros(NC, dtype=np.float32)
    return off, w, h, axis


def boxes(offset, W, H, ratio):
    w, h, axis = geometry(W, H, ratio)
    out = np.zeros((len(offset), 4), np.float32)
    if axis == 0:
        out[:] = np.stack([offset, np.zeros(len(offset)), offset + w,
                           np.full(len(offset), h)], 1)
    elif axis == 1:
        out[:] = np.stack([np.zeros(len(offset)), offset,
                           np.full(len(offset), w), offset + h], 1)
    else:
        out[:] = [0., 0., W, H]
    return out


def iou(a, b):
    a = np.asarray(a, np.float32)[:, None, :]
    b = np.asarray(b, np.float32)[None, :, :]
    x1 = np.maximum(a[..., 0], b[..., 0]); y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2]); y2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    aa = np.maximum(a[..., 2] - a[..., 0], 0) * np.maximum(a[..., 3] - a[..., 1], 0)
    bb = np.maximum(b[..., 2] - b[..., 0], 0) * np.maximum(b[..., 3] - b[..., 1], 0)
    return inter / np.maximum(aa + bb - inter, 1e-8)


def load_gray(path, size=96):
    im = Image.open(path).convert("L").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.


def maps(gray, prev):
    # Small finite-difference maps are stable across source resolutions.
    gx = np.diff(gray, axis=1, prepend=gray[:, :1]); gy = np.diff(gray, axis=0, prepend=gray[:1])
    grad = np.sqrt(gx * gx + gy * gy)
    blur = np.asarray(Image.fromarray(np.uint8(np.clip(gray * 255, 0, 255))).filter(
        ImageFilter.GaussianBlur(radius=3)), dtype=np.float32) / 255.
    contrast = np.abs(gray - blur)
    motion = np.abs(gray - prev) if prev is not None else np.zeros_like(gray)
    # Motion is deliberately part of saliency, but retained as a separate
    # feature so a fit can learn whether change helps or hurts composition.
    sal = grad + .6 * contrast + .8 * motion + 1e-6
    sharp = np.abs(np.diff(gray, n=2, axis=1, prepend=gray[:, :1], append=gray[:, -1:]))
    return grad, contrast, motion, sal, sharp


def region_stats(m, x1, y1, x2, y2):
    H, W = m.shape; xx1 = max(0, int(np.floor(x1 * W))); xx2 = min(W, max(xx1 + 1, int(np.ceil(x2 * W))))
    yy1 = max(0, int(np.floor(y1 * H))); yy2 = min(H, max(yy1 + 1, int(np.ceil(y2 * H))))
    return m[yy1:yy2, xx1:xx2], (xx1, yy1, xx2, yy2)


def _rect_sum(m, x1, y1, x2, y2):
    """Batch rectangle sums for m=[B,H,W], coordinates inclusive/exclusive."""
    b = np.arange(len(m))[:, None]
    ii = np.pad(np.cumsum(np.cumsum(m, axis=1), axis=2), ((0, 0), (1, 0), (1, 0)))
    return ii[b, y2[None, :], x2[None, :]] - ii[b, y1[None, :], x2[None, :]] \
        - ii[b, y2[None, :], x1[None, :]] + ii[b, y1[None, :], x1[None, :]]


def _rect_mean_std(m, x1, y1, x2, y2):
    area = np.maximum((x2 - x1) * (y2 - y1), 1)
    sm = _rect_sum(m, x1, y1, x2, y2)
    sq = _rect_sum(m * m, x1, y1, x2, y2)
    mean = sm / area[None, :]
    std = np.sqrt(np.maximum(sq / area[None, :] - mean * mean, 0.))
    return mean, std, sm


def _box_blur(gray, radius=3):
    """Fast batch box blur, adequate for a low-level contrast proxy."""
    k = 2 * radius + 1
    p = np.pad(gray, ((0, 0), (radius, radius), (radius, radius)), mode="edge")
    ii = np.pad(np.cumsum(np.cumsum(p, axis=1), axis=2), ((0, 0), (1, 0), (1, 0)))
    return (ii[:, k:, k:] - ii[:, :-k, k:] - ii[:, k:, :-k] + ii[:, :-k, :-k]) / float(k * k)


def frame_features_batch(gray, prev, W, H, ratio):
    """Vectorized [B,96,96] -> [B,129,D] candidate features."""
    off, w, h, axis = candidates(W, H, ratio)
    B, Hs, Ws = gray.shape
    gx = np.diff(gray, axis=2, prepend=gray[:, :, :1]); gy = np.diff(gray, axis=1, prepend=gray[:, :1, :])
    grad = np.sqrt(gx * gx + gy * gy); contrast = np.abs(gray - _box_blur(gray))
    motion = np.abs(gray - prev) if prev is not None else np.zeros_like(gray)
    sal = grad + .6 * contrast + .8 * motion + 1e-6
    sharp = np.abs(np.diff(gray, n=2, axis=2, prepend=gray[:, :, :1], append=gray[:, :, -1:]))
    if axis == 0:
        x1 = np.floor(off / W * Ws).astype(np.int64); x2 = np.ceil((off + w) / W * Ws).astype(np.int64)
        y1 = np.zeros(NC, np.int64); y2 = np.full(NC, max(1, int(np.ceil(h / H * Hs))), np.int64)
    elif axis == 1:
        x1 = np.zeros(NC, np.int64); x2 = np.full(NC, max(1, int(np.ceil(w / W * Ws))), np.int64)
        y1 = np.floor(off / H * Hs).astype(np.int64); y2 = np.ceil((off + h) / H * Hs).astype(np.int64)
    else:
        x1 = np.zeros(NC, np.int64); x2 = np.full(NC, Ws, np.int64); y1 = np.zeros(NC, np.int64); y2 = np.full(NC, Hs, np.int64)
    x1 = np.clip(x1, 0, Ws - 1); x2 = np.clip(np.maximum(x2, x1 + 1), 1, Ws)
    y1 = np.clip(y1, 0, Hs - 1); y2 = np.clip(np.maximum(y2, y1 + 1), 1, Hs)
    maps_ = [(grad, "grad"), (contrast, "contrast"), (motion, "motion"), (sal, "sal"), (sharp, "sharp")]
    total = {name: m.sum((1, 2))[:, None] + 1e-6 for m, name in maps_}
    parts = []
    for m, name in maps_:
        mean, std, sm = _rect_mean_std(m, x1, y1, x2, y2)
        parts += [mean[:, :, None], std[:, :, None], (sm / total[name])[:, :, None]]
    yy, xx = np.mgrid[0:Hs, 0:Ws].astype(np.float32)
    mass = sal.sum((1, 2)) + 1e-6
    scx = (sal * xx[None]).sum((1, 2)) / (mass * max(1, Ws - 1)); scy = (sal * yy[None]).sum((1, 2)) / (mass * max(1, Hs - 1))
    cx = ((x1 + x2) / 2 / Ws).astype(np.float32); cy = ((y1 + y2) / 2 / Hs).astype(np.float32)
    extras = np.stack([np.abs(cx[None] - scx[:, None]), np.abs(cy[None] - scy[:, None]),
                       np.broadcast_to(cx, (B, NC)), np.broadcast_to(cy, (B, NC)),
                       np.broadcast_to(np.minimum(cx, 1 - cx), (B, NC)),
                       np.broadcast_to(np.minimum(cy, 1 - cy), (B, NC)),
                       np.broadcast_to(np.minimum(np.abs(cx - 1/3), np.abs(cx - 2/3)), (B, NC)),
                       np.broadcast_to(np.minimum(np.abs(cy - 1/3), np.abs(cy - 2/3)), (B, NC)),
                       np.full((B, NC), float(axis == 0)), np.full((B, NC), float(axis == 1))], axis=-1)
    return np.concatenate(parts + [extras], axis=-1).astype(np.float32)


def frame_features(gray, prev, W, H, ratio):
    off, w, h, axis = candidates(W, H, ratio)
    grad, contrast, motion, sal, sharp = maps(gray, prev)
    Hs, Ws = gray.shape; yy, xx = np.mgrid[0:Hs, 0:Ws].astype(np.float32)
    total = {"grad": grad.sum() + 1e-6, "contrast": contrast.sum() + 1e-6,
             "motion": motion.sum() + 1e-6, "sal": sal.sum() + 1e-6,
             "sharp": sharp.sum() + 1e-6}
    mass = sal.sum() + 1e-6; scx = float((sal * xx).sum() / (mass * max(1, Ws - 1)))
    scy = float((sal * yy).sum() / (mass * max(1, Hs - 1)))
    out = []
    for o in off:
        if axis == 0:
            x1, x2, y1, y2 = o / W, (o + w) / W, 0., h / H
        elif axis == 1:
            x1, x2, y1, y2 = 0., w / W, o / H, (o + h) / H
        else:
            x1, x2, y1, y2 = 0., 1., 0., 1.
        vals = []
        inside_masks = []
        for name, m in [("grad", grad), ("contrast", contrast), ("motion", motion), ("sal", sal), ("sharp", sharp)]:
            q, b = region_stats(m, x1, y1, x2, y2); inside_masks.append(b)
            vals += [float(q.mean()), float(q.std()), float(q.sum() / total[name])]
        # Saliency center distance, crop center and boundary/thirds priors.
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        vals += [abs(cx - scx), abs(cy - scy), cx, cy,
                 min(cx, 1 - cx), min(cy, 1 - cy),
                 min(abs(cx - 1/3), abs(cx - 2/3)),
                 min(abs(cy - 1/3), abs(cy - 2/3)),
                 float(axis == 0), float(axis == 1)]
        out.append(vals)
    return np.asarray(out, np.float32)


def unit_target(unit, j, offsets):
    gf = {int(f): i for i, f in enumerate(unit["gf"])}
    kf = int(unit["keys"][int(unit["J"][j])])
    if kf not in gf: return None
    gt = unit["gt"][:, gf[kf], :]
    return iou(boxes(offsets, unit["W"], unit["H"], unit["ratio"]), gt).mean(1).astype(np.float32)


def table_units(root, name):
    return pickle.loads((root / "tables" / f"{name}.pkl").read_bytes())


def extract(root, names, limit=0, rv_frames=None):
    X=[]; Y=[]; SRC=[]; AX=[]; ROW=[]; t0=time.time(); count=0
    for name in names:
        units=table_units(root,name)
        if limit: units=units[:limit]
        for ui,u in enumerate(units):
            keys=[int(k) for k in u["keys"]]; frames=[]
            # Decode the entire keyframe sequence, then calculate all frame
            # maps and all 129 candidates as one NumPy batch.
            for j in u["J"]:
                kf=keys[int(j)]
                if u["ds"] == "live":
                    live_dir = "keyframes_val" if name == "live_confirm3" else "keyframes_train"
                    p=root / live_dir / str(u["vid"]) / f"{kf}.png"
                else:
                    p=rv_frames / "d1" / str(u["vid"]) / f"{kf}.png"
                frames.append(load_gray(p))
            if not frames:
                continue
            cur=np.stack(frames, axis=0)
            prev=np.concatenate([cur[:1], cur[:-1]], axis=0)
            batch_x=frame_features_batch(cur, prev, u["W"],u["H"],u["ratio"])
            for jj,j in enumerate(u["J"]):
                x=batch_x[jj]
                off,_,_,axis=candidates(u["W"],u["H"],u["ratio"]); y=unit_target(u,jj,off)
                if y is None: continue
                X.append(x);Y.append(y);SRC.extend([f'{u["ds"]}:{u["vid"]}']*len(y));AX.extend([int(axis if axis is not None else -1)]*len(y));ROW.extend([f'{name}:{u["vid"]}:{jj}']*len(y));count+=1
            if (ui+1)%100==0: print(json.dumps({"table":name,"units":ui+1,"rows":len(SRC),"elapsed_s":round(time.time()-t0,1)}),flush=True)
    return np.concatenate(X),np.concatenate(Y),np.asarray(SRC),np.asarray(AX),np.asarray(ROW)


def fit_ridge(x,y,src,axis,alpha=10.):
    # Per-axis standardized ridge; source-balanced row weights keep long clips
    # from dominating. Intercept is unpenalized.
    n,d=x.shape; mu=x.mean(0); sd=x.std(0); sd[sd<1e-5]=1.; z=(x-mu)/sd
    A=np.concatenate([z,np.ones((n,1),np.float32)],1)
    # One vectorized source count lookup; the previous list comprehension
    # compared every row with every source and dominated the fit on LIVE.
    if src is None:
        w=np.ones(n,np.float64)
    else:
        _, inv=np.unique(src,return_inverse=True); counts=np.bincount(inv)
        w=1.0/np.maximum(counts[inv],1)
    G=(A.T*w)@A; G[:-1,:-1]+=alpha*np.eye(d); b=(A.T*w)@y
    beta=np.linalg.solve(G,b).astype(np.float32); return beta,mu.astype(np.float32),sd.astype(np.float32)


def predict(x, model):
    beta,mu,sd=model; return np.concatenate([(x-mu[None,:])/sd[None,:],np.ones((len(x),1),np.float32)],1)@beta


def report(name,score,y,src,axis):
    vals=[]
    for s in sorted(set(src)):
        q=src==s
        k=max(1,int(round(.85*q.sum())))
        vals.append(float(y[q][np.argsort(-score[q],kind="mergesort")[:k]].mean()))
    # The candidate selected on every annotated frame, not an oracle.
    selected=y[np.argmax(score.reshape(-1,NC),axis=1)+np.arange(len(y))//NC*NC] if False else None
    return {"name":name,"rows":int(len(y)),"sources":int(len(set(src))),"row_score_rho":float(np.corrcoef(np.argsort(np.argsort(score)),np.argsort(np.argsort(y)))[0,1]),"source_macro_top85":float(np.mean(vals))}


def eval_rows(X,Y,S,A,models):
    out={}
    # X/Y are [rows,NC,D]; flatten is only for fitting; evaluation takes argmax.
    for name,model in models.items():
        pred=np.stack([predict(X[:,j,:],model) for j in range(NC)],1)
        jj=pred.argmax(1); chosen=Y[np.arange(len(Y)),jj]
        src=np.asarray(S); vals=[]
        for s in sorted(set(src)):
            q=src==s; k=max(1,int(round(.85*q.sum()))); vals.append(float(chosen[q][np.argsort(-pred[q].max(1),kind="mergesort")[:k]].mean()))
        out[name]={"rows":int(len(Y)),"sources":int(len(set(src))),"selected_mean_iou":float(chosen.mean()),"source_macro_selected_iou":float(np.mean(vals))}
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",type=Path,required=True); ap.add_argument("--rv-frames",type=Path,required=True); ap.add_argument("--out",type=Path,required=True); ap.add_argument("--limit",type=int,default=0); a=ap.parse_args()
    train_names=["rv_train","live_train"]; dev_names=["rv_dev","live_dev"]; confirm_names=["rv_confirm2","live_confirm3"]
    xt,yt,st,at,rt=extract(a.root,train_names,a.limit,a.rv_frames); xd,yd,sd,ad,rd=extract(a.root,dev_names,a.limit,a.rv_frames); xc,yc,sc,ac,rc=extract(a.root,confirm_names,a.limit,a.rv_frames)
    # Rebuild [unit rows, candidate, feature] because extraction appends 129 candidates per frame.
    def reshape(X,Y,S,A,R): return X.reshape(-1,NC,X.shape[-1]),Y.reshape(-1,NC),S.reshape(-1,NC)[:,0],A.reshape(-1,NC)[:,0]
    xt,yt,st,at=reshape(xt,yt,st,at,rt);xd,yd,sd,ad=reshape(xd,yd,sd,ad,rd);xc,yc,sc,ac=reshape(xc,yc,sc,ac,rc)
    models={}; metrics={}
    for label,mask in [("all",np.ones(len(xt),bool)),("no_motion",np.ones(len(xt),bool))]:
        cols=xt.shape[-1]; keep=np.arange(cols)
        if label=="no_motion":
            # remove the three motion-derived candidate stats and saliency
            # center's motion contribution is intentionally retained.
            keep=np.array([i for i in range(cols) if i not in (6,7,8)],dtype=int)
        b=fit_ridge(xt[:,:,keep].reshape(-1,len(keep)),yt.reshape(-1),np.repeat(st,NC),np.repeat(at,NC))
        models[label]=(b,keep); 
    out={"protocol":"R89_LOWLEVEL_MOTION_SALIENCY_COMPOSITION","train_units":len(xt),"dev_units":len(xd),"confirm_units":len(xc),"feature_dim":int(xt.shape[-1]),"models":{}}
    def eval_model(label,data):
        X,Y,S,A=data; beta,keep=models[label]; pred=np.stack([predict(X[:,j,keep],beta) for j in range(NC)],1); jj=pred.argmax(1); chosen=Y[np.arange(len(Y)),jj]; center=Y[:,NC//2]; vals=[]; cvals=[]; row_vals=[]; row_cvals=[]
        for s in sorted(set(S)):
            q=S==s;k=max(1,int(round(.85*q.sum()))); vals.append(float(chosen[q][np.argsort(-pred[q].max(1),kind='mergesort')[:k]].mean())); cvals.append(float(center[q][np.argsort(-center[q],kind='mergesort')[:k]].mean())); row_vals.append(float(chosen[q].mean())); row_cvals.append(float(center[q].mean()))
        return {"units":int(len(Y)),"sources":int(len(set(S))),"selected_mean_iou":float(chosen.mean()),"center_mean_iou":float(center.mean()),"source_macro_selected_iou":float(np.mean(row_vals)),"source_macro_center_iou":float(np.mean(row_cvals)),"delta_source_macro":float(np.mean(row_vals)-np.mean(row_cvals)),"top85_frame_selection_selected":float(np.mean(vals)),"top85_frame_selection_center":float(np.mean(cvals))}
    for label in models: out["models"][label]={"dev":eval_model(label,(xd,yd,sd,ad)),"confirm":eval_model(label,(xc,yc,sc,ac))}
    np.savez_compressed(a.out.with_suffix(".npz"), **{f"{k}_beta":v[0][0] for k,v in models.items()}, **{f"{k}_mu":v[0][1] for k,v in models.items()}, **{f"{k}_sd":v[0][2] for k,v in models.items()}, **{f"{k}_keep":v[1] for k,v in models.items()})
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))

if __name__=='__main__': main()
