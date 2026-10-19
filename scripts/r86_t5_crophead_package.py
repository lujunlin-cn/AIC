#!/usr/bin/env python3
"""R86: deploy T5 H1 point->window calibration on the current V9 observation path.

The H1 coefficients are fit only on the preregistered RetargetVid/LIVE native
crop tables (T5_CROPHEAD_V3).  This package keeps the R54 frame set and
parameter accounting fixed, and changes only the keyframe point mapping before
V9 hold+EMA decoding.  No official labels or scores are used.
"""
from __future__ import annotations
import argparse, hashlib, json, tempfile, zipfile
from pathlib import Path
import numpy as np


def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def geom(W,H,r):
    rw,rh=map(float,r); w=min(float(W),float(H)*rw/rh); h=w*rh/rw
    return w,h,(0 if w < W-1e-9 else (1 if h < H-1e-9 else None))
def center_to_offset(c,W,H,r):
    w,h,axis=geom(W,H,r)
    if axis is None:return np.zeros_like(c)
    L,s=(W,w) if axis==0 else (H,h)
    return np.clip(c*L-s/2,0,L-s)
def crops(off,W,H,r):
    w,h,axis=geom(W,H,r)
    if axis==0:return np.stack([off,np.zeros(len(off)),np.full(len(off),w)],1)
    if axis==1:return np.stack([np.zeros(len(off)),off,np.full(len(off),w)],1)
    return np.tile(np.array([[0.,0.,w]]),(len(off),1))
def qwen_centres(points,keyframes,reset,raw,axis):
    n=len(reset); out=np.asarray(raw[:,axis],float).copy(); cur=None; j=-1
    kf=sorted(zip(keyframes,points))
    for t in range(n):
        if reset[t]:cur=None
        while j+1<len(kf) and kf[j+1][0] <= t:
            j+=1; p=kf[j][1]; cur=None if p is None or p[2] else float(p[axis])
        if cur is not None:out[t]=cur
    return out
def ema(c,reset):
    out=np.zeros(len(c)); prev=None
    for t,x in enumerate(c):
        if reset[t]:prev=None
        p=float(x) if prev is None else .25*float(x)+.75*prev
        prev=float(np.clip(p,0,1)); out[t]=prev
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--parent',type=Path,required=True)
    ap.add_argument('--index',type=Path,required=True)
    ap.add_argument('--obs-cache',type=Path,required=True)
    ap.add_argument('--points',type=Path,required=True)
    ap.add_argument('--h1-metrics',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    parent={str(x['video_id']):x for x in rows(a.parent)}
    intake={str(x['video_id']):x for x in rows(a.index)}
    metrics=json.loads(a.h1_metrics.read_text())
    h1={str(k):tuple(v) for k,v in metrics['H1'].items()}
    point_files=list(a.points.glob('*.json'))+list(a.points.glob('points/*.json'))
    points={p.stem:json.loads(p.read_text()) for p in point_files}
    if set(parent)!=set(intake) or set(parent)!=set(points):raise ValueError('coverage mismatch')
    out_rows=[]; changed=0; frame_count=0; max_width_diff=0.; max_frame_delta=0.; per=[]
    for vid in sorted(parent,key=lambda x:int(x) if x.isdigit() else x):
        p=parent[vid]; z=np.load(a.obs_cache/f'{vid}_t.npz'); W,H=int(z['W']),int(z['H']); ratio=list(map(float,intake[vid]['targetRatioWH']))
        n=int(intake[vid]['frame_count']); pred=p['predictions']
        if len(pred)!=len({q['frame'] for q in pred}):raise ValueError(f'{vid}: duplicate parent frames')
        if any(int(q['frame'])<0 or int(q['frame'])>=n for q in pred):raise ValueError(f'{vid}: frame range')
        point=points[vid]; payload=point['ratios']['t']; _,_,axis=geom(W,H,ratio)
        if axis is None: bx=np.tile(np.array([[0.,0.,geom(W,H,ratio)[0]]]),(n,1))
        else:
            pts=[]; win=(geom(W,H,ratio)[0]/W if axis==0 else geom(W,H,ratio)[1]/H)
            a0,b0=h1[str(axis)]
            for q in payload['points']:
                if q is None: pts.append(None); continue
                qq=list(q)
                if not bool(qq[2]): qq[axis]=float(np.clip(a0*qq[axis]+(1-a0)*.5+b0*win,0,1))
                pts.append(qq)
            c=qwen_centres(pts,point['keyframes'],z['reset'].astype(bool),z['raw'],axis)
            bx=crops(center_to_offset(ema(c,z['reset'].astype(bool)),W,H,ratio),W,H,ratio)
        byf={int(q['frame']):q for q in pred}; new=[]; ch=0; wd=0.; fd=0.
        for q in pred:
            f=int(q['frame']); nq=dict(q); old=np.asarray(q['bboxes'],float); nb=bx[f].astype(float)
            ch += int(not np.array_equal(old,nb)); wd=max(wd,float(abs(old[2]-nb[2]))); fd=max(fd,float(np.max(abs(old-nb))))
            nq['bboxes']=[float(x) for x in nb]; new.append(nq)
        nr=dict(p); nr['predictions']=new; nr['model_params_b']=float(p.get('model_params_b',8.854531425)); nr['model_size_mb']=float(p.get('model_size_mb',8854.531425)); out_rows.append(nr)
        changed+=ch; frame_count+=len(pred); max_width_diff=max(max_width_diff,wd); max_frame_delta=max(max_frame_delta,fd)
        per.append({'video_id':vid,'kept':len(pred),'changed':ch,'max_bbox_delta':fd})
    a.output.mkdir(parents=True,exist_ok=False)
    pred=a.output/'predictions.jsonl'; pred.write_text(''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in out_rows))
    zp=a.output/(a.output.name+'.zip')
    with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z:z.write(pred,'predictions.jsonl')
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zp) as z:z.extractall(td)
        if sha(Path(td)/'predictions.jsonl')!=sha(pred):raise RuntimeError('roundtrip mismatch')
    man={'submission_id':a.output.name,'status':'PACKAGED_NOT_UPLOADED','official_platform_score':None,'parent_submission':'R54_TIMELENS_CENTER_RESIDUAL_V9_B3','mechanism':'T5_CROPHEAD_V3 H1 source-disjoint native-crop point-to-window calibration on current V9 observation path; R54 frame mask fixed','h1_metrics_sha256':sha(a.h1_metrics),'parent_predictions_sha256':sha(a.parent),'index_sha256':sha(a.index),'video_count':len(out_rows),'prediction_count':frame_count,'bbox_changed_on_kept_frames':changed,'max_width_delta':max_width_diff,'max_bbox_delta':max_frame_delta,'total_parameters':sum(float(x.get('parameters',0)) for x in p.get('parameter_components',[])) if False else 8854531425,'model_params_b':8.854531425,'official_test_labels_used':False,'prediction_sha256':sha(pred),'zip_sha256':sha(zp),'zip_bytes':zp.stat().st_size,'per_video':per}
    (a.output/'manifest.json').write_text(json.dumps(man,indent=2)+'\n'); (a.output/(zp.name+'.sha256')).write_text(sha(zp)+'  '+zp.name+'\n')
    print(json.dumps({k:v for k,v in man.items() if k!='per_video'},indent=2))
if __name__=='__main__':main()
