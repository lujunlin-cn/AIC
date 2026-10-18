#!/usr/bin/env python3
import argparse, hashlib, json, math, tempfile, zipfile
from pathlib import Path
import numpy as np

def sha(p):
    h=hashlib.sha256();
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def geometry(W,H,r):
    rw,rh=map(float,r); w=min(float(W),float(H)*rw/rh); h=w*rh/rw
    return w,h,(0 if w < W-1e-9 else (1 if h < H-1e-9 else None))
def centres(points,keyframes,reset,raw,axis):
    n=len(reset); out=np.asarray(raw[:,axis] if raw.ndim==2 else raw,float).copy(); cur=None; j=-1
    kf=sorted(zip(keyframes,points))
    for t in range(n):
        if reset[t]: cur=None
        while j+1<len(kf) and kf[j+1][0] <= t:
            j+=1; p=kf[j][1]; cur=None if p is None or p[2] else float(p[axis])
        if cur is not None: out[t]=cur
    return out
def ema(c,reset):
    out=np.zeros(len(c)); prev=None
    for t,x in enumerate(c):
        if reset[t]: prev=None
        p=float(x) if prev is None else .25*float(x)+.75*prev; prev=float(np.clip(p,0,1)); out[t]=prev
    return out
def crops(c,W,H,r):
    w,h,axis=geometry(W,H,r); off=np.zeros(len(c))
    if axis==0: off=np.clip(c*W-w/2,0,W-w); return np.stack([off,np.zeros(len(c)),np.full(len(c),w)],1)
    if axis==1: off=np.clip(c*H-h/2,0,H-h); return np.stack([np.zeros(len(c)),off,np.full(len(c),w)],1)
    return np.tile(np.array([[0.,0.,w]]),(len(c),1))
def read_jsonl(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]

ap=argparse.ArgumentParser(); ap.add_argument('--parent',type=Path,required=True); ap.add_argument('--index',type=Path,required=True); ap.add_argument('--obs-cache',type=Path,required=True); ap.add_argument('--points',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
parent={str(x['video_id']):x for x in read_jsonl(a.parent)}; index={str(x['video_id']):x for x in read_jsonl(a.index)}
point_files=list(a.points.glob('*.json'))
point_files += list(a.points.glob('shard_*/points/*.json'))
point={}
for p in point_files: point[p.stem]=json.loads(p.read_text())
assert len(point)==426,(len(point),len(point_files))
rows=[]; changed=0; fallback=0
for vid,row in sorted(parent.items(),key=lambda kv:int(kv[0])):
    m=index[vid]; z=np.load(a.obs_cache/f'{vid}_t.npz'); W,H=int(z['W']),int(z['H']); r=list(map(float,m['targetRatioWH'])); p=point[vid]['ratios']['t']; c=centres(p['points'],point[vid]['keyframes'],z['reset'].astype(bool),z['raw'],geometry(W,H,r)[2] if geometry(W,H,r)[2] is not None else 0); bx=crops(ema(c,z['reset'].astype(bool)),W,H,r)
    fallback += sum(1 for s in p['status'] if s!='ok_r70')
    out=[]
    for q in row['predictions']:
        f=int(q['frame']); nq=dict(q); nb=[float(x) for x in bx[f]]
        if list(q['bboxes']) != nb: changed+=1
        nq['bboxes']=nb; out.append(nq)
    nr=dict(row); nr['predictions']=out; nr['model_params_b']=8.854531425; nr['model_size_mb']=8854.531425; rows.append(nr)
a.output.mkdir(parents=True,exist_ok=False); pred=a.output/'predictions.jsonl'
with pred.open('w') as f:
    for r in rows: f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
zip_path=a.output/(a.output.name+'.zip')
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z: z.write(pred,'predictions.jsonl')
with tempfile.TemporaryDirectory() as td:
    with zipfile.ZipFile(zip_path) as z: z.extractall(td)
    assert sha(Path(td)/'predictions.jsonl')==sha(pred)
man={'submission_id':a.output.name,'status':'PACKAGED_NOT_UPLOADED_R70_SPATIAL_READOUT','official_platform_score':None,'parent_submission':'R54_TIMELENS_CENTER_RESIDUAL_V9_B3','mechanism':'R70 public native-crop candidate utility head; R54 time mask fixed; only spatial boxes changed','parameter_components':[{'name':'TimeLens2-8B','parameters':8767123696},{'name':'V9 vision-only tower + R70 head + YuNet','parameters':87407729}], 'total_parameters':8854531425,'model_params_b':8.854531425,'video_count':len(rows),'prediction_count':sum(len(r['predictions']) for r in rows),'bbox_changed_on_kept_frames':changed,'r70_point_fallbacks':fallback,'prediction_sha256':sha(pred),'zip_sha256':sha(zip_path),'zip_bytes':zip_path.stat().st_size,'official_test_labels_used':False}
(a.output/'manifest.json').write_text(json.dumps(man,indent=2)+'\n'); (a.output/(zip_path.name+'.sha256')).write_text(sha(zip_path)+'  '+zip_path.name+'\n'); print(json.dumps(man,indent=2))
