#!/usr/bin/env python3
"""R106-R108: event topology decoders built from two frozen TimeLens views.

These are deliberately different from the old inside-mask/phase/edge
residuals.  They use the geometry of interval boundaries and the disagreement
between independent 2fps and 4fps observations.  The V9/B3 full-frame boxes
and the exact 85% per-video budget remain fixed, so a package changes only the
temporal decision.  No official labels or platform scores are read.

Routes:
  shell      boundary proximity on both sides of a predicted event;
  lifecycle  onset shell + completion phase + boundary shell;
  disagreement  cross-rate interval disagreement plus consensus shell.
"""
from __future__ import annotations
import argparse, hashlib, json, math, re, tempfile, zipfile
from pathlib import Path
import numpy as np

SPAN_RE = re.compile(r"\[\s*\[.*?\]\s*\]")
PARAMS = 8_854_531_425

def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def rank01(x: np.ndarray) -> np.ndarray:
    x=np.asarray(x,float)
    return np.argsort(np.argsort(x,kind='mergesort'),kind='mergesort').astype(float)/max(len(x)-1,1)

def parse(text: str) -> list[tuple[float,float]]:
    m=SPAN_RE.search(text or '')
    if not m: return []
    try: a=json.loads(m.group(0))
    except Exception: return []
    out=[]
    for z in a:
        if len(z)>=2:
            x,y=float(z[0]),float(z[1])
            if math.isfinite(x) and math.isfinite(y) and y>=x: out.append((x,y))
    return out

def mask(t: np.ndarray, iv: list[tuple[float,float]]) -> np.ndarray:
    return np.asarray([any(a-1e-6<=x<=b+1e-6 for a,b in iv) for x in t],float)

def shell(t: np.ndarray, iv: list[tuple[float,float]], tau: float=1.5) -> np.ndarray:
    out=np.zeros(len(t),float)
    for a,b in iv:
        # Distance to the nearest event boundary, including the event itself.
        d=np.minimum(np.abs(t-a),np.abs(t-b))
        out=np.maximum(out,np.exp(-d/max(tau,1e-6)))
    return out

def lifecycle(t: np.ndarray, iv: list[tuple[float,float]]) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    onset=np.zeros(len(t),float); completion=np.zeros(len(t),float); edge=np.zeros(len(t),float)
    for a,b in iv:
        inside=(t>=a-1e-6)&(t<=b+1e-6); p=np.clip((t-a)/max(b-a,1e-6),0,1)
        onset=np.maximum(onset,np.where(inside,1-p,0.0))
        completion=np.maximum(completion,np.where(inside,p,0.0))
        edge=np.maximum(edge,np.where(inside,2*np.abs(p-.5),0.0))
    return onset,completion,edge

def route_score(t: np.ndarray, iv2: list[tuple[float,float]], iv4: list[tuple[float,float]], route: str) -> np.ndarray:
    # Centre is retained as a weak safety prior.  All other terms are fixed,
    # rank-normalised, and video-local; no global threshold is introduced.
    centre=1.0-np.abs(t/max(float(t[-1]),1e-9)-0.5)*2.0
    m2=mask(t,iv2); m4=mask(t,iv4)
    s2=shell(t,iv2); s4=shell(t,iv4)
    onset,complete,edge=lifecycle(t,iv4)
    if route=='shell':
        # A high value just before/inside/after a predicted event keeps the
        # action transition and visible result instead of its static middle.
        q=np.maximum(s2,s4)
        return rank01(centre)+0.35*(rank01(q)-0.5)
    if route=='lifecycle':
        # The 4fps view supplies completion timing; the 2fps view supplies a
        # second boundary estimate.  This is a fixed event lifecycle policy.
        q=0.35*rank01(onset)+0.35*rank01(complete)+0.30*rank01(np.maximum(s2,s4))
        return rank01(centre)+0.35*(q-0.5)
    if route=='disagreement':
        # Sampling-rate disagreement marks unresolved transitions.  Consensus
        # shell suppresses isolated parser noise while retaining both sides.
        disagree=np.abs(m2-m4)+0.5*np.abs(s2-s4)
        consensus=np.minimum(s2,s4)
        q=0.60*rank01(disagree)+0.40*rank01(consensus)
        return rank01(centre)+0.45*(q-0.5)
    raise ValueError(route)

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument('--responses2',type=Path,required=True); ap.add_argument('--responses4',type=Path,required=True); ap.add_argument('--parent',type=Path,required=True); ap.add_argument('--index',type=Path,required=True); ap.add_argument('--route',choices=['shell','lifecycle','disagreement'],required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    def load(p):
        d=json.loads(p.read_text()); return {str(x['video_id']):parse(x.get('response','')) for x in d.get('results',[])}
    r2,r4=load(a.responses2),load(a.responses4)
    idx={str(json.loads(x)['video_id']):json.loads(x) for x in a.index.read_text().splitlines() if x.strip()}
    parent=[json.loads(x) for x in a.parent.read_text().splitlines() if x.strip()]
    if set(r2)!=set(idx) or set(r4)!=set(idx) or set(str(x['video_id']) for x in parent)!=set(idx): raise ValueError('coverage mismatch')
    out=[]; total=0; changed=0; details=[]
    key=lambda x:int(str(x['video_id'])) if str(x['video_id']).isdigit() else str(x['video_id'])
    for row in sorted(parent,key=key):
        vid=str(row['video_id']); meta=idx[vid]; allp=row['predictions']; n=int(meta['frame_count'])
        if len(allp)!=n: raise ValueError(f'{vid}: parent is not full-frame')
        frames=np.asarray([int(p['frame']) for p in allp],np.int64); fps=float(meta['fps']); t=frames/fps
        sc=route_score(t,r2[vid],r4[vid],a.route); k=int(round(.85*n)); chosen=np.sort(frames[np.argsort(-sc,kind='mergesort')[:k]])
        by={int(p['frame']):p for p in allp}; q=dict(row); q['predictions']=[by[int(f)] for f in chosen]; q['model_size_mb']=PARAMS/1e6; q['model_params_b']=PARAMS/1e9
        out.append(q); total+=k
        old=np.asarray([int(p['frame']) for p in allp[:k]])
        ch=int(len(set(map(int,chosen))-set(map(int,old)))); changed+=ch
        details.append({'video_id':vid,'n_full':n,'n_keep':k,'changed_vs_full_prefix':ch,'intervals_2fps':r2[vid],'intervals_4fps':r4[vid],'selected_in_2fps':int(mask(t,r2[vid])[np.isin(frames,chosen)].sum()),'selected_in_4fps':int(mask(t,r4[vid])[np.isin(frames,chosen)].sum())})
    a.output.mkdir(parents=False,exist_ok=False); pred=a.output/'predictions.jsonl'; pred.write_text(''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in out))
    zp=a.output/(a.output.name+'.zip')
    with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z:z.write(pred,'predictions.jsonl')
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zp) as z:z.extractall(td)
        if sha(Path(td)/'predictions.jsonl')!=sha(pred): raise RuntimeError('roundtrip mismatch')
    man={'submission_id':a.output.name,'status':'PACKAGED_NOT_UPLOADED','official_platform_score':None,'parent_submission':'R54_TIMELENS_CENTER_RESIDUAL_V9_B3','mechanism':{'shell':'two-resolution event-boundary shell; rank(center)+.35 rank(exp(-distance-to-boundary/1.5s))','lifecycle':'event lifecycle onset/completion/boundary fusion from 2fps+4fps intervals','disagreement':'cross-resolution interval disagreement and consensus shell; rank(center)+.45 rank(disagreement/consensus)' }[a.route],'route':a.route,'responses_2fps_sha256':sha(a.responses2),'responses_4fps_sha256':sha(a.responses4),'parent_full_predictions_sha256':sha(a.parent),'index_sha256':sha(a.index),'video_count':len(out),'prediction_count':total,'changed_vs_full_prefix':changed,'total_parameters':PARAMS,'model_params_b':PARAMS/1e9,'official_test_labels_used':False,'prediction_sha256':sha(pred),'zip_sha256':sha(zp),'zip_bytes':zp.stat().st_size,'per_video':details}
    (a.output/'manifest.json').write_text(json.dumps(man,ensure_ascii=False,indent=2)+'\n'); (a.output/(zp.name+'.sha256')).write_text(sha(zp)+'  '+zp.name+'\n')
    print(json.dumps({k:v for k,v in man.items() if k!='per_video'},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
