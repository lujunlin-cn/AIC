#!/usr/bin/env python3
"""Build five content-driven temporal masks over the frozen B3 parent.

The five modes are fixed mechanisms, not a keep/threshold sweep:
event_peak, event_window, redundancy, novelty_coverage, and gated_fusion.
All modes keep exactly floor(0.85*N) frames per video and copy B3 boxes.
"""
import argparse, json, math, shutil, zipfile
from pathlib import Path
import numpy as np

def z(x):
    x=np.asarray(x,float); s=x.std()
    return (x-x.mean())/(s+1e-8)

def feats(root, vid):
    fs=[]
    for p in Path(root, str(vid)).glob('*.npy'):
        try: fs.append((int(p.stem), np.load(p).astype('float32')))
        except ValueError: pass
    fs.sort();
    if not fs: return np.empty(0,int), np.empty((0,1),float)
    a=np.stack([x[1] for x in fs]); a/=np.linalg.norm(a,axis=1,keepdims=True)+1e-8
    return np.array([x[0] for x in fs]), a

def interpolate(frames, pts, vals):
    if len(pts)==0: return np.zeros(len(frames))
    return np.interp(frames, pts, vals, left=vals[0], right=vals[-1])

def select(mode, frames, pts, emb, target, replay, exact):
    n=len(frames); k=max(1, int(math.floor(.85*n)))
    if len(pts)<2:
        order=np.arange(n)
    else:
        d=1-(emb[:-1]*emb[1:]).sum(1)
        # associate change with the midpoint of the transition
        mid=(pts[:-1]+pts[1:])/2
        vel=interpolate(frames,mid,d)
        acc=np.maximum(0, np.gradient(np.gradient(vel)))
        if mode=='event_peak': score=z(vel)+0.5*z(acc)
        elif mode=='event_window': score=interpolate(frames,mid,d)
        elif mode=='redundancy':
            # local redundancy is low when either neighbour changes; retain it.
            local=np.convolve(vel,np.ones(9)/9,mode='same')
            score=z(vel)+0.8*z(local)
        elif mode=='novelty_coverage':
            # cumulative state novelty: a frame is useful when its state differs
            # from the preceding and following observed states.
            novelty=np.zeros(n)
            for j,f in enumerate(frames):
                q=np.searchsorted(pts,f); q=min(max(q,1),len(pts)-1)
                novelty[j]=d[q-1]
            score=z(novelty)+0.35*z(np.abs(np.gradient(novelty)))
        elif mode=='gated_fusion':
            # content evidence gates the frozen weak temporal head: replay/exact
            # frames are retained only when their local content is not redundant.
            prior=np.zeros(n)
            rs=set(replay); es=set(exact)
            prior=np.array([1.0 if f in rs else (0.6 if f in es else 0.) for f in frames])
            score=z(vel)+0.8*z(prior)
        else: raise ValueError(mode)
    # deterministic tie break by original frame number
    order=np.lexsort((frames, -score))
    chosen=set(frames[order[:k]].tolist())
    return chosen

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--parent',required=True); ap.add_argument('--index',required=True)
    ap.add_argument('--features',required=True); ap.add_argument('--replay',required=True); ap.add_argument('--exact',required=True)
    ap.add_argument('--out',required=True); ap.add_argument('--size',type=int,default=450529778)
    a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    parent={r['video_id']:r for r in map(json.loads,open(a.parent))}
    replay={r['video_id']:[x['frame'] for x in r['predictions']] for r in map(json.loads,open(a.replay))}
    exact={r['video_id']:[x['frame'] for x in r['predictions']] for r in map(json.loads,open(a.exact))}
    modes=['event_peak','event_window','redundancy','novelty_coverage','gated_fusion']
    allstats={}
    for mode in modes:
        rows=[]; total=0; changed=0
        for vid in sorted(parent,key=int):
            p=parent[vid]; pred=p['predictions']; frames=np.array([x['frame'] for x in pred],int)
            pts,e=feats(a.features,vid)
            chosen=select(mode,frames,pts,e,0.85,replay.get(vid,[]),exact.get(vid,[]))
            q=[x for x in pred if x['frame'] in chosen]
            rows.append({'video_id':vid,'targetRatioWH':p['targetRatioWH'],'model_size_mb':a.size/1e6,'predictions':q})
            total+=len(q); changed+=len(pred)-len(q)
        d=out/mode; d.mkdir(exist_ok=True)
        pred=d/'predictions.jsonl'; pred.write_text(''.join(json.dumps(x,separators=(',',':'))+'\n' for x in rows))
        zpath=d/(mode+'.zip')
        with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as zf: zf.write(pred,'predictions.jsonl')
        man={'candidate':mode,'status':'PACKAGED_NOT_UPLOADED','mechanism':'content-driven temporal selection over frozen 768D visual states','parent':'R26_POS_CENTER_EXACT085_B3 / B3 spatial predictions','budget':'floor(0.85*N) per video','video_count':len(rows),'prediction_count':total,'dropped':changed,'total_parameters':a.size,'features':str(a.features),'replay':str(a.replay),'exact':str(a.exact),'zip':str(zpath)}
        (d/'manifest.json').write_text(json.dumps(man,indent=2)+'\n'); allstats[mode]=man
    (out/'summary.json').write_text(json.dumps(allstats,indent=2)+'\n')
if __name__=='__main__': main()
