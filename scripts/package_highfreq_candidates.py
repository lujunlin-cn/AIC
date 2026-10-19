#!/usr/bin/env python3
"""Package the fixed 4fps semantic residual and its boundary guarded mate.

The builder never changes boxes or the exact per-video budget.  The second
package is a fixed 5 percent edge-band intervention using the first package as
the candidate and R26 as the parent.  It is intentionally separate from the
R45 builder so manifests name the actual high-frequency mechanism.
"""
from __future__ import annotations
import argparse, hashlib, json, math, tempfile, zipfile
from pathlib import Path

KEEP=.85; B3_PARAMS=450283377; IV2_PARAMS=6816814181; BAND=.05

def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def read(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def filt(base,cand,n):
 bs={int(p['frame']) for p in base}; cs={int(p['frame']) for p in cand}; add=sorted(cs-bs); rem=sorted(bs-cs)
 if len(add)!=len(rem): raise ValueError('unbalanced')
 if not add: return sorted(bs),0,0
 lo,hi=min(bs),max(bs); band=max(1,int(math.ceil(BAND*n)))
 aa=[x for x in add if lo-band<=x<lo or hi<x<=hi+band]
 rr=[x for x in rem if lo<=x<lo+band or hi-band<x<=hi]
 q=min(len(aa),len(rr)); dist=lambda x:(min(abs(x-lo),abs(x-hi)),x)
 return sorted((bs-set(sorted(rr,key=dist)[:q]))|set(sorted(aa,key=dist)[:q])),q,len(add)-q
def package(rows,out,name,parent_name,mechanism,params):
 out.mkdir(parents=True,exist_ok=False); pred=out/'predictions.jsonl'
 with pred.open('w') as f:
  for source in rows:
   # New-stage rules require the total loaded inference parameter count in B.
   # Keep the historical MB field synchronized for older local validators.
   r=dict(source)
   r['model_params_b']=params/1e9
   r['model_size_mb']=params/1e6
   f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
 zp=out/f'{name}.zip'
 with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z:z.write(pred,'predictions.jsonl')
 with tempfile.TemporaryDirectory() as td:
  with zipfile.ZipFile(zp) as z:z.extractall(td)
  if sha(Path(td)/'predictions.jsonl')!=sha(pred):raise ValueError('roundtrip')
 m={'submission_id':name,'status':'PACKAGED_NOT_UPLOADED','uploaded':False,'official_platform_score':None,
    'parent_submission':parent_name,'mechanism':mechanism,'changed_factor':'temporal frame ranking only; B3 spatial boxes copied from R26',
    'keep_rule':'round(0.85 * full original frame count)','video_count':len(rows),'prediction_count':sum(len(r['predictions']) for r in rows),
    'total_parameters':params,'model_params_b':params/1e9,'model_size_mb':params/1e6,'predictions_sha256':sha(pred),'zip_sha256':sha(zp),'zip_bytes':zp.stat().st_size,
    'spatial_boxes_source':'R26_POS_CENTER_EXACT085_B3'}
 (out/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n'); (out/(zp.name+'.sha256')).write_text(sha(zp)+'  '+zp.name+'\n')
 return m
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--r26',type=Path,required=True); ap.add_argument('--candidate',type=Path,required=True); ap.add_argument('--local-candidate',type=Path); ap.add_argument('--nfull-manifest',type=Path,required=True); ap.add_argument('--output-root',type=Path,required=True); args=ap.parse_args()
 base=read(args.r26); cand=read(args.candidate); nfull={str(x['video_id']):int(x['n_full']) for x in json.loads(args.nfull_manifest.read_text())['per_video']}
 bm={str(r['video_id']):r for r in base}; cm={str(r['video_id']):r for r in cand};
 if set(bm)!=set(cm) or len(bm)!=426:raise ValueError('coverage')
 full=[]; boundary=[]; swaps=0
 for v in sorted(bm,key=lambda x:int(x) if x.isdigit() else x):
  br,cr=bm[v],cm[v]; bp={int(p['frame']):p for p in br['predictions']}; cp={int(p['frame']):p for p in cr['predictions']}; n=nfull[v]; expect=round(KEEP*n)
  if len(bp)!=expect or len(cp)!=expect: raise ValueError(f'{v} budget')
  for f in set(bp)&set(cp):
   if bp[f].get('bboxes')!=cp[f].get('bboxes'): raise ValueError(f'{v} box changed')
  r=dict(cr); full.append(r)
  chosen,q,_=filt(list(bp.values()),list(cp.values()),n); boundary.append({**dict(br),'predictions':[bp[f] if f in bp else cp[f] for f in chosen]}); swaps+=q
 root=args.output_root; root.mkdir(parents=True,exist_ok=True)
 m1=package(full,root/'R46_HIGHFREQ_SEMANTIC_RESIDUAL_B3','R46_HIGHFREQ_SEMANTIC_RESIDUAL_B3','R26_POS_CENTER_EXACT085_B3','4fps local state + frozen IV2 event anchor residual, eta=.20',B3_PARAMS+IV2_PARAMS)
 m2=package(boundary,root/'R46_HIGHFREQ_BOUNDARY_GUARDED_B3','R46_HIGHFREQ_BOUNDARY_GUARDED_B3','R26_POS_CENTER_EXACT085_B3','same 4fps semantic residual restricted to fixed 5 percent R26 edge bands',B3_PARAMS+IV2_PARAMS)
 m2['retained_boundary_swaps']=swaps; (root/'R46_HIGHFREQ_BOUNDARY_GUARDED_B3'/'manifest.json').write_text(json.dumps(m2,ensure_ascii=False,indent=2)+'\n')
 if args.local_candidate:
  lc=read(args.local_candidate); lm={str(r['video_id']):r for r in lc}; local=[]; local_b=[]
  if set(lm)!=set(bm): raise ValueError('local coverage')
  for v in sorted(bm,key=lambda x:int(x) if x.isdigit() else x):
   br=bm[v]; lr=lm[v]; bp={int(p['frame']):p for p in br['predictions']}; lp={int(p['frame']):p for p in lr['predictions']}; n=nfull[v]
   if len(lp)!=round(KEEP*n): raise ValueError(f'{v} local budget')
   if any(bp[f].get('bboxes')!=lp[f].get('bboxes') for f in set(bp)&set(lp)): raise ValueError(f'{v} local box changed')
   local.append(dict(lr)); chosen,_,_=filt(list(bp.values()),list(lp.values()),n); local_b.append({**dict(br),'predictions':[bp[f] if f in bp else lp[f] for f in chosen]})
  ml=package(local,root/'R46_HIGHFREQ_LOCAL_RESIDUAL_B3','R46_HIGHFREQ_LOCAL_RESIDUAL_B3','R26_POS_CENTER_EXACT085_B3','4fps local state residual without semantic anchor, eta=.20',B3_PARAMS)
  mb=package(local_b,root/'R46_HIGHFREQ_LOCAL_BOUNDARY_GUARDED_B3','R46_HIGHFREQ_LOCAL_BOUNDARY_GUARDED_B3','R26_POS_CENTER_EXACT085_B3','same local 4fps residual restricted to fixed 5 percent R26 edge bands',B3_PARAMS)
  print(json.dumps({'local':ml,'local_boundary':mb},ensure_ascii=False,indent=2))
 print(json.dumps({'full':m1,'boundary':m2},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
