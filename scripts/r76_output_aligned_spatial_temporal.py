#!/usr/bin/env python3
"""R76: output-aligned spatial utility drives temporal selection.

R70/R72/R73 used the max candidate utility (or max-center gain) while the
submitted spatial box remained the V9 B3 box.  This package evaluates the
R70 utility head at the actual V9 B3 offset by linear interpolation over the
129 legal windows, then adds a tie-aware rank residual to the exact R54 score.
It keeps the V9 B3 boxes and exact per-video budget unchanged.
"""
from __future__ import annotations
import argparse, hashlib, json, math, re, tempfile, zipfile
from pathlib import Path
import numpy as np

ETA=.20
TIME_LENS_PARAMS=8767123696
V9_PARAMS=87407729
SPAN_RE=re.compile(r"\[\s*\[.*?\]\s*\]")
NC=129

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def rows(p:Path): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

def parse(s):
 m=SPAN_RE.search(s or '')
 if not m:return []
 try:a=json.loads(m.group(0))
 except Exception:return []
 out=[]
 for x in a if isinstance(a,list) else []:
  try:
   aa,bb=float(x[0]),float(x[1])
   if math.isfinite(aa) and math.isfinite(bb) and bb>=aa: out.append((aa,bb))
  except Exception: pass
 return out

def geometry(W,H,ratio):
 rw,rh=map(float,ratio); target=rw/rh
 if W/H>target: return H*target,H,0
 if W/H<target: return W,W/target,1
 return W,H,None

def rank(a):
 a=np.asarray(a,float); return np.argsort(np.argsort(a,kind="mergesort"),kind="mergesort").astype(float)/max(len(a)-1,1)

def midrank(a):
 a=np.asarray(a,float); n=len(a)
 if n<=1:return np.zeros(n,float)
 o=np.argsort(a,kind='mergesort'); out=np.empty(n,float); i=0
 while i<n:
  j=i+1
  while j<n and a[o[j]]==a[o[i]]: j+=1
  out[o[i:j]]=0.5*(i+j-1)/max(n-1,1); i=j
 return out

def response_utility(path,n,meta,parent_row):
 d=json.loads(path.read_text()); t=d['ratios']['t']; kf=np.asarray(d['keyframes'],float); us=t['utility']
 if len(kf)!=len(us) or len(kf)<1 or any(s!='ok_r70' for s in t['status']): raise ValueError(f'{path}: utility contract')
 W,H=int(meta['width']),int(meta['height']); ratio=meta['targetRatioWH']; cw,ch,axis=geometry(W,H,ratio)
 offsets=np.zeros(len(kf),float); vals=[]
 by={int(x['frame']):x for x in parent_row['predictions']}
 for i,(frame,u) in enumerate(zip(kf,us)):
  fi=int(round(float(frame)))
  if fi not in by: raise ValueError(f'missing full parent frame {fi}')
  b=by[fi]['bboxes']; off=0.0 if axis is None else float(b[0] if axis==0 else b[1]); extent=0.0 if axis is None else float((W-cw) if axis==0 else (H-ch))
  q=0.0 if extent<=0 else np.clip(off/extent,0.0,1.0)*(NC-1)
  scores=np.asarray(u.get('scores',[]),float)
  if len(scores)!=NC: raise ValueError(f'{path}: missing 129 scores')
  vals.append(float(np.interp(q,np.arange(NC,dtype=float),scores)))
  offsets[i]=off
 frame=np.arange(n,dtype=float); scalar=np.interp(frame,kf,np.asarray(vals,float))
 return scalar, {'keyframes':int(len(kf)),'output_score_mean':float(np.mean(vals)),'output_score_std':float(np.std(vals)),'max_score_mean':float(np.mean([float(u['max_score']) for u in us])),'center_score_mean':float(np.mean([float(u['center_score']) for u in us])),'output_offset_mean':float(np.mean(offsets)),'axis':axis}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--parent',type=Path,required=True); ap.add_argument('--r54',type=Path,required=True); ap.add_argument('--index',type=Path,required=True); ap.add_argument('--responses',type=Path,required=True); ap.add_argument('--utility-dir',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--eta',type=float,default=ETA); a=ap.parse_args()
 intake={str(x['video_id']):x for x in rows(a.index)}; full={str(x['video_id']):x for x in rows(a.parent)}; old={str(x['video_id']):x for x in rows(a.r54)}; resp={str(x['video_id']):x.get('response','') for x in json.loads(a.responses.read_text())['results']}
 if set(full)!=set(intake) or set(full)!=set(old) or set(full)!=set(resp): raise ValueError('coverage mismatch')
 out=[]; details=[]; total=swaps=0
 for vid in sorted(full,key=lambda z:int(z) if z.isdigit() else z):
  meta=intake[vid]; r=full[vid]; n=int(meta['frame_count']); fps=float(meta['fps']); frames=np.arange(n,dtype=np.int64); times=frames/max(fps,1e-9); spans=parse(resp[vid]); inside=np.asarray([any(x-1e-6<=t<=y+1e-6 for x,y in spans) for t in times],float); centre=1-np.abs(times/max(float(times[-1]),1e-9)-.5)*2
  base=rank(centre)+.20*(rank(inside)-.5); k=int(round(.85*n)); base_idx=np.argsort(-base,kind='mergesort')[:k]; base_set=set(frames[base_idx].tolist()); old_set={int(x['frame']) for x in old[vid]['predictions']}
  if base_set!=old_set: raise ValueError(f'R54 mismatch {vid}')
  utility,us=response_utility(a.utility_dir/f'{vid}.json',n,meta,r); score=base+a.eta*(midrank(utility)-.5); idx=np.argsort(-score,kind='mergesort')[:k]; chosen=np.sort(frames[idx]); by={int(x['frame']):x for x in r['predictions']}; q=dict(r); q['predictions']=[by[int(f)] for f in chosen]; q['model_size_mb']=(TIME_LENS_PARAMS+V9_PARAMS)/1e6; q['model_params_b']=(TIME_LENS_PARAMS+V9_PARAMS)/1e9; out.append(q)
  ns=set(int(x) for x in chosen); ss=len(base_set^ns)//2; swaps+=ss; total+=k; details.append({'video_id':vid,'n_full':n,'n_keep':k,'swaps_vs_r54':ss,'overlap_vs_r54':len(base_set&ns)/max(k,1),'utility':us,'selected_output_utility_rank':float(midrank(utility)[idx].mean()),'selected_base_rank':float(rank(base)[idx].mean())})
 a.output.mkdir(parents=True,exist_ok=False); pred=a.output/'predictions.jsonl'; pred.write_text(''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in out)); zp=a.output/f'{a.output.name}.zip'
 with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z:z.write(pred,'predictions.jsonl')
 with tempfile.TemporaryDirectory() as td:
  with zipfile.ZipFile(zp) as z:z.extractall(td)
  if sha(Path(td)/'predictions.jsonl')!=sha(pred): raise RuntimeError('roundtrip')
 man={'submission_id':a.output.name,'status':'PACKAGED_NOT_UPLOADED','official_platform_score':None,'parent_submission':'R54_TIMELENS_CENTER_RESIDUAL_V9_B3','mechanism':'output-aligned R70 spatial utility residual: evaluate utility at V9 B3 output offset, then tie-aware rank residual in exact R54 time score','utility_definition':'linear interpolation of R70 129 candidate scores at the actual V9 B3 crop offset on 1-second keyframes, then original-frame interpolation','eta':a.eta,'utility_rank':'midrank exact ties','spatial_mode':'v9_b3_artifact','total_parameters':TIME_LENS_PARAMS+V9_PARAMS,'model_params_b':(TIME_LENS_PARAMS+V9_PARAMS)/1e9,'video_count':len(out),'prediction_count':total,'temporal_swaps_vs_r54':swaps,'prediction_sha256':sha(pred),'zip_sha256':sha(zp),'zip_bytes':zp.stat().st_size,'official_test_labels_used':False,'per_video':details}
 (a.output/'manifest.json').write_text(json.dumps(man,indent=2,ensure_ascii=False)+'\n'); (a.output/f'{zp.name}.sha256').write_text(sha(zp)+'  '+zp.name+'\n'); print(json.dumps({k:v for k,v in man.items() if k!='per_video'},indent=2))
if __name__=='__main__': main()
