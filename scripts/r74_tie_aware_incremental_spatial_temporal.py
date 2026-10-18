#!/usr/bin/env python3
"""R74: tie-aware incremental spatial utility for temporal selection.

R70/R72 use absolute best-crop score max_score.  This candidate uses the
marginal crop gain over the center legal window (max_score-center_score), then
adds it as a fixed residual to R54's exact temporal score.  The center term is
already the spatial baseline in the historical R54/V9 pipeline; absolute
max_score can reward frames whose crop is merely easy/redundant.  No official
labels are read and the 85% budget and R54 parent boxes remain fixed.
"""
from __future__ import annotations
import argparse,hashlib,json,math,re,tempfile,zipfile
from pathlib import Path
import numpy as np
ETA=.20; TIME_LENS_PARAMS=8767123696; V9_PARAMS=87407729
SPAN_RE=re.compile(r"\[\s*\[.*?\]\s*\]")
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def parse(s):
 m=SPAN_RE.search(s or '');
 if not m:return []
 try:a=json.loads(m.group(0))
 except:return []
 return [(float(x[0]),float(x[1])) for x in a if len(x)>=2 and math.isfinite(float(x[0])) and math.isfinite(float(x[1])) and float(x[1])>=float(x[0])]
def rank(a):
 a=np.asarray(a,float);return np.argsort(np.argsort(a,kind='mergesort'),kind='mergesort').astype(float)/max(len(a)-1,1)
def rank_mid(a):
 a=np.asarray(a,float); n=len(a)
 if n<=1:return np.zeros(n,float)
 order=np.argsort(a,kind='mergesort'); out=np.empty(n,float); i=0
 while i<n:
  j=i+1
  while j<n and a[order[j]]==a[order[i]]: j+=1
  out[order[i:j]]=(0.5*(i+j-1))/(n-1); i=j
 return out
def utility(path,n):
 d=json.loads(path.read_text());t=d['ratios']['t'];kf=np.asarray(d['keyframes'],float);u=t['utility'];
 if len(kf)!=len(u) or len(kf)<2:raise ValueError(path)
 mx=np.asarray([float(x['max_score']) for x in u]);cen=np.asarray([float(x['center_score']) for x in u]);
 gain=np.maximum(mx-cen,0.0); # confidence: only positive improvement over center is spatially additive
 # Keyframe-to-frame interpolation is the existing R70 contract.
 fr=np.arange(n,dtype=float); return np.interp(fr,kf,gain),{'keyframes':len(kf),'gain_mean':float(gain.mean()),'gain_std':float(gain.std()),'gain_nonzero':float((gain>1e-6).mean())}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--parent',type=Path,required=True);ap.add_argument('--r54',type=Path,required=True);ap.add_argument('--index',type=Path,required=True);ap.add_argument('--responses',type=Path,required=True);ap.add_argument('--utility-dir',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--eta',type=float,default=ETA);a=ap.parse_args()
 intake={str(x['video_id']):x for x in rows(a.index)};full={str(x['video_id']):x for x in rows(a.parent)};old={str(x['video_id']):x for x in rows(a.r54)};resp={str(x['video_id']):x.get('response','') for x in json.loads(a.responses.read_text())['results']}
 if set(full)!=set(intake)|set():
  if set(full)!=set(intake):raise ValueError('coverage')
 if set(full)!=set(resp) or set(full)!=set(old):raise ValueError('coverage')
 out=[];details=[];tot=swaps=0
 for vid in sorted(full,key=lambda z:int(z) if z.isdigit() else z):
  meta=intake[vid];r=full[vid];n=int(meta['frame_count']);fps=float(meta['fps']);frames=np.arange(n,dtype=np.int64);times=frames/max(fps,1e-9);sp=parse(resp[vid]);inside=np.asarray([any(x-1e-6<=t<=y+1e-6 for x,y in sp) for t in times],float);centre=1-np.abs(times/max(float(times[-1]),1e-9)-.5)*2
  base=rank(centre)+.20*(rank(inside)-.5);k=int(round(.85*n));base_idx=np.argsort(-base,kind='mergesort')[:k];base_set=set(frames[base_idx].tolist());old_set={int(x['frame']) for x in old[vid]['predictions']}
  if base_set!=old_set:raise ValueError(f'R54 mismatch {vid}')
  gain,us=utility(a.utility_dir/f'{vid}.json',n);score=base+a.eta*(rank_mid(gain)-.5);idx=np.argsort(-score,kind='mergesort')[:k];chosen=np.sort(frames[idx]);by={int(x['frame']):x for x in r['predictions']};q=dict(r);q['predictions']=[by[int(f)] for f in chosen];q['model_size_mb']=(TIME_LENS_PARAMS+V9_PARAMS)/1e6;q['model_params_b']=(TIME_LENS_PARAMS+V9_PARAMS)/1e9;out.append(q)
  ns=set(int(x) for x in chosen);ss=len(base_set^ns)//2;swaps+=ss;tot+=k;details.append({'video_id':vid,'n_full':n,'n_keep':k,'swaps_vs_r54':ss,'overlap_vs_r54':len(base_set&ns)/max(k,1),'gain':us,'selected_gain_rank':float(rank(gain)[idx].mean()),'selected_base_rank':float(rank(base)[idx].mean())})
 a.output.mkdir(parents=True,exist_ok=False);pred=a.output/'predictions.jsonl';pred.write_text(''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in out));zp=a.output/f'{a.output.name}.zip';
 with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z:z.write(pred,'predictions.jsonl')
 with tempfile.TemporaryDirectory() as td:
  with zipfile.ZipFile(zp) as z:z.extractall(td)
  if sha(Path(td)/'predictions.jsonl')!=sha(pred):raise RuntimeError('roundtrip')
 man={'submission_id':a.output.name,'status':'PACKAGED_NOT_UPLOADED','official_platform_score':None,'parent_submission':'R54_TIMELENS_CENTER_RESIDUAL_V9_B3','mechanism':'tie-aware incremental spatial utility temporal residual: midrank(max_score-center_score) added to exact R54 score','utility_definition':'positive marginal predicted crop IoU over center legal window; 129-window R70 topsoft utility; equal gains receive midrank','eta':a.eta,'spatial_mode':'v9_b3_artifact','total_parameters':TIME_LENS_PARAMS+V9_PARAMS,'model_params_b':(TIME_LENS_PARAMS+V9_PARAMS)/1e9,'video_count':len(out),'prediction_count':tot,'temporal_swaps_vs_r54':swaps,'prediction_sha256':sha(pred),'zip_sha256':sha(zp),'zip_bytes':zp.stat().st_size,'utility_rank':'midrank with exact ties averaged; prevents stable frame-index order on clipped gains','official_test_labels_used':False,'per_video':details}
 (a.output/'manifest.json').write_text(json.dumps(man,indent=2,ensure_ascii=False)+'\n');(a.output/f'{zp.name}.sha256').write_text(sha(zp)+'  '+zp.name+'\n');print(json.dumps({k:v for k,v in man.items() if k!='per_video'},indent=2))
if __name__=='__main__':main()
