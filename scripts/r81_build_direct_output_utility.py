#!/usr/bin/env python3
"""Build an output-aligned direct utility temporal candidate.

The direct utility head was trained on public source-disjoint native crop data
with target IoU of the deployed V9 B3 crop. Official inference uses only the
R70 topsoft candidate logits at 1-second keyframes plus the actual V9 B3 crop
position from the audited full-frame parent. No official labels are read.
"""
from __future__ import annotations
import argparse,hashlib,json,math,re,tempfile,zipfile
from pathlib import Path
import numpy as np, torch
KEEP=.85; ETA_PARAMS=8767123696; V9_PARAMS=87407729; R70_PARAMS=1511425; DIRECT_PARAMS=50561; NC=129
SPAN_RE=re.compile(r"\[\s*\[.*?\]\s*\]")
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def parse(s):
 m=SPAN_RE.search(s or '')
 if not m:return []
 try:a=json.loads(m.group(0))
 except Exception:return []
 out=[]
 for z in a if isinstance(a,list) else []:
  try:aa,bb=float(z[0]),float(z[1]);
  except Exception:continue
  if math.isfinite(aa) and math.isfinite(bb) and bb>=aa:out.append((aa,bb))
 return out
def rank(a):
 a=np.asarray(a,float);return np.argsort(np.argsort(a,kind='mergesort'),kind='mergesort').astype(float)/max(len(a)-1,1)
def midrank(a):
 a=np.asarray(a,float);n=len(a)
 if n<=1:return np.zeros(n,float)
 o=np.argsort(a,kind='mergesort');r=np.empty(n,float);i=0
 while i<n:
  j=i+1
  while j<n and a[o[j]]==a[o[i]]:j+=1
  r[o[i:j]]=0.5*(i+j-1)/max(n-1,1);i=j
 return r
def geometry(W,H,ratio):
 rw,rh=map(float,ratio);t=rw/rh
 if W/H>t:return H*t,H,0
 if W/H<t:return W,W/t,1
 return W,H,None
class Direct(torch.nn.Module):
 def __init__(self):
  super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(130,256),torch.nn.GELU(),torch.nn.LayerNorm(256),torch.nn.Linear(256,64),torch.nn.GELU(),torch.nn.Linear(64,1))
 def forward(self,x):return self.net(x).squeeze(-1)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--parent',type=Path,required=True);ap.add_argument('--r54',type=Path,required=True);ap.add_argument('--index',type=Path,required=True);ap.add_argument('--responses',type=Path,required=True);ap.add_argument('--utility-dir',type=Path,required=True);ap.add_argument('--head',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 intake={str(x['video_id']):x for x in rows(a.index)};full={str(x['video_id']):x for x in rows(a.parent)};old={str(x['video_id']):x for x in rows(a.r54)};resp={str(x['video_id']):x.get('response','') for x in json.loads(a.responses.read_text())['results']}
 if set(full)!=set(intake) or set(full)!=set(old) or set(full)!=set(resp):raise ValueError('coverage mismatch')
 ck=torch.load(a.head,map_location='cpu',weights_only=False);m=Direct();m.load_state_dict(ck['state_dict']);m.eval();mu=torch.from_numpy(np.asarray(ck['mu'],np.float32));sd=torch.from_numpy(np.asarray(ck['sd'],np.float32))
 out=[];details=[];total=swaps=0;all_util=[]
 for vid in sorted(full,key=lambda x:int(x) if x.isdigit() else x):
  meta=intake[vid];parent=full[vid];n=int(meta['frame_count']);fps=float(meta['fps']);frames=np.arange(n,dtype=np.int64);times=frames/max(fps,1e-9);spans=parse(resp[vid]);inside=np.asarray([any(x-1e-6<=t<=y+1e-6 for x,y in spans) for t in times],float);centre=1-np.abs(times/max(float(times[-1]),1e-9)-.5)*2;base=rank(centre)+.20*(rank(inside)-.5);k=int(round(KEEP*n));base_set=set(frames[np.argsort(-base,kind='mergesort')[:k]].tolist());old_set={int(x['frame']) for x in old[vid]['predictions']}
  if base_set!=old_set:raise ValueError(f'R54 mismatch {vid}')
  d=json.loads((a.utility_dir/f'{vid}.json').read_text());t=d['ratios']['t'];kfs=np.asarray(d['keyframes'],float);us=t['utility'];by={int(x['frame']):x for x in parent['predictions']};W,H=int(meta['width']),int(meta['height']);cw,ch,axis=geometry(W,H,meta['targetRatioWH']);ext=0 if axis is None else ((W-cw) if axis==0 else (H-ch));key=[];positions=[]
  for f,u in zip(kfs,us):
   scores=np.asarray(u['scores'],np.float32)
   if len(scores)!=NC:raise ValueError(f'{vid}: missing scores')
   b=by[int(round(float(f)))]['bboxes'];off=0 if axis is None else float(b[0] if axis==0 else b[1]);pos=.5 if ext<=0 else float(np.clip(round(np.clip(off/ext,0,1)*(NC-1))/(NC-1),0,1));key.append(np.r_[scores,pos]);positions.append(pos)
  X=torch.from_numpy(np.asarray(key,np.float32));
  with torch.inference_mode(): kval=m((X-mu)/sd).numpy()
  utility=np.interp(np.arange(n,dtype=float),kfs,kval);score=rank(base)*(0.5+0.5*midrank(utility));chosen=np.sort(frames[np.argsort(-score,kind='mergesort')[:k]]);byf={int(x['frame']):x for x in parent['predictions']};q=dict(parent);q['predictions']=[byf[int(f)] for f in chosen];q['model_size_mb']=(ETA_PARAMS+V9_PARAMS+R70_PARAMS+DIRECT_PARAMS)/1e6;q['model_params_b']=(ETA_PARAMS+V9_PARAMS+R70_PARAMS+DIRECT_PARAMS)/1e9;out.append(q);ns=set(int(f) for f in chosen);ss=len(base_set^ns)//2;swaps+=ss;total+=k;all_util.extend(kval.tolist());details.append({'video_id':vid,'n_full':n,'n_keep':k,'swaps_vs_r54':ss,'overlap_vs_r54':len(base_set&ns)/max(k,1),'keyframes':len(kfs),'utility_mean':float(np.mean(kval)),'utility_std':float(np.std(kval)),'output_position_mean':float(np.mean(positions)),'selected_utility_rank':float(midrank(utility)[np.searchsorted(frames,chosen)].mean())})
 a.output.mkdir(parents=True,exist_ok=False);pred=a.output/'predictions.jsonl';pred.write_text(''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in out));zp=a.output/f'{a.output.name}.zip'
 with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z:z.write(pred,'predictions.jsonl')
 with tempfile.TemporaryDirectory() as td:
  with zipfile.ZipFile(zp) as z:z.extractall(td)
  if sha(Path(td)/'predictions.jsonl')!=sha(pred):raise RuntimeError('roundtrip')
 total_params=ETA_PARAMS+V9_PARAMS+R70_PARAMS+DIRECT_PARAMS
 man={'submission_id':a.output.name,'status':'PACKAGED_NOT_UPLOADED','official_platform_score':None,'parent_submission':'R54_TIMELENS_CENTER_RESIDUAL_V9_B3','mechanism':'direct deployed-box native-IoU utility head: R70 topsoft 129 logits plus audited V9 B3 output position, fused multiplicatively with R54 temporal value','utility_definition':'source-disjoint public head predicts native IoU of deployed V9 B3 crop; official keyframe input is R70 topsoft scores plus V9 parent output offset','fusion':'rank(R54 temporal value) * (0.5 + 0.5*midrank(direct_output_utility))','spatial_mode':'v9_b3_artifact','parameter_components':[{'name':'TimeLens2-8B','parameters':ETA_PARAMS},{'name':'V9 vision-only SigLIP2+B3+YuNet','parameters':V9_PARAMS},{'name':'R70 topsoft utility head (generates official logits)','parameters':R70_PARAMS},{'name':'direct output utility head','parameters':DIRECT_PARAMS}],'total_parameters':total_params,'model_params_b':total_params/1e9,'video_count':len(out),'prediction_count':total,'temporal_swaps_vs_r54':swaps,'prediction_sha256':sha(pred),'zip_sha256':sha(zp),'zip_bytes':zp.stat().st_size,'official_test_labels_used':False,'direct_head_sha256':sha(a.head),'per_video':details}
 (a.output/'manifest.json').write_text(json.dumps(man,ensure_ascii=False,indent=2)+'\n');(a.output/(zp.name+'.sha256')).write_text(sha(zp)+'  '+zp.name+'\n');print(json.dumps({k:v for k,v in man.items() if k!='per_video'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
