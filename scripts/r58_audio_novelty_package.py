#!/usr/bin/env python3
"""Build the aggressive TimeLens + audio novelty candidate.

The base set is R54 (TimeLens2 temporal intervals + V9 B3 boxes).  We perform
one predeclared intervention: within each video, exchange five percent of the
lowest-audio-novelty selected frames for the highest-novelty unselected frames.
The score is an offline, parameter-free audio onset/novelty descriptor.  B3
boxes for newly admitted frames come from the audited all-frame R26 artifact;
existing R54 frames retain their V9 boxes.  This is intentional and recorded
in the manifest so the official result is interpreted as a temporal/audio
experiment with a small spatial-artifact confound.
"""
from __future__ import annotations
import argparse, hashlib, json, math, tempfile, zipfile
from pathlib import Path
import numpy as np

PARAMS=8_854_531_425
SWAP_FRAC=.05

def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def rows(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def key(r):
 s=str(r['video_id']); return int(s) if s.isdigit() else s
def audio_at(rec, t):
 if not rec.get('present'): return 0.0
 ts=np.asarray(rec['times'],np.float64); z=np.asarray(rec['novelty'],np.float64)
 if not len(ts): return 0.0
 return float(np.interp(float(t),ts,z,left=z[0],right=z[-1]))

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--base',type=Path,required=True); ap.add_argument('--allframes',type=Path,required=True); ap.add_argument('--index',type=Path,required=True); ap.add_argument('--audio',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
 base={str(x['video_id']):x for x in rows(args.base)}; full={str(x['video_id']):x for x in rows(args.allframes)}; idx={str(x['video_id']):x for x in rows(args.index)}; aud=json.loads(args.audio.read_text()); out=[]; details=[]; total=0
 if len(base)!=426 or len(full)!=426: raise ValueError('coverage')
 for vid in sorted(base,key=lambda v:int(v) if v.isdigit() else v):
  q=base[vid]; f=full[vid]; meta=idx[vid]; fps=float(meta['fps']); n=int(meta['frame_count']); allp={int(p['frame']):p for p in f['predictions']}; old={int(p['frame']):p for p in q['predictions']}; chosen=set(old); k=int(round(.85*n));
  if len(allp)!=n or len(chosen)!=k: raise ValueError(f'{vid} coverage {len(allp)} {len(chosen)} {k}')
  # Use a ±0.25 s neighborhood maximum to avoid a single audio sample phase
  # deciding a frame; this still rewards a genuine transient.
  rec=aud.get(vid,{})
  score={fr:max(audio_at(rec,(fr+d)/fps) for d in range(-max(1,int(round(.25*fps))),max(1,int(round(.25*fps)))+1)) for fr in range(n)}
  inside=sorted(chosen,key=lambda fr:(score[fr],fr))
  outside=sorted(set(range(n))-chosen,key=lambda fr:(score[fr],-fr),reverse=True)
  m=min(int(round(SWAP_FRAC*k)),len(inside),len(outside)); drop=set(inside[:m]); add=[]
  # Avoid adding audio frames whose score is exactly baseline/zero: if there
  # is no present audio, this becomes a no-op with a fully auditable fallback.
  for fr in outside:
   if len(add)>=m: break
   if score[fr] > score[inside[m-1]] or (score[fr]>0 and score[fr]>=np.percentile(list(score.values()),90)):
    add.append(fr)
  # Fill the declared swap count deterministically when audio exists; absent
  # audio leaves the TimeLens set untouched.
  if rec.get('present') and len(add)<m and max(score.values(), default=0.0)>1e-7: add=outside[:m]
  add=set(add);
  # If audio is absent or flat, no replacement is made; keep drop count
  # equal to the actual admission count rather than the declared budget.
  if len(add)<len(drop): drop=set(inside[:len(add)])
  final=sorted((chosen-drop)|add)
  if len(final)!=k: raise ValueError(f'{vid}: final {len(final)} != {k}')
  preds=[]; changed=0; old_box=0
  for fr in final:
   if fr in old: p=dict(old[fr])
   else: p=dict(allp[fr]); changed+=1; old_box+=1
   preds.append(p)
  z=dict(q); z['predictions']=preds; z['model_size_mb']=PARAMS/1e6; z['model_params_b']=PARAMS/1e9; out.append(z); total+=len(preds)
  details.append({'video_id':vid,'n_full':n,'n_keep':k,'audio_present':bool(rec.get('present')),'swap_budget':m,'changed_frames':changed,'drop_count':len(drop),'add_count':len(add),'score_selected_mean':float(np.mean([score[x] for x in chosen])),'score_added_mean':float(np.mean([score[x] for x in add])) if add else 0.0})
 args.output.mkdir(parents=False,exist_ok=False); pred=args.output/'predictions.jsonl'; pred.write_text(''.join(json.dumps(z,ensure_ascii=False,separators=(',',':'))+'\n' for z in out)); zp=args.output/(args.output.name+'.zip')
 with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z: z.write(pred,'predictions.jsonl')
 with tempfile.TemporaryDirectory() as td:
  with zipfile.ZipFile(zp) as z:
   if z.namelist()!=['predictions.jsonl']: raise ValueError(z.namelist())
   z.extractall(td)
  if sha(Path(td)/'predictions.jsonl')!=sha(pred): raise ValueError('roundtrip')
 m={'submission_id':args.output.name,'status':'PACKAGED_NOT_UPLOADED','official_platform_score':None,'parent_submission':'R54_TIMELENS_CENTER_RESIDUAL_V9_B3','mechanism':'TimeLens temporal set plus one predeclared 5% per-video swap using parameter-free dense audio onset/novelty','swap_fraction':SWAP_FRAC,'audio_features':'20Hz RMS/peak/ZCR/FFT-centroid, positive log-energy novelty, local contrast, ±0.25s max','spatial_mode':'hybrid: R54 V9 boxes retained for existing frames; R26 audited B3 all-frame boxes for added frames','eligibility':'PARAMETER_COMPLIANT_IF_V9_ARTIFACT_ACCEPTED','parameter_components':[{'name':'TimeLens2-8B','parameters':8767123696},{'name':'V9 vision-only SigLIP2 + B3 head + YuNet','parameters':87407729},{'name':'audio DSP','parameters':0}],'total_parameters':PARAMS,'model_params_b':PARAMS/1e9,'video_count':len(out),'prediction_count':total,'prediction_sha256':sha(pred),'zip_sha256':sha(zp),'zip_bytes':zp.stat().st_size,'base_sha256':sha(args.base),'allframes_sha256':sha(args.allframes),'audio_sha256':sha(args.audio),'index_sha256':sha(args.index),'per_video':details}
 (args.output/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n'); (args.output/(zp.name+'.sha256')).write_text(sha(zp)+'  '+zp.name+'\n'); print(json.dumps({k:v for k,v in m.items() if k!='per_video'},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
