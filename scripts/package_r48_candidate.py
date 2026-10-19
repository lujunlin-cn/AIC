#!/usr/bin/env python3
"""Package R48 and its fixed 5% edge guarded derivative."""
from __future__ import annotations
import argparse, hashlib, json, math, tempfile, zipfile
from pathlib import Path
P=450283377; KEEP=.85; BAND=.05
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def read(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def boundary(base,cand,n):
 b={int(x['frame']) for x in base}; c={int(x['frame']) for x in cand}; add=sorted(c-b); rem=sorted(b-c); lo,hi=min(b),max(b); band=max(1,int(math.ceil(BAND*n))); aa=[x for x in add if lo-band<=x<lo or hi<x<=hi+band]; rr=[x for x in rem if lo<=x<lo+band or hi-band<x<=hi]; q=min(len(aa),len(rr)); dist=lambda x:(min(abs(x-lo),abs(x-hi)),x); return sorted((b-set(sorted(rr,key=dist)[:q]))|set(sorted(aa,key=dist)[:q])),q
def pack(rows,out,name,mech):
 out.mkdir(parents=True,exist_ok=False); pred=out/'predictions.jsonl'
 with pred.open('w') as f:
  for x in rows:
   r=dict(x); r['model_params_b']=P/1e9; r['model_size_mb']=P/1e6; f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
 z=out/f'{name}.zip'
 with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as q:q.write(pred,'predictions.jsonl')
 with tempfile.TemporaryDirectory() as t:
  with zipfile.ZipFile(z) as q:q.extractall(t)
  assert sha(Path(t)/'predictions.jsonl')==sha(pred)
 m={'submission_id':name,'status':'PACKAGED_NOT_UPLOADED','uploaded':False,'official_platform_score':None,'parent_submission':'R26_POS_CENTER_EXACT085_B3','mechanism':mech,'changed_factor':'temporal frame ranking only; R26 B3 spatial boxes copied','keep_rule':'round(0.85 * full original frame count)','video_count':len(rows),'prediction_count':sum(len(x['predictions']) for x in rows),'total_parameters':P,'model_params_b':P/1e9,'model_size_mb':P/1e6,'predictions_sha256':sha(pred),'zip_sha256':sha(z),'zip_bytes':z.stat().st_size,'spatial_boxes_source':'R26_POS_CENTER_EXACT085_B3'}
 (out/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n'); (out/(z.name+'.sha256')).write_text(sha(z)+'  '+z.name+'\n'); return m
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--r26',type=Path,required=True); ap.add_argument('--candidate',type=Path,required=True); ap.add_argument('--nfull',type=Path,required=True); ap.add_argument('--output-root',type=Path,required=True); a=ap.parse_args(); b=read(a.r26); c=read(a.candidate); nf={str(x['video_id']):int(x['n_full']) for x in json.loads(a.nfull.read_text())['per_video']}; bm={str(x['video_id']):x for x in b}; cm={str(x['video_id']):x for x in c}; assert set(bm)==set(cm) and len(bm)==426; full=[]; edge=[]; swaps=0
 for v in sorted(bm,key=lambda x:int(x)):
  br,cr=bm[v],cm[v]; assert len(br['predictions'])==len(cr['predictions'])==round(KEEP*nf[v]); bp={int(x['frame']):x for x in br['predictions']}; cp={int(x['frame']):x for x in cr['predictions']}; assert all(bp[f].get('bboxes')==cp[f].get('bboxes') for f in set(bp)&set(cp)); full.append(cr); chosen,q=boundary(list(bp.values()),list(cp.values()),nf[v]); edge.append({**dict(br),'predictions':[bp[f] if f in bp else cp[f] for f in chosen]}); swaps+=q
 root=a.output_root; root.mkdir(parents=True,exist_ok=True); m=pack(full,root/'R48_CONSENSUS_GATED_RESIDUAL_B3','R48_CONSENSUS_GATED_RESIDUAL_B3','1fps/4fps local-state consensus gate, eta=.20'); e=pack(edge,root/'R48_CONSENSUS_GATED_BOUNDARY_B3','R48_CONSENSUS_GATED_BOUNDARY_B3','same R48 rule restricted to fixed 5% R26 edge bands'); e['retained_boundary_swaps']=swaps; (root/'R48_CONSENSUS_GATED_BOUNDARY_B3'/'manifest.json').write_text(json.dumps(e,ensure_ascii=False,indent=2)+'\n'); print(json.dumps({'full':m,'boundary':e},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
