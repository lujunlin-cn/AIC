#!/usr/bin/env python3
"""Build tie-aware spatial-disagreement temporal candidate.

R63 used stable ordinal ranks.  Since most V9/R26 geometry deltas are exactly
zero, that implementation assigned different ranks to equal zero values in
frame order, silently adding a position prior.  This builder uses average
ranks for ties, so equal geometry deltas receive equal utility.
"""
from __future__ import annotations
import argparse, hashlib, json, math, re, tempfile, zipfile
from pathlib import Path
import numpy as np

TIME_LENS_PARAMS=8_767_123_696
V9_PARAMS=87_407_729
ETA=.20
KEEP=.85
D_EPS=1e-6
TIME_RE=re.compile(r"\[\s*\[.*?\]\s*\]")

def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def read(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def parse(text):
 m=TIME_RE.search(text or "")
 if not m:return []
 try:a=json.loads(m.group(0))
 except Exception:return []
 out=[]
 for x in a if isinstance(a,list) else []:
  try:a0,b0=float(x[0]),float(x[1])
  except Exception:continue
  if math.isfinite(a0) and math.isfinite(b0) and b0>=a0:out.append((a0,b0))
 return out
def rank_tieavg(v):
 v=np.asarray(v,dtype=np.float64); n=len(v)
 order=np.argsort(v,kind="mergesort"); s=v[order]; out=np.empty(n,dtype=np.float64)
 i=0
 while i<n:
  j=i+1
  while j<n and s[j]==s[i]: j+=1
  out[order[i:j]]=((i+j-1)/2.0)
  i=j
 return out/max(n-1,1)
def rank_stable(v):
 # Exact R54 rank implementation.  This is intentionally stable for the
 # temporal center and interval masks so gamma=0 is an exact reproduction.
 v=np.asarray(v,dtype=np.float64)
 return np.argsort(np.argsort(v,kind="mergesort"),kind="mergesort").astype(float)/max(len(v)-1,1)
def disagreement(v9,r26,width,height):
 old={int(x["frame"]):x["bboxes"] for x in r26}; scale=max(float(width),float(height),1.0)
 out=[]
 for x in v9:
  a=x["bboxes"]; b=old[int(x["frame"])]
  out.append(sum(abs(float(u)-float(v)) for u,v in zip(a,b))/scale)
 return np.asarray(out,float)
def main():
 ap=argparse.ArgumentParser()
 for k in ("parent","timelens","index","v9_full","r26_full","output"): ap.add_argument("--"+k,type=Path,required=True)
 ap.add_argument("--gamma",type=float,default=1.0)
 args=ap.parse_args()
 parent={str(x["video_id"]):x for x in read(args.parent)}
 idx={str(x["video_id"]):x for x in read(args.index)}
 v9={str(x["video_id"]):x for x in read(args.v9_full)}
 old={str(x["video_id"]):x for x in read(args.r26_full)}
 tl=json.loads(args.timelens.read_text()); resp={str(x["video_id"]):str(x.get("response","")) for x in tl.get("results",[])}
 rows=[]; details=[]; swaps=0
 for vid in sorted(parent,key=lambda x:int(x) if x.isdigit() else x):
  m=idx[vid]; n=int(m["frame_count"]); fps=float(m["fps"]); frames=v9[vid]["predictions"]; oldf=old[vid]["predictions"]
  t=np.arange(n,dtype=float)/fps; ints=parse(resp.get(vid,""))
  inside=np.asarray([any(a-1e-6<=z<=b+1e-6 for a,b in ints) for z in t],float)
  center=1-np.abs(t/max(t[-1],1e-9)-.5)*2
  # Ignore serializer/interpolation noise below the R53 geometric audit
  # threshold.  IoU >= .999999 is not a meaningful spatial intervention.
  d=disagreement(frames,oldf,m.get("width",1),m.get("height",1)); d=np.where(d<=D_EPS,0.0,d)
  gamma=float(args.gamma)
  score=rank_stable(center)+ETA*(rank_stable(inside)-.5)+gamma*(rank_tieavg(d)-.5)
  k=int(round(KEEP*n)); chosen=np.sort(np.argsort(-score,kind="mergesort")[:k]); by={int(x["frame"]):x for x in frames}
  q=dict(v9[vid]); q["predictions"]=[dict(by[int(f)]) for f in chosen]; q["model_size_mb"]=(TIME_LENS_PARAMS+V9_PARAMS)/1e6; q["model_params_b"]=(TIME_LENS_PARAMS+V9_PARAMS)/1e9; rows.append(q)
  pset={int(x["frame"]) for x in parent[vid]["predictions"]}; swaps+=len(pset^set(map(int,chosen)))//2
  details.append({"video_id":vid,"n_full":n,"n_keep":k,"swaps_vs_r54":len(pset^set(map(int,chosen)))//2,"position_mean":float(np.mean(chosen/max(n-1,1))),"delta_zero_fraction":float(np.mean(d==0))})
 args.output.mkdir(parents=True,exist_ok=False); pred=args.output/"predictions.jsonl"
 pred.write_text("".join(json.dumps(x,ensure_ascii=False,separators=(",",":"))+"\n" for x in rows))
 zp=args.output/(args.output.name+".zip")
 with zipfile.ZipFile(zp,"w",zipfile.ZIP_DEFLATED) as z:z.write(pred,"predictions.jsonl")
 with tempfile.TemporaryDirectory() as td:
  with zipfile.ZipFile(zp) as z:z.extractall(td)
  if sha(Path(td)/"predictions.jsonl")!=sha(pred): raise RuntimeError("roundtrip")
 m={"submission_id":args.output.name,"status":"PACKAGED_NOT_UPLOADED","uploaded":False,"official_platform_score":None,
 "parent_submission":"R54_TIMELENS_CENTER_RESIDUAL_V9_B3","mechanism":"R54 exact stable center+interval utility plus tie-aware average-rank V9/R26 geometry disagreement; equal deltas receive equal rank","utility":f"rank_stable(center)+0.20*rank_stable(TimeLens2 interval)+{float(args.gamma):g}*rank_tieavg(V9_R26_geometry_disagreement)-0.5","gamma":float(args.gamma),"geometry_delta_epsilon":D_EPS,"rank_tie_policy":"average ranks for exact geometry ties; exact R54 stable rank for center and interval","video_count":len(rows),"prediction_count":sum(len(x["predictions"]) for x in rows),"total_parameters":TIME_LENS_PARAMS+V9_PARAMS,"model_params_b":(TIME_LENS_PARAMS+V9_PARAMS)/1e9,"prediction_sha256":sha(pred),"zip_sha256":sha(zp),"zip_bytes":zp.stat().st_size,"r54_swaps":swaps,"mean_position":float(np.mean([x["position_mean"] for x in details])),"mean_zero_delta_fraction":float(np.mean([x["delta_zero_fraction"] for x in details]))}
 (args.output/"manifest.json").write_text(json.dumps(m,ensure_ascii=False,indent=2)+"\n"); (args.output/(zp.name+".sha256")).write_text(sha(zp)+"  "+zp.name+"\n")
 print(json.dumps(m,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
