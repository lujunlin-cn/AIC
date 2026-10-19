#!/usr/bin/env python3
"""Convert fixed TimeLens2 intervals to exact-budget AIC packages.

The converter has two explicit spatial modes.  ``b3_artifact`` copies the
already-produced B3 crop as a static prediction artifact and therefore needs a
separate eligibility review; ``center_geometry`` loads no spatial network and
is parameter-compliant by construction but is expected to lose crop quality.
"""
from __future__ import annotations
import argparse, hashlib, json, math, re, tempfile, zipfile
from pathlib import Path
import numpy as np
import cv2

TIME_RE = re.compile(r"\[\s*\[.*?\]\s*\]")
TIME_LENS_PARAMS = 8_767_123_696
V9_VISIONONLY_PARAMS = 87_407_729

def sha(p):
    h=hashlib.sha256();
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def parse(s):
    m=TIME_RE.search(s or "")
    if not m:return []
    try:
        a=json.loads(m.group(0)); return [(float(x[0]),float(x[1])) for x in a if len(x)>=2 and math.isfinite(float(x[0])) and math.isfinite(float(x[1]))]
    except Exception:return []
def center_box(meta):
    W,H=float(meta["width"]),float(meta["height"]); rw,rh=map(float,meta["targetRatioWH"])
    if W/H >= rw/rh:
        w=H*rw/rh; return [float((W-w)/2),0.0,float(w)]
    w=W; h=W*rh/rw; return [0.0,float((H-h)/2),float(w)]
def read_jsonl(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--spans",type=Path,required=True); ap.add_argument("--parent",type=Path,required=True); ap.add_argument("--index",type=Path,required=True); ap.add_argument("--video-root",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--mode",choices=["b3_artifact","v9_b3_artifact","center_geometry"],required=True); args=ap.parse_args()
    spans=json.loads(args.spans.read_text()); byspan={str(x["video_id"]):parse(x.get("response","")) for x in spans.get("results",[])}
    idx={str(x["video_id"]):x for x in read_jsonl(args.index)}; parents=read_jsonl(args.parent); out=[]; details=[]; total=0
    for row in sorted(parents,key=lambda x:int(x["video_id"]) if str(x["video_id"]).isdigit() else str(x["video_id"])):
        vid=str(row["video_id"]); meta=idx[vid]; allp=row["predictions"]; n=int(meta["frame_count"]); assert len(allp)==n,(vid,len(allp),n)
        fps=float(meta.get("fps", 0.0))
        if fps <= 0:
            cap = cv2.VideoCapture(str(args.video_root / f"{vid}.mp4"))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            cap.release()
        if fps <= 0:
            raise ValueError(f"{vid}: missing positive fps in index and video")
        intervals=byspan.get(vid,[]); frames=np.asarray([int(x["frame"]) for x in allp],dtype=int); t=frames/max(fps,1e-9)
        inside=np.asarray([any(a-1e-6 <= z <= b+1e-6 for a,b in intervals) for z in t],dtype=float)
        center=1.0-np.abs(t/max(float(t[-1]),1e-9)-.5)*2.0
        score=inside*2.0+center*1e-6
        k=int(round(.85*n)); order=np.lexsort((frames,-score)); chosen=np.sort(frames[order[:k]])
        source={int(x["frame"]):x for x in allp}; cb=center_box(meta); preds=[]
        for f in chosen:
            p=dict(source[int(f)])
            if args.mode=="center_geometry": p["bboxes"]=cb
            preds.append(p)
        q=dict(row); q["predictions"]=preds
        total_params = TIME_LENS_PARAMS + (V9_VISIONONLY_PARAMS if args.mode == "v9_b3_artifact" else 0)
        q["model_size_mb"] = total_params / 1e6
        q["model_params_b"] = total_params / 1e9
        out.append(q); total+=len(preds)
        details.append({"video_id":vid,"n_full":n,"n_keep":k,"intervals":intervals,"selected_in_spans":int(sum(any(a-1e-6 <= f/max(fps,1e-9) <= b+1e-6 for a,b in intervals) for f in chosen))})
    args.output.mkdir(parents=True,exist_ok=False); pred=args.output/"predictions.jsonl"
    with pred.open("w") as f:
        for q in out:f.write(json.dumps(q,ensure_ascii=False,separators=(",",":"))+"\n")
    zip_path=args.output/(args.output.name+".zip")
    with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED) as z:z.write(pred,"predictions.jsonl")
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zip_path) as z:z.extractall(td)
        assert sha(Path(td)/"predictions.jsonl")==sha(pred)
    total_params = TIME_LENS_PARAMS + (V9_VISIONONLY_PARAMS if args.mode == "v9_b3_artifact" else 0)
    if args.mode == "center_geometry":
        b3_note = "deterministic legal center crop; no spatial neural model"
        eligibility = "PARAMETER_COMPLIANT"
        components = [{"name":"TimeLens2-8B","parameters":TIME_LENS_PARAMS}]
    elif args.mode == "v9_b3_artifact":
        b3_note = ("B3 boxes inherited from the audited V9 vision-only spatial artifact; "
                   "this package does not rerun the V9 spatial forward pass. "
                   "Eligibility requires accepting the prior artifact as the deployed "
                   "85.843M SigLIP2 + 1.511M head + 53,104 YuNet component.")
        eligibility = "PARAMETER_COMPLIANT_IF_V9_ARTIFACT_ACCEPTED"
        components = [{"name":"TimeLens2-8B","parameters":TIME_LENS_PARAMS},
                      {"name":"V9 vision-only SigLIP2 + B3 head + YuNet","parameters":V9_VISIONONLY_PARAMS}]
    else:
        b3_note = "static B3 bbox artifact copied from prior R26 output; no B3 model loaded in this package generation; eligibility review required"
        eligibility = "REVIEW_REQUIRED"
        components = [{"name":"TimeLens2-8B","parameters":TIME_LENS_PARAMS},
                      {"name":"unverified static B3 artifact","parameters":0}]
    manifest={"submission_id":args.output.name,"status":"PACKAGED_NOT_UPLOADED","official_platform_score":None,"parent_submission":"R26_POS_CENTER_EXACT085_B3","mechanism":"TimeLens2-8B fixed generic highlight intervals converted to exact 85 percent frame budget","spatial_mode":args.mode,"spatial_note":b3_note,"eligibility":eligibility,"parameter_components":components,"time_lens_parameters":TIME_LENS_PARAMS,"total_parameters":total_params,"model_params_b":total_params/1e9,"video_count":len(out),"prediction_count":total,"prediction_sha256":sha(pred),"zip_sha256":sha(zip_path),"zip_bytes":zip_path.stat().st_size,"per_video":details}
    (args.output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n"); (args.output/(zip_path.name+".sha256")).write_text(sha(zip_path)+"  "+zip_path.name+"\n")
    print(json.dumps({k:v for k,v in manifest.items() if k!="per_video"},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
