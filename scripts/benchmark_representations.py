#!/usr/bin/env python3
"""Frozen-checkpoint multi-metric re-evaluation; no threshold search."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from aic.features import FeatureCacheDataset
from aic.models import TemporalUNet, temporal_shift
from aic.temporal_metrics import (ranking_report,human_summaries,summary_report,
                                 RANKING_PROTOCOL,SUMMARY_PROTOCOL,AUTHOR_REVISION)
from aic.train import video_proxy_metrics

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")
def read_tvsum(mat):
    import h5py
    with h5py.File(mat) as f:
        g=f["tvsum50"]; records={}
        for i in range(g["video"].size):
            def field(name): return np.asarray(f[g[name][()].ravel()[i]])
            def text(name): return "".join(chr(int(c)) for c in field(name).ravel())
            vid=text("video"); a=field("user_anno")
            if a.shape[0]!=20: a=a.T
            n=int(field("nframes").item())
            if a.shape!=(20,n):raise ValueError(f"{vid} bad annotation shape")
            records[vid]={"video_id":vid,"category":text("category"),"title":text("title"),
                          "nframes":n,"user_anno":a,"continuous":(a.mean(axis=0)-1)/4}
    return records
def prepare_gt(mat,outdir):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    records=read_tvsum(mat);metadata=[]
    for vid,r in records.items():
        p=out/(vid+".npz")
        np.savez_compressed(p,human_masks=human_summaries(r["user_anno"]),continuous=r["continuous"])
        metadata.append({k:v for k,v in r.items() if k not in ("user_anno","continuous")})
    write(out/"metadata.json",{"mat_sha256":digest(mat),"protocol":SUMMARY_PROTOCOL,"videos":metadata})
    return metadata
def load_head(checkpoint,device):
    state=torch.load(checkpoint,map_location="cpu",weights_only=True)
    weights=state.get("model",state.get("state_dict"))
    if weights is None: raise ValueError("checkpoint lacks model state")
    full=any(k.startswith("temporal.") for k in weights)
    if full:weights={k.removeprefix("temporal."):v for k,v in weights.items() if k.startswith("temporal.")}
    dim=int(weights["adapter.weight"].shape[1])
    aux=int(weights["aux_adapter.0.weight"].shape[1]) if "aux_adapter.0.weight" in weights else 0
    model=TemporalUNet(dim,aux).to(device)
    model.load_state_dict(weights,strict=True);model.eval()
    shift=bool(state.get("config",{}).get("temporal_shift",False))
    return model,shift,state
def evaluate(job,gtroot,outroot,device):
    start=time.perf_counter();torch.set_num_threads(4)
    model,shift,state=load_head(job["checkpoint"],device)
    ds=FeatureCacheDataset(job["manifest"]);meta={x["video_id"]:x for x in json.loads((Path(gtroot)/"metadata.json").read_text())["videos"]}
    out=Path(outroot)/job["run_id"];out.mkdir(parents=True,exist_ok=False)
    rows=[];latencies=[]
    with torch.inference_mode():
        for item in ds:
            vid=item["video_id"]; x=torch.from_numpy(item["features"]).unsqueeze(0).to(device)
            if x.shape[-1]!=model.input_dim:raise ValueError("encoder dimension mismatch")
            y=torch.from_numpy(item["labels"]);valid=np.asarray(item["mask"],bool)
            if shift:x=temporal_shift(x)
            aux=torch.from_numpy(item["aux"]).unsqueeze(0).to(device) if model.aux_dim else None
            if str(device).startswith("cuda"):torch.cuda.synchronize()
            t=time.perf_counter()
            logits=model(x,aux=aux,lengths=torch.tensor([x.shape[1]],device=device))[0].float().cpu()
            if str(device).startswith("cuda"):torch.cuda.synchronize()
            latencies.append(time.perf_counter()-t);p=logits.sigmoid().numpy()
            with np.load(Path(gtroot)/(vid+".npz")) as gt:
                indices=item["frame_indices"]
                if np.any(np.diff(indices)<=0):raise ValueError("frame indices not strictly increasing")
                if np.max(np.abs(y.numpy()[valid]-gt["continuous"][indices][valid]))>1e-5:
                    raise ValueError("cached labels disagree with original human scores")
                fullp=np.interp(np.arange(meta[vid]["nframes"]),indices,p)
                native=summary_report(fullp,gt["human_masks"])
            row={**meta[vid],**video_proxy_metrics(logits,y,torch.from_numpy(valid),job["threshold"],vid),
                 **ranking_report(p[valid],y.numpy()[valid]),**native,"threshold":job["threshold"]}
            rows.append(row)
            np.savez_compressed(out/(vid+".npz"),probabilities=p,labels=y.numpy(),mask=valid,
                                frame_indices=indices,timestamps=item["timestamps"])
    numeric=["f1","spearman","kendall_tau_b","ndcg","ndcg_at_15pct","ap_fixed_gt_0p5",
             "top15_mean_relevance","summary_f1","prediction_rate","empty_prediction"]
    stats={}
    for k in numeric:
        values=[r[k] for r in rows if r[k] is not None]
        stats[k]={"mean":float(np.mean(values)),"std":float(np.std(values,ddof=1)) if len(values)>1 else 0.,
                  "median":float(np.median(values)),"n_defined":len(values)} if values else None
    result={"job":job,"ranking_protocol":RANKING_PROTOCOL,"summary_protocol":SUMMARY_PROTOCOL,
            "author_revision":AUTHOR_REVISION,"summary_interpolation":"linear_original_frame_indices_endpoint_hold_v1",
            "elapsed_seconds":time.perf_counter()-start,"head_seconds":sum(latencies),
            "checkpoint_sha256":digest(job["checkpoint"]),"manifest_sha256":digest(job["manifest"]),
            "checkpoint_bytes":Path(job["checkpoint"]).stat().st_size,
            "official_f_video":None,"competition_score":None,"statistics":stats,"per_video":rows}
    write(out/"metrics.json",result)
    print(json.dumps({"run_id":job["run_id"],"metrics":{k:v["mean"] if v else None for k,v in stats.items()}}),flush=True)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--jobs");ap.add_argument("--mat");ap.add_argument("--gt-root",required=True)
    ap.add_argument("--output");ap.add_argument("--device",default="cpu");a=ap.parse_args()
    if a.mat:prepare_gt(a.mat,a.gt_root)
    if a.jobs:
        jobs=json.loads(Path(a.jobs).read_text())
        for job in jobs:evaluate(job,a.gt_root,a.output,torch.device(a.device))
if __name__=="__main__":main()
