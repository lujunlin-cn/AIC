#!/usr/bin/env python3
"""Zero-shot raw SumMe evaluation: frozen TVSum models, no OOD tuning."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np,torch
from scipy.io import loadmat
from aic.models import load_inference_model
from aic.features import _letterbox
from aic.video import probe_video,iter_sampled_frames
from aic.temporal_metrics import ranking_report,summary_mask
from aic.contract import VideoMetadata,center_crop,write_submission

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);ap.add_argument("--model",required=True)
    a=ap.parse_args();cfg=json.loads(Path(a.config).read_text());job=cfg["models"][a.model]
    root=Path(cfg["root"]);out=Path(cfg["output"])/a.model;out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);model,meta=load_inference_model(job["bundle"],"cuda:0")
    rows=[];audit=[];sub=[];index={};start=time.perf_counter()
    for p in sorted((root/"videos").glob("*.mp4")):
        info=probe_video(p);gt=loadmat(root/"GT"/(p.stem+".mat"));n=int(gt["nFrames"].item())
        record={"video_id":p.stem,"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),
            **info.to_dict(),"gt_nframes":n,"gt_sha256":hashlib.sha256((root/"GT"/(p.stem+".mat")).read_bytes()).hexdigest(),
            "included":info.frame_count==n,"split":"ood_evaluation_only","dataset":"SumMe","version":"zenodo4884870_subset8_v1"}
        audit.append(record)
        if info.frame_count!=n:continue
        t=time.perf_counter();sampled=list(iter_sampled_frames(p,sample_fps=2,size=224))
        with torch.inference_mode():
            feats=[]
            for j in range(0,len(sampled),16):
                x=_letterbox(np.stack([s[2] for s in sampled[j:j+16]])).to("cuda:0")
                feats.append(model.encode_frames(x).float())
            features=torch.cat(feats)
            scores=model(features[None])[0].sigmoid().float().cpu().numpy()
        indices=np.array([s[0] for s in sampled])
        full=np.interp(np.arange(n),indices,scores)
        human=np.asarray(gt["user_score"])>0
        relevance=np.asarray(gt["gt_score"]).ravel()
        # A deterministic 15% mask, evaluated against native human summaries.
        span=np.ptp(full);selected=summary_mask((full-full.min())/span if span else np.zeros(n))
        intersection=(human & selected[:,None]).sum(axis=0)
        den=human.sum(axis=0)+selected.sum()
        f1=np.divide(2*intersection,den,out=np.zeros(human.shape[1]),where=den>0)
        pred=full>=job["threshold"]
        row={"video_id":p.stem,"native_mean_user_f1":float(f1.mean()),"native_max_user_f1":float(f1.max()),
             **ranking_report(full,relevance),"summary_rate":float(selected.mean()),
             "threshold":job["threshold"],"raw_threshold_rate":float(pred.mean()),
             "elapsed_seconds":time.perf_counter()-t}
        rows.append(row)
        np.savez_compressed(out/(p.stem+".npz"),probabilities=scores,frame_indices=indices,
                            frame_scores=full,relevance=relevance,summary_mask=selected)
        crop=center_crop(info.width,info.height,[9,16])
        index[p.stem]=VideoMetadata(p.stem,info.width,info.height,n,(9,16))
        sub.append({"video_id":p.stem,"targetRatioWH":[9,16],"model_size_mb":meta["loaded_bytes"]/1e6,
                    "predictions":[{"frame":int(i),"bboxes":crop} for i in np.flatnonzero(pred)]})
        print(json.dumps(row),flush=True)
    val=write_submission(out/"submission.jsonl",sub,index,stage="final",actual_model_size_mb=meta["loaded_bytes"]/1e6)
    keys=["native_mean_user_f1","native_max_user_f1","spearman","kendall_tau_b","ndcg","ndcg_at_15pct"]
    report={"protocol":"SUMME_RAW_ZERO_SHOT_SUBSET8_V1","model":a.model,"config":cfg,"weights":meta["loaded_bytes"],
            "bundle_sha256":meta["loaded_sha256"],"parameter_count":sum(p.numel() for p in model.parameters()),
            "strict_video_count":len(rows),"per_video":rows,"manifest":audit,
            "means":{k:float(np.mean([r[k] for r in rows if r[k] is not None])) for k in keys},
            "latency_total_seconds":time.perf_counter()-start,
            "peak_vram_bytes":torch.cuda.max_memory_allocated(),"validator":val.to_dict(),
            "official_f_video":None,"competition_score":None}
    (out/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report["means"]),flush=True)
if __name__=="__main__":main()
