#!/usr/bin/env python3
"""Preregistered repeated nested source-group CV for cached representations."""
import argparse,copy,hashlib,json,os,random,subprocess,sys,time
from pathlib import Path
import numpy as np,torch
from aic.models import TemporalUNet
from aic.features import FeatureCacheDataset
from aic.train import video_proxy_metrics
from scripts.benchmark_representations import evaluate,write,digest

def train_one(cfg,fold,model_name,seed,out,device,records):
    out.mkdir(parents=True,exist_ok=False)
    config={**cfg,"fold":fold,"model_name":model_name,"seed":seed,"run_id":out.name}
    write(out/"config.json",config)
    (out/"command.txt").write_text(" ".join(sys.argv)+"\n")
    (out/"git_commit.txt").write_text(cfg["git_commit"]+"\n")
    (out/"environment.txt").write_text(sys.version+"\n"+str(torch.__version__)+"\n"+
        subprocess.check_output(["nvidia-smi"],text=True))
    split_sets=[set(fold[k]) for k in ("train","inner_dev","outer_test")]
    assert not any(a & b for i,a in enumerate(split_sets) for b in split_sets[i+1:])
    for k in ("train","inner_dev","outer_test"):
        p=out/(k+".jsonl");p.write_text("".join(json.dumps(records[x])+"\n" for x in fold[k]))
    data={}
    for k in ("train","inner_dev"):
        ds=FeatureCacheDataset(out/(k+".jsonl")); data[k]=[]
        for x in ds:
            data[k].append((x["video_id"],torch.from_numpy(x["features"]).float().unsqueeze(0).to(device),
                            torch.from_numpy(x["labels"]).float().to(device),
                            torch.from_numpy(x["mask"]).bool().to(device)))
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    head_type=cfg.get('head_type','temporal_unet')
    if head_type=='linear':
        from aic.linear_head import LinearScorer
        model=LinearScorer(data['train'][0][1].shape[-1]).to(device)
    elif head_type=='temporal_unet':
        model=TemporalUNet(data["train"][0][1].shape[-1]).to(device)
    else:raise ValueError('Unsupported preregistered head type')
    opt=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    best=float("inf");best_epoch=None;history=[];start=time.monotonic()
    for epoch in range(cfg["epochs"]):
        model.train(); order=np.random.permutation(len(data["train"])); tl=[]
        for i in order:
            _,x,y,m=data["train"][i];opt.zero_grad(set_to_none=True)
            z=model(x)[0];loss=torch.nn.functional.binary_cross_entropy_with_logits(z[m],y[m])
            if not torch.isfinite(loss):raise RuntimeError("non-finite loss")
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();tl.append(loss.item())
        model.eval();vl=[]
        with torch.inference_mode():
            for _,x,y,m in data["inner_dev"]:
                z=model(x)[0];vl.append(torch.nn.functional.binary_cross_entropy_with_logits(z[m],y[m]).item())
        v=float(np.mean(vl));history.append({"epoch":epoch,"train_bce":float(np.mean(tl)),"inner_dev_bce":v})
        bundle={"model":copy.deepcopy(model.cpu().state_dict()),"input_dim":model.input_dim,"seed":seed,
                "run_id":out.name,"config":config,"epoch":epoch,"optimizer":opt.state_dict()}
        torch.save(bundle,out/"last.pt");model.to(device)
        if v < best:
            best=v;best_epoch=epoch;torch.save(bundle,out/"best.pt")
        if time.monotonic()-start > cfg["max_seconds"]:raise TimeoutError("per-training budget exceeded")
    model.load_state_dict(torch.load(out/"best.pt",map_location=device,weights_only=True)["model"]);model.eval()
    threshold_scores=[]
    with torch.inference_mode():
        predictions=[(vid,model(x)[0],y,m) for vid,x,y,m in data["inner_dev"]]
    for th in cfg["thresholds"]:
        values=[video_proxy_metrics(z,y,m,th,vid)["f1"] for vid,z,y,m in predictions]
        threshold_scores.append({"threshold":th,"macro_f1":float(np.mean(values))})
    # Predetermined tie break: smallest threshold among equal scores.
    chosen=max(threshold_scores,key=lambda r:(r["macro_f1"],-r["threshold"]))["threshold"]
    elapsed=time.monotonic()-start
    write(out/"selection.json",{"checkpoint_rule":"minimum_inner_dev_continuous_BCE",
        "best_epoch":best_epoch,"best_inner_bce":best,"thresholds":threshold_scores,
        "chosen_threshold":chosen,"training_seconds":elapsed,"outer_inspected":False})
    torch.save({"model":{k:v.detach().cpu().half() for k,v in model.state_dict().items()}},out/"head_fp16.pt")
    job={"run_id":out.name+"_outer","checkpoint":str(out/"best.pt"),"manifest":str(out/"outer_test.jsonl"),
         "threshold":chosen,"model":model_name,"fold_id":fold["fold_id"],"repeat":fold["repeat"],"seed":seed,
         "split_role":"outer_cv_evaluation_only","training_seconds":elapsed,"head_weight_bytes":(out/"head_fp16.pt").stat().st_size,
         'head_type':head_type,'head_parameter_count':sum(p.numel() for p in model.parameters())}
    del data,predictions,model,opt,bundle
    torch.cuda.empty_cache()
    evaluate(job,cfg["gt_root"],str(out),device)
    print(json.dumps({"complete":out.name,"training_seconds":elapsed,"threshold":chosen}),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);ap.add_argument("--model",required=True)
    a=ap.parse_args();cfg=json.loads(Path(a.config).read_text());torch.set_num_threads(4)
    protocol=json.loads(Path(cfg["protocol"]).read_text())
    records={}
    for manifest in cfg["models"][a.model]["manifests"]:
        for line in Path(manifest).read_text().splitlines():
            r=json.loads(line);records[r["video_id"]]=r
    assert set(records)==set(protocol["video_ids"])
    for fold in protocol["folds"]:
        for seed in cfg["seeds"]:
            run_id=f"{cfg.get('run_prefix','RG_NCV')}_{a.model}_r{fold['repeat']}_f{fold['fold_id']}_s{seed}"
            out=Path(cfg["output"])/run_id
            if (out/(run_id+"_outer")/"metrics.json").exists():continue
            train_one(cfg,fold,a.model,seed,out,torch.device("cuda:0"),records)
if __name__=="__main__":main()
