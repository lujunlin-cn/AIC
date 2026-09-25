"""Reuse audited raw-backbone caches for a controlled temporal-head OOD check."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from aic.temporal_metrics import ranking_report,summary_mask
from scripts.benchmark_representations import load_head
from scripts.benchmark_summe_baselines import native_f1


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--model',required=True);a=ap.parse_args()
    cfg=json.loads(Path(a.config).read_text());out=Path(cfg['output'])/a.model;out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    checkpoints=sorted(Path(cfg['cv_root']).glob(cfg['run_prefix']+'_'+a.model+'_*/best.pt'))
    if len(checkpoints)!=cfg['expected_heads']:raise ValueError('incomplete preregistered heads')
    data=[];manifest=[]
    for cohort_id,root in enumerate(cfg['feature_roots'],1):
        for p in sorted((Path(root)/a.model).glob('*.npz')):
            with np.load(p) as z:
                data.append((p.stem,torch.from_numpy(z['features']).float().unsqueeze(0).to('cuda:0'),z['frame_indices'],z['human'],z['relevance'],cohort_id))
            manifest.append({'video_id':p.stem,'cohort':cohort_id,'cache':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    if len(data)!=14 or len({x[0] for x in data})!=14:raise ValueError('expected exact 14 independent source caches')
    start=time.perf_counter();runs=[]
    for ck in checkpoints:
        model,shift,_=load_head(ck,'cuda:0');assert not shift
        threshold=json.loads((ck.parent/'selection.json').read_text())['chosen_threshold'];per=[]
        with torch.inference_mode():
            for vid,x,idx,human,rel,cohort in data:
                p=model(x)[0].sigmoid().cpu().numpy();full=np.interp(np.arange(len(rel)),idx,p)
                mask=summary_mask((full-full.min())/np.ptp(full) if np.ptp(full) else np.zeros_like(full))
                per.append({'video_id':vid,'cohort':cohort,**native_f1(mask,human),**ranking_report(full,rel),
                            'threshold':threshold,'prediction_rate':float((full>=threshold).mean())})
        runs.append({'run_id':ck.parent.name,'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest(),'per_video':per})
    result={'model':a.model,'runs':runs,'manifest':manifest,'config':cfg,'elapsed_seconds':time.perf_counter()-start,
            'scope':'head-only ablation using already raw-extracted matched backbone features; no new backbone generalization claim',
            'limitation':'SumMe already exposed in prior representation comparisons; this is exploratory external confirmation, not pristine test',
            'official_f_video':None,'competition_score':None}
    (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n');print(a.model,'complete',len(runs))


if __name__=='__main__':main()
