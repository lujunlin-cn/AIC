"""Frozen weak-label heads versus native QVH human saliency (sanity only).

Query-free model scored separately per native query on human-rated clips only.
This is NOT QVH native mAP and not full system OOD (source videos exposed).
No selection, thresholds, checkpoint changes or training from human labels.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from aic.features import FeatureCacheDataset
from aic.models import TemporalUNet


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--annotations',type=Path,required=True);p.add_argument('--source-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    torch.set_num_threads(4)
    source={r['video_id']:r['source_id'] for r in map(json.loads,a.source_manifest.read_text().splitlines())}
    annotations={}
    for split in ('train','val'):
        for row in map(json.loads,(a.annotations/(split+'.jsonl')).read_text().splitlines()):
            annotations.setdefault(row['vid'],[]).append({**row,'native_split':split})
    results={}; variant_names=['videomaev2','deits_clip_mean','a0_clip_mean','vitb_clip_mean']
    for variant in variant_names:
        root=a.run/variant;state=torch.load(root/'temporal_best.pt',map_location='cpu',weights_only=True)
        model=TemporalUNet(state['input_dim']);model.load_state_dict(state['model']);model.eval()
        cache=FeatureCacheDataset(root/'val.jsonl');records=[]
        with torch.inference_mode():
            for item in cache:
                qs=annotations.get(source[item['video_id']],[])
                x=torch.from_numpy(item['features'])[None]
                scores=model(x,lengths=torch.tensor([x.shape[1]]))[0].sigmoid().numpy()
                for q in qs:
                    idx=np.asarray(q['relevant_clip_ids']);human=np.asarray(q['saliency_scores'],dtype=float).mean(1)
                    # Native clips are 2 seconds. Center alignment is explicit.
                    pred=np.interp(idx*2.+1.,item['timestamps'],scores)
                    valid=len(human)>1 and np.std(human)>0 and np.std(pred)>0
                    records.append({'video_id':item['video_id'],'qid':q['qid'],'native_split':q['native_split'],
                                    'annotated_clips':len(idx),'spearman':float(spearmanr(pred,human).statistic) if valid else None})
        valid=[r['spearman'] for r in records if r['spearman'] is not None]
        results[variant]={'queries':len(records),'defined_queries':len(valid),
                          'spearman_query_macro':float(np.mean(valid)) if valid else None,'per_query':records,
                          'checkpoint_sha256':hashlib.sha256((root/'temporal_best.pt').read_bytes()).hexdigest()}
    report={'protocol':'QVH_HUMAN_ANNOTATED_CLIPS_SANITY_V1','scope':'previously exposed weak-label validation videos; query-free scores; native human annotated clips only; not native QVH AP or independent OOD','official_f_video':None,'models':results}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:{z:v[z] for z in ['queries','defined_queries','spearman_query_macro']} for k,v in results.items()},indent=2))

if __name__=='__main__':main()
