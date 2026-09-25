#!/usr/bin/env python3
"""Aggregate source-paired OOD/spatial evidence without selecting models."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.analyze_nested_cv import paired_bootstrap,stats

def ood(root):
    models={};files=[];keys=['native_mean_user_f1','native_max_user_f1','spearman','kendall_tau_b','ndcg_at_15pct']
    for p in sorted(Path(root).glob('*/metrics.json')):
        d=json.loads(p.read_text());runs=d.get('runs',[{'per_video':d.get('per_video',[])}]);files.append(str(p));models[d['model']]={}
        for vid in sorted(r['video_id'] for r in runs[0]['per_video']):
            models[d['model']][vid]={k:float(np.mean([v[k] for r in runs for v in r['per_video'] if v['video_id']==vid and v[k] is not None])) for k in keys}
    if set(models)!=set(['A0','DeiT_S','ViT_B']):raise ValueError('all three completed models required')
    ids=sorted(models['A0'])
    assert all(sorted(m)==ids for m in models.values())
    ci={}
    for a,b in [('A0','DeiT_S'),('A0','ViT_B'),('DeiT_S','ViT_B')]:
        ci[b+'-'+a]={k:paired_bootstrap([models[a][v][k] for v in ids],[models[b][v][k] for v in ids]) for k in keys}
    return {'means':{m:{k:stats([r[k] for r in rows.values()])['mean'] for k in keys} for m,rows in models.items()},
            'per_video':models,'paired':ci,'n_videos':len(ids),'heads_per_model':len(runs),'sources':files,
            'unit':'average all TVSum-frozen heads per source, then bootstrap source videos; not an ensemble',
            'caveat':'size-selected raw videos, annotation-ordinal clock, excludes frame-count mismatch; not full SumMe population',
            'official_f_video':None,'competition_score':None}

def spatial(path):
    d=json.loads(Path(path).read_text());rows=d['per_video'];ids=sorted({r['video_id'] for r in rows})
    out={'source':str(path),'unit':'source video after equal average of two ratios','comparisons':{},'kinematics':{}}
    for mode in d['methods']:
        v={m:[np.mean([r['iou'] for r in rows if r['video_id']==vid and r['mode']==m]) for vid in ids] for m in ['center',mode]}
        out['comparisons'][mode]=paired_bootstrap(v['center'],v[mode])
        out['kinematics'][mode]={k:float(np.mean([r[k] for r in rows if r['mode']==mode])) for k in ['center_error_normalized','velocity','acceleration','jerk','legality','face_detection_rate']}
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--ood');ap.add_argument('--spatial');ap.add_argument('--output',required=True);a=ap.parse_args()
    d=ood(a.ood) if a.ood else spatial(a.spatial);Path(a.output).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
    print(json.dumps(d.get('means',d.get('comparisons')),indent=2))
if __name__=='__main__':main()
