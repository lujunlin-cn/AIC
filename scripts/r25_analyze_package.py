"""Audit R25 geometry and temporal budget without labels or raw video access.

Publish aggregates only; no per-video test predictions or head scores.
"""
import argparse
import hashlib
import json
import math
import statistics
import zipfile
from pathlib import Path


def distribution(values):
    values = sorted(values)
    def quantile(p):
        x = (len(values) - 1) * p
        lo, hi = math.floor(x), math.ceil(x)
        return values[lo] + (values[hi] - values[lo]) * (x - lo)
    return dict(n=len(values), min=values[0], p10=quantile(.1),
                median=quantile(.5), p90=quantile(.9), max=values[-1],
                mean=statistics.mean(values))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--package', required=True)
    ap.add_argument('--index', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    package = Path(args.package)
    manifest = json.loads((package / 'manifest.json').read_text())
    index_bytes = Path(args.index).read_bytes()
    assert hashlib.sha256(index_bytes).hexdigest() == manifest['run']['index_sha256']
    metas = {x['video_id']: x for x in map(json.loads, index_bytes.splitlines())}
    zpath = package / 'R25_AHA_BF16_K085.zip'
    assert hashlib.sha256(zpath.read_bytes()).hexdigest() == manifest['zip_sha256']
    with zipfile.ZipFile(zpath) as z:
        assert z.namelist() == ['predictions.jsonl']
        payload = z.read('predictions.jsonl')
    assert hashlib.sha256(payload).hexdigest() == manifest['predictions_sha256']
    rows = list(map(json.loads, payload.splitlines()))
    assert len(rows) == len(metas) == 426
    assert {x['video_id'] for x in rows} == set(metas)
    records = []; max_window = 0; boxes = 0; cell_splits = 0
    score_stds = [x['score_std'] for x in manifest['per_video']]
    for row in rows:
        meta = metas[row['video_id']]; n = meta['frame_count']; fps = meta['fps']
        selected = {p['frame'] for p in row['predictions']}
        assert len(selected) == len(row['predictions']) and selected <= set(range(n))
        kept = len(selected); target = max(1, round(.85*n))
        samples = sorted({min(n-1, max(0, round(t*fps))) for t in range(max(1, math.ceil(meta['duration'])))})
        # Exact nearest-sample cells, midpoint ties belong to earlier sample.
        bounds = [0] + [(a+b)//2+1 for a,b in zip(samples,samples[1:])] + [n]
        for a,b in zip(bounds,bounds[1:]):
            take = sum(f in selected for f in range(a,b))
            cell_splits += int(0 < take < b-a)
        blocks = sum(f == 0 or f-1 not in selected for f in selected)
        thirds = []
        for i in range(3):
            a,b = i*n//3,(i+1)*n//3
            thirds.append(sum(f in selected for f in range(a,b))/(b-a))
        records.append(dict(n=n,k=kept,target=target,p=kept/n,seconds=n/fps,
                            all_selected=kept==n,samples=len(samples),blocks=blocks,thirds=thirds,
                            first_sample_selected=samples[0] in selected,last_sample_selected=samples[-1] in selected))
        assert kept >= target
        ratio = meta['targetRatioWH'][0]/meta['targetRatioWH'][1]
        max_w = min(meta['width'], meta['height']*ratio)
        for p in row['predictions']:
            w = p['bboxes'][2]
            max_window += int(abs(w-max_w) < 1e-3)
            boxes += 1
    total=sum(x['n'] for x in records); kept=sum(x['k'] for x in records)
    target=sum(x['target'] for x in records)
    groups=[]
    for lo,hi in [(0,5),(5,10),(10,20),(20,60),(60,float('inf'))]:
        group=[x for x in records if lo <= x['seconds'] < hi]
        if group:
            groups.append(dict(seconds_range=[lo,hi if math.isfinite(hi) else None],
                               videos=len(group),macro_keep=statistics.mean(x['p'] for x in group),
                               all_selected=sum(x['all_selected'] for x in group),
                               mean_samples=statistics.mean(x['samples'] for x in group)))
    output=dict(schema='r25_package_aggregate_audit_v1', zip_sha256=manifest['zip_sha256'],
                index_sha256=manifest['run']['index_sha256'],videos=len(rows),total_frames=total,
                kept_frames=kept,exact_085_budget_frames=target,excess_frames_over_rounded_budget=kept-target,
                micro_keep=kept/total,macro_keep=statistics.mean(x['p'] for x in records),
                per_video_keep=distribution([x['p'] for x in records]),
                all_selected_videos=sum(x['all_selected'] for x in records),
                sample_counts=distribution([x['samples'] for x in records]),
                selected_run_counts=distribution([x['blocks'] for x in records]),
                whole_cell_splits=cell_splits,duration_groups=groups,
                one_selected_run_videos=sum(x['blocks']==1 for x in records),
                first_sample_selected_videos=sum(x['first_sample_selected'] for x in records),
                last_sample_selected_videos=sum(x['last_sample_selected'] for x in records),
                macro_keep_by_time_third=[statistics.mean(x['thirds'][i] for x in records) for i in range(3)],
                head_score_std=distribution(score_stds),
                maximal_legal_crop_fraction=max_window/boxes,
                limitations=['No GT: no temporal recall, IoU, or component error attribution.',
                             'No raw head scores: no saturation, ties, head correlation, or rank quality measurement.',
                             'Time-third statistics describe masks; they do not establish position bias or accuracy.'])
    assert cell_splits == 0
    Path(args.output).write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(output,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
