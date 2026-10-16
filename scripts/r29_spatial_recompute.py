#!/usr/bin/env python3
"""Independent scalar recomputation of R29 public spatial pilot records.

No model, images, NumPy, official test paths or shared R28/R29 scoring helpers.
The sampled-GT JSON is a public-only export made by the continuation script.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def rect_iou(a, b, inclusive=False):
    extra = int(inclusive)
    inter = max(0., min(a[2], b[2]) - max(a[0], b[0]) + extra) * max(0., min(a[3], b[3]) - max(a[1], b[1]) + extra)
    aa = max(0., a[2] - a[0] + extra) * max(0., a[3] - a[1] + extra)
    bb = max(0., b[2] - b[0] + extra) * max(0., b[3] - b[1] + extra)
    return inter / max(aa + bb - inter, 1e-9)


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def key(r):
    return str(r['video_id']), int(r['frame']), tuple(r['ratio'])


def center_box(ratio):
    rw, rh = ratio
    w = min(640., 360. * rw / rh)
    h = min(360., 640. * rh / rw)
    return [(640 - w) / 2, (360 - h) / 2, (640 + w) / 2, (360 + h) / 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', type=Path, required=True)
    ap.add_argument('--sampled-gt', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--allow-partial', action='store_true', help='Implementation check only; never use a partial run to select a method')
    args = ap.parse_args()
    gt_rows = json.loads(args.sampled_gt.read_text())
    for row in gt_rows:
        if 'gt' not in row:
            row['gt'] = row['gt_boxes']
    gt = {key(r): r for r in gt_rows}
    if len(gt) != len(gt_rows):
        raise ValueError('Duplicate sampled GT keys')
    predictions = [json.loads(l) for l in args.predictions.read_text().splitlines() if l.strip()]
    if len({key(r) for r in predictions}) != len(predictions):
        raise ValueError('Duplicate prediction keys')
    predicted_keys = {key(r) for r in predictions}
    if not predicted_keys.issubset(gt) or (not args.allow_partial and predicted_keys != set(gt)):
        raise ValueError('Expected complete predictions matching the sampled-GT export')
    per = []
    errors = []
    for r in predictions:
        g = gt[key(r)]
        if r['png_sha256'] != g['png_sha256']:
            raise ValueError('PNG identity mismatch')
        crop, ref = r['crop'], center_box(r['ratio'])
        native_crop, native_ref = list(map(round, crop)), list(map(round, ref))
        truth = g['gt']
        clamped = [[max(0., v) for v in b] for b in truth]
        scores = {
            'iou': mean(rect_iou(crop, b) for b in truth),
            'center_iou': mean(rect_iou(ref, b) for b in truth),
            'iou_continuous_clamped_gt': mean(rect_iou(crop, b) for b in clamped),
            'center_iou_continuous_clamped_gt': mean(rect_iou(ref, b) for b in clamped),
            'iou_native_rv': mean(rect_iou(native_crop, b, True) for b in clamped),
            'center_iou_native_rv': mean(rect_iou(native_ref, b, True) for b in clamped),
        }
        for metric, value in scores.items():
            if metric in r:
                errors.append(abs(value - r[metric]))
        per.append({'video_id': r['video_id'], 'frame': r['frame'], 'ratio': r['ratio'],
                    'axis': r['axis'], 'parse_valid': r['parse_valid'], 'center': r['center'], **scores})
    results = {}
    for mode in ['all', 'horizontal', 'vertical']:
        chunk = [r for r in per if mode == 'all' or r['axis'] == mode]
        source = defaultdict(list)
        for r in chunk:
            source[r['video_id']].append(r)
        scores = {}
        for model, control in [('iou', 'center_iou'), ('iou_native_rv', 'center_iou_native_rv'),
                               ('iou_continuous_clamped_gt', 'center_iou_continuous_clamped_gt')]:
            per_source = {v: {'model': mean(r[model] for r in rows), 'center': mean(r[control] for r in rows),
                              'delta': mean(r[model] - r[control] for r in rows)} for v, rows in source.items()}
            scores[model] = {'model_source_macro': mean(r['model'] for r in per_source.values()),
                             'center_source_macro': mean(r['center'] for r in per_source.values()),
                             'delta_source_macro': mean(r['delta'] for r in per_source.values()), 'per_source': per_source}
        results[mode] = scores
    report = {'status': 'RECOMPUTED', 'rows': len(per), 'expected_rows': len(gt),
              'complete': predicted_keys == set(gt), 'sources': len({r['video_id'] for r in per}),
              'max_abs_saved_metric_error': max(errors), 'saved_metrics_match': max(errors) <= 1e-12,
              'parse_valid': sum(r['parse_valid'] for r in per),
              'negative_gt_coordinates': sum(v < 0 for r in gt_rows for b in r['gt'] for v in b),
              'results': results, 'per_frame': per,
              'predictions_sha256': hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
              'sampled_gt_sha256': hashlib.sha256(args.sampled_gt.read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Independent arithmetic recomputation; same four exposed sources, not an independent generalization test or B3 comparison'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'per_frame'}))
    if not report['saved_metrics_match']:
        raise SystemExit('Saved metrics do not reproduce')


if __name__ == '__main__':
    main()
