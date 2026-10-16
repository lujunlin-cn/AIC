#!/usr/bin/env python3
"""Frozen, CPU-only candidate adequacy diagnostic on the 24 public pilot rows.

GT-assisted oracle is never a deployable selector or a learnability estimate.
This script uses its own scalar rectangle calculation, no model or image access.
"""
import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def iou(a, b):
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / max(union, 1e-9)


def key(row):
    return str(row['video_id']), int(row['frame']), tuple(row['ratio'])


def run(args):
    started = time.monotonic()
    protocol = json.loads(args.protocol.read_text())
    if protocol['candidate_count'] != 129 or protocol['expected_rows'] != 24:
        raise ValueError('This diagnostic is frozen at 129 candidates and 24 public rows')
    gt_rows = json.loads(args.sampled_gt.read_text())
    gt = {key(r): r for r in gt_rows}
    predictions = [json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()]
    if len(gt) != len(gt_rows) or len({key(r) for r in predictions}) != len(predictions):
        raise ValueError('Duplicate row keys')
    if len(predictions) != protocol['expected_rows'] or {key(r) for r in predictions} != set(gt):
        raise ValueError('Only complete public pilot predictions are accepted')
    if {str(r['video_id']) for r in predictions} != set(protocol['sources']):
        raise ValueError('Unexpected source set')
    per = []
    for row in predictions:
        reference = gt[key(row)]
        if row['png_sha256'] != reference['png_sha256']:
            raise ValueError('Prediction/image identity mismatch')
        truths = reference.get('gt', reference.get('gt_boxes'))
        if len(truths) != protocol['annotations_per_row']:
            raise ValueError('Unexpected annotation count')
        width, height = map(float, reference['image_wh'])
        rw, rh = map(float, row['ratio'])
        crop_w, crop_h = min(width, height * rw / rh), min(height, width * rh / rw)
        x_slide = crop_w < width - 1e-8
        axis, axis_index = ('x', 0) if x_slide else ('y', 1)
        travel = width - crop_w if x_slide else height - crop_h
        if travel <= 0 or row['axis'] != ('horizontal' if x_slide else 'vertical'):
            raise ValueError('Unexpected free-axis geometry')
        n = protocol['candidate_count']
        candidates = []
        for index in range(n):
            offset = travel * index / (n - 1)
            left, top = (offset, 0.) if x_slide else (0., offset)
            candidates.append([left, top, left + crop_w, top + crop_h])
        expected_dims = [crop_w, crop_h]
        actual_dims = [row['crop'][2] - row['crop'][0], row['crop'][3] - row['crop'][1]]
        if any(abs(a - b) > 1e-7 for a, b in zip(expected_dims, actual_dims)):
            raise ValueError('Qwen crop has different size from frozen maximum window')
        utility = [mean(iou(candidate, truth) for truth in truths) for candidate in candidates]
        oracle_index = max(range(n), key=lambda index: utility[index])
        projected_index = min(range(n), key=lambda index: abs(candidates[index][axis_index] - row['crop'][axis_index]))
        center_index = (n - 1) // 2
        qwen = mean(iou(row['crop'], truth) for truth in truths)
        if abs(qwen - row['iou']) > 1e-12:
            raise ValueError('Original saved Qwen continuous IoU does not reproduce')
        qwen_is_center = max(abs(a - b) for a, b in zip(row['crop'], candidates[center_index])) <= 1e-7
        per.append({
            'video_id': row['video_id'], 'frame': row['frame'], 'ratio': row['ratio'], 'axis': axis,
            'center': utility[center_index], 'qwen': qwen, 'projected_qwen': utility[projected_index],
            'oracle': utility[oracle_index], 'oracle_index': oracle_index, 'oracle_crop': candidates[oracle_index],
            'projected_index': projected_index, 'projected_crop': candidates[projected_index],
            'qwen_crop': row['crop'], 'qwen_is_center': qwen_is_center,
            'qwen_minus_center': qwen - utility[center_index],
            'projection_minus_qwen': utility[projected_index] - qwen,
            'oracle_minus_qwen': utility[oracle_index] - qwen,
            'oracle_minus_center': utility[oracle_index] - utility[center_index],
            'oracle_center_distance_pixels': abs(candidates[oracle_index][axis_index] - candidates[center_index][axis_index]),
            'utility_curve': utility,
        })
    metrics = {}
    for axis in ['all', 'x', 'y']:
        sources = defaultdict(list)
        for row in per:
            if axis == 'all' or row['axis'] == axis:
                sources[row['video_id']].append(row)
        per_source = {source: {method: mean(row[method] for row in rows)
                               for method in ['center', 'qwen', 'projected_qwen', 'oracle']}
                      for source, rows in sources.items()}
        metrics[axis] = {'rows': sum(len(rows) for rows in sources.values()), 'sources': len(sources),
                         'per_source': per_source,
                         'source_macro': {method: mean(scores[method] for scores in per_source.values())
                                          for method in ['center', 'qwen', 'projected_qwen', 'oracle']},
                         'qwen_is_center_rows': sum(row['qwen_is_center'] for row in per if axis == 'all' or row['axis'] == axis)}
    report = {'status': 'COMPLETE_EXPLORATORY_GT_ASSISTED_DIAGNOSTIC', 'protocol': protocol,
              'protocol_sha256': sha(args.protocol), 'predictions_sha256': sha(args.predictions),
              'sampled_gt_sha256': sha(args.sampled_gt), 'script_sha256': sha(Path(__file__)),
              'cpu_wall_seconds': time.monotonic() - started, 'metrics': metrics, 'per_frame': per}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ['protocol', 'per_frame']}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--sampled-gt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
