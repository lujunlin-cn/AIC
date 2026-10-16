#!/usr/bin/env python3
"""Continue the frozen R28 public spatial pilot without regenerating old rows.

Only the four explicitly configured, already exposed RetargetVid sources are
read. R28 prompt, processor input, sampling and generation settings are reused.
The old 13 rows are validated against the same PNGs, geometry and labels before
any model is loaded. Continuous raw-GT scores reproduce R28; native RV scores
add the documented clamp/round/inclusive convention as a separate measurement.
An outer ``timeout --signal=TERM --kill-after=30s 20m`` is required at launch.
"""
import argparse
import hashlib
import json
import re
import signal
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_center(raw):
    # Intentionally identical to R28; no response-format or prompt tuning.
    for match in re.finditer(r'\{[^{}]*\}', raw):
        try:
            c = json.loads(match.group())['center']
            if type(c) in (int, float) and np.isfinite(c) and 0 <= c <= 1:
                return float(c), True
        except (ValueError, KeyError, TypeError):
            pass
    return .5, False


def box(center, ratio):
    rw, rh = ratio
    width = min(640., 360. * rw / rh)
    height = min(360., 640. * rh / rw)
    axis = 'horizontal' if width < 640 - 1e-9 else 'vertical'
    x = np.clip(center * 640 - width / 2, 0, 640 - width) if axis == 'horizontal' else 0.
    y = np.clip(center * 360 - height / 2, 0, 360 - height) if axis == 'vertical' else 0.
    return np.array([x, y, x + width, y + height]), axis


def overlaps(crop, gt, inclusive=False):
    eps = 1. if inclusive else 0.
    inter = np.maximum(np.minimum(crop[2:], gt[:, 2:]) - np.maximum(crop[:2], gt[:, :2]) + eps, 0).prod(1)
    union = np.maximum(crop[2:] - crop[:2] + eps, 0).prod() + np.maximum(gt[:, 2:] - gt[:, :2] + eps, 0).prod(1) - inter
    return inter / np.maximum(union, 1e-9)


def measure(crop, center_crop, gt):
    raw = overlaps(crop, gt)
    ref = overlaps(center_crop, gt)
    clamped = np.maximum(gt, 0.)
    # Native RetargetVid: round predicted boundary coordinates, then +1 area;
    # its evaluator clamps negative annotation coordinates without rounding GT.
    native = overlaps(np.rint(crop), clamped, inclusive=True)
    native_ref = overlaps(np.rint(center_crop), clamped, inclusive=True)
    return {
        'iou': float(raw.mean()), 'center_iou': float(ref.mean()),
        'iou_continuous_clamped_gt': float(overlaps(crop, clamped).mean()),
        'center_iou_continuous_clamped_gt': float(overlaps(center_crop, clamped).mean()),
        'iou_native_rv': float(native.mean()), 'center_iou_native_rv': float(native_ref.mean()),
        'per_annotator_iou_continuous_raw_gt': raw.tolist(),
        'per_annotator_iou_native_rv': native.tolist(),
        'gt_negative_coordinates': int((gt < 0).sum()),
    }


def key(row):
    return str(row['video_id']), int(row['frame']), tuple(row['ratio'])


def prepare(cfg, previous_path):
    if cfg['sources'] != ['031', '032', '033', '034'] or cfg.get('protected_sources_access') or cfg.get('training'):
        raise ValueError('Continuation is restricted to the frozen four exposed public sources')
    if cfg['ratios'] != [[1, 3], [3, 1]] or cfg['frame_fractions'] != [.25, .5, .75]:
        raise ValueError('Unexpected change to the frozen R28 sampling')
    expected = []
    for vid in cfg['sources']:
        cached = {int(p.stem): p for p in (Path(cfg['public_keyframe_root']) / vid).glob('*.png')}
        if not cached:
            raise ValueError('Missing public cached PNGs for ' + vid)
        gt = {tuple(r): np.stack([
            np.loadtxt(Path(cfg['annotation_root']) / f'annotator_{a}' / f'{vid}_{r[0]}-{r[1]}.txt', delimiter=',')
            for a in range(1, 7)]) for r in cfg['ratios']}
        n = gt[tuple(cfg['ratios'][0])].shape[1]
        if any(a.shape != (6, n, 4) or not np.isfinite(a).all() for a in gt.values()):
            raise ValueError('Invalid public GT geometry/length/finite values: ' + vid)
        for frac in cfg['frame_fractions']:
            frame = min(cached, key=lambda k: (abs(k - int(frac * n)), k))
            if not 0 <= frame < n:
                raise ValueError('Cached frame outside annotation range')
            with Image.open(cached[frame]) as im:
                W, H = im.size
            if abs(W / H - 640 / 360) > .001:
                raise ValueError('Unexpected public RV image aspect ratio')
            for ratio in cfg['ratios']:
                expected.append({'video_id': vid, 'frame': frame, 'ratio': ratio,
                                 'png_path': cached[frame], 'png_sha256': sha(cached[frame]),
                                 'image_wh': [W, H], 'gt': gt[tuple(ratio)][:, frame],
                                 'gt_length': n, 'requested_fraction': frac})
    if len({key(r) for r in expected}) != 24:
        raise ValueError('Frozen sampling did not yield 24 distinct source/frame/ratio keys')
    by_key = {key(r): r for r in expected}
    previous = {}
    for line in previous_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        k = key(row)
        if k in previous or k not in by_key:
            raise ValueError('Duplicate/unexpected key in previous predictions: ' + str(k))
        e = by_key[k]
        if row['png_sha256'] != e['png_sha256']:
            raise ValueError('Previous PNG hash mismatch: ' + str(k))
        c, valid = parse_center(row['raw'])
        crop, axis = box(c, e['ratio'])
        if c != row['center'] or valid != row['parse_valid'] or axis != row['axis']:
            raise ValueError('Previous parsed center/axis mismatch: ' + str(k))
        if not np.allclose(crop, row['crop'], atol=1e-9, rtol=0):
            raise ValueError('Previous geometry mismatch: ' + str(k))
        computed = measure(crop, box(.5, e['ratio'])[0], e['gt'])
        if abs(computed['iou'] - row['iou']) > 1e-12 or abs(computed['center_iou'] - row['center_iou']) > 1e-12:
            raise ValueError('Previous R28 continuous score does not reproduce: ' + str(k))
        previous[k] = row
    if not previous:
        raise ValueError('No previous predictions; this is a continuation, not a new pilot')
    return expected, previous


def source_macro(rows, value):
    vids = sorted({r['video_id'] for r in rows})
    per = {v: float(np.mean([r[value] for r in rows if r['video_id'] == v])) for v in vids}
    return {'mean': float(np.mean(list(per.values()))), 'per_source': per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--previous', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--validate-only', action='store_true')
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    started = time.monotonic()
    expected, previous = prepare(cfg, args.previous)
    if args.validate_only:
        print(json.dumps({'status': 'INPUTS_VALIDATED', 'expected': len(expected), 'previous': len(previous),
                          'remaining': len(expected) - len(previous), 'previous_sha256': sha(args.previous)}))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    status = {'experiment_id': 'R29_R28_SPATIAL_CONTINUATION', 'status': 'RUNNING',
              'config_sha256': sha(args.config), 'previous_sha256': sha(args.previous),
              'script_sha256': sha(Path(__file__)), 'expected_rows': len(expected), 'previous_rows': len(previous),
              'scope': 'Frozen R28 public engineering pilot continuation; no official inference, no fresh holdout, no aligned B3 control'}
    (args.output / 'run_status.json').write_text(json.dumps(status, indent=2) + '\n')
    public_gt = [{k: (v.tolist() if isinstance(v, np.ndarray) else str(v) if isinstance(v, Path) else v)
                  for k, v in e.items()} for e in expected]
    (args.output / 'sampled_gt.json').write_text(json.dumps(public_gt, indent=2) + '\n')
    signal.signal(signal.SIGTERM, lambda sig, frame: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM/outer timeout')))
    rows = []
    model = None
    try:
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        torch.manual_seed(cfg['seed'])
        prop = torch.cuda.get_device_properties(0)
        env = {'python': sys.version, 'torch': torch.__version__, 'transformers': transformers.__version__,
               'cuda': torch.version.cuda, 'gpu': prop.name, 'gpu_total_bytes': prop.total_memory,
               'generation_settings': {k: cfg[k] for k in ['thinking', 'do_sample', 'max_new_tokens', 'seed', 'max_image_side']}}
        (args.output / 'environment_observed.json').write_text(json.dumps(env, indent=2) + '\n')
        load_start = time.monotonic()
        proc = AutoProcessor.from_pretrained(cfg['model'])
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            cfg['model'], dtype=torch.bfloat16, attn_implementation='eager').eval().to('cuda:0')
        torch.cuda.synchronize()
        load_seconds = time.monotonic() - load_start
        for e in expected:
            k = key(e)
            if k in previous:
                row = dict(previous[k], provenance='R28_validated_previous')
            else:
                img = Image.open(e['png_path']).convert('RGB')
                img.thumbnail((cfg['max_image_side'], cfg['max_image_side']))
                _, axis = box(.5, e['ratio'])
                prompt = cfg['prompt_template'].replace('{rw}', str(e['ratio'][0])).replace('{rh}', str(e['ratio'][1])).replace('{axis}', axis)
                msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': img}, {'type': 'text', 'text': prompt}]}]
                inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                 return_dict=True, return_tensors='pt',
                                                 enable_thinking=cfg['thinking']).to('cuda:0')
                ts = time.monotonic()
                with torch.inference_mode():
                    output = model.generate(**inputs, do_sample=cfg['do_sample'], max_new_tokens=cfg['max_new_tokens'])
                torch.cuda.synchronize()
                raw = proc.batch_decode(output[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
                center, valid = parse_center(raw)
                crop, axis = box(center, e['ratio'])
                row = {'video_id': e['video_id'], 'frame': e['frame'], 'png_sha256': e['png_sha256'],
                       'ratio': e['ratio'], 'axis': axis, 'raw': raw, 'parse_valid': valid,
                       'center': center, 'crop': crop.tolist(), 'seconds': time.monotonic() - ts,
                       'provenance': 'R29_continued_frozen_R28',
                       'model_input_ids_shape': list(inputs['input_ids'].shape),
                       'image_grid_thw': inputs['image_grid_thw'].tolist() if 'image_grid_thw' in inputs else None}
            row.update(measure(np.asarray(row['crop']), box(.5, e['ratio'])[0], e['gt']))
            rows.append(row)
            with (args.output / 'predictions.jsonl').open('a') as f:
                f.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)
        report = dict(status, status='COMPLETE', rows=len(rows), sources=len(cfg['sources']),
                      parse_valid=sum(r['parse_valid'] for r in rows),
                      resumed_rows=len(previous), new_inference_rows=len(rows) - len(previous),
                      runtime_unique_parameters=sum(p.numel() for p in model.parameters()),
                      parameter_count_note='PyTorch model.parameters() deduplicates shared Parameter objects; checkpoint buffers excluded. All inference modules in this model included; no candidate compliance claim.',
                      model_load_seconds=load_seconds, elapsed_seconds=time.monotonic() - started,
                      new_generation_seconds=sum(r['seconds'] for r in rows if r['provenance'] == 'R29_continued_frozen_R28'),
                      peak_vram_allocated=torch.cuda.max_memory_allocated(), peak_vram_reserved=torch.cuda.max_memory_reserved())
        for metric, ref in [('iou', 'center_iou'), ('iou_native_rv', 'center_iou_native_rv'),
                            ('iou_continuous_clamped_gt', 'center_iou_continuous_clamped_gt')]:
            report[metric] = source_macro(rows, metric)
            report[ref] = source_macro(rows, ref)
            report[metric + '_delta'] = {v: report[metric]['per_source'][v] - report[ref]['per_source'][v] for v in cfg['sources']}
            report[metric + '_by_axis'] = {
                axis: {'model': source_macro([r for r in rows if r['axis'] == axis], metric),
                       'center': source_macro([r for r in rows if r['axis'] == axis], ref)}
                for axis in ['horizontal', 'vertical']}
        report['gt_negative_coordinates'] = sum(r['gt_negative_coordinates'] for r in rows)
        report['predictions_sha256'] = sha(args.output / 'predictions.jsonl')
        report['sampled_gt_sha256'] = sha(args.output / 'sampled_gt.json')
        (args.output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
        status.update(status='COMPLETE', rows=len(rows), elapsed_seconds=time.monotonic() - started)
        print(json.dumps(report), flush=True)
    except BaseException as exc:
        status.update(status='INTERRUPTED_OR_FAILED', rows_written=len(rows),
                      exception_type=type(exc).__name__, exception=str(exc), elapsed_seconds=time.monotonic() - started)
        raise
    finally:
        (args.output / 'run_status.json').write_text(json.dumps(status, indent=2) + '\n')


if __name__ == '__main__':
    main()
