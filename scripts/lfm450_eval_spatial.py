"""LFM2.5-VL-450M zero-shot spatial probe on the frozen RetargetVid dev set (v2).

Frozen before any dev-set query (the 3-frame format smoke of two draft
prompts failed and is kept as evidence in smoke_fp16_mt4.json):
  - processor: official defaults from the checkpoint (no tile override)
  - NPU fp16, attn eager, torch.npu jit_compile=False (binary kernels; the
    default jit mode recompiles per shape at ~300 s/query)
  - greedy decoding, max_new_tokens 64
  - two prompts, both in the model card's grounding format:
      GROUND   ratio-agnostic 'main subject' detection (one query per keyframe,
               reused for both ratio tasks)
      GROUNDR  same format plus the target aspect ratio (one query per ratio)
  - geometry rule (fixed): largest valid box -> centre -> hold within shot
    (qwen_centres, B0 raw fallback on invalid/missing keyframes) -> B0 EMA
    alpha .25 -> max window of the target ratio, clipped to the frame.

Every keyframe of the 1 s teacher pool is queried, so LFM and the historical
teacher point (QWEN_T, diagnostic reference only) see identical frames and the
identical path.  LFM never reads teacher outputs.
"""
import argparse, json, os, re, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='/data/aic/pretrained/lfm2_5_vl_450m')
ap.add_argument('--manifest', type=Path, required=True)
ap.add_argument('--cache', type=Path, default=Path('/data/aic/experiments/OBS_CACHE_RETARGET_ALL200'))
ap.add_argument('--points-dir', type=Path, default=Path('/data/aic/experiments/QWEN_RV200_POINT_D1/points'))
ap.add_argument('--frames-dir', type=Path, default=Path('/data/aic/experiments/QWEN_RV200_KEYFRAMES/d1'))
ap.add_argument('--cards', default='4')
ap.add_argument('--dtype', default='float16', choices=['float16', 'bfloat16'])
ap.add_argument('--attn', default='eager')
ap.add_argument('--max-new', type=int, default=64)
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshards', type=int, default=1)
ap.add_argument('--output-dir', type=Path, required=True)
args = ap.parse_args()

os.environ['ASCEND_RT_VISIBLE_DEVICES'] = args.cards

import torch, torch_npu  # noqa: E402
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoProcessor, Lfm2VlForConditionalGeneration  # noqa: E402

import sys  # noqa: E402
sys.path.insert(0, '/root/AIC')
from aic.max_window_path import centre_to_offset, ema_offsets, geometry, qwen_centres, qwen_centres_interp, to_crops  # noqa: E402
from scripts.max_window_path_eval import RATIOS, boxes, load_gt  # noqa: E402
from scripts.benchmark_spatial import iou  # noqa: E402

FMT = ('Response must be a JSON array: [{"label": ..., "bbox": [x1, y1, x2, y2]}, ...]. '
       'Coordinates are normalized to [0,1].')
PROMPT_GROUND = 'Detect all instances of: main subject. ' + FMT


def prompt_groundr(rw, rh):
    shape = 'tall portrait' if rw < rh else 'wide landscape'
    return (f'This frame will be cropped to a {rw}:{rh} {shape} window. '
            f'Detect the main subject that must stay inside the crop. ' + FMT)


BOX_RE = re.compile(r'"bbox"\s*:\s*\[\s*([-\d.eE]+)\s*,\s*([-\d.eE]+)\s*,\s*([-\d.eE]+)\s*,\s*([-\d.eE]+)\s*\]')


def parse_boxes(reply):
    """Returns (largest valid box [x1,y1,x2,y2] in [0,1], status, n_boxes)."""
    found = BOX_RE.findall(reply)
    if not found:
        return None, 'no_bbox', 0
    valid = []
    for g in found:
        try:
            x1, y1, x2, y2 = [float(v) for v in g]
        except ValueError:
            continue
        if min(x1, y1, x2, y2) < -0.02 or max(x1, y1, x2, y2) > 1.02:
            continue
        x1, y1, x2, y2 = [min(max(v, 0.), 1.) for v in (x1, y1, x2, y2)]
        if x2 - x1 < 0.005 or y2 - y1 < 0.005:
            continue
        valid.append([x1, y1, x2, y2])
    if not valid:
        return None, 'invalid_bbox', len(found)
    best = max(valid, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    return best, 'ok', len(found)


t0 = time.time()
proc = AutoProcessor.from_pretrained(args.model)
model = Lfm2VlForConditionalGeneration.from_pretrained(
    args.model, dtype=getattr(torch, args.dtype), low_cpu_mem_usage=True,
    attn_implementation=args.attn).eval().to('npu')
load_s = time.time() - t0
dev = json.loads(args.manifest.read_text())
vids = [v['vid'] for v in dev['videos']][args.shard::args.nshards]
out_dir = args.output_dir
(out_dir / 'crops').mkdir(parents=True, exist_ok=True)
print(json.dumps({'load_s': round(load_s, 1), 'shard': args.shard, 'nshards': args.nshards,
                  'n_videos': len(vids)}), flush=True)

tq = 0.0
nq = 0


def ask(img, text):
    global tq, nq
    msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': img}, {'type': 'text', 'text': text}]}]
    x = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                 return_dict=True, return_tensors='pt').to('npu')
    x['pixel_values'] = x['pixel_values'].to(model.dtype)
    t = time.time()
    with torch.no_grad():
        out = model.generate(**x, max_new_tokens=args.max_new, do_sample=False)
    torch.npu.synchronize()
    dt = time.time() - t
    tq += dt
    nq += 1
    return proc.batch_decode(out[:, x['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip(), dt


def legal(c, W, H, ratio):
    c = np.asarray(c, float)
    w = c[:, 2]
    h = w * ratio[1] / ratio[0]
    return bool((c[:, 0] >= -1e-6).all() and (c[:, 1] >= -1e-6).all() and (c[:, 0] + w <= W + 1e-6).all()
                and (c[:, 1] + h <= H + 1e-6).all())


rows_f = (out_dir / f'rows_s{args.shard}.jsonl').open('a')
raw_f = (out_dir / f'raw_s{args.shard}.jsonl').open('a')
for vid in vids:
    m = next(v for v in dev['videos'] if v['vid'] == vid)
    meta = json.loads((args.points_dir / f'{vid}.json').read_text())
    W, H = meta['W'], meta['H']
    kfs = meta['keyframes']
    imgs = {kf: Image.open(args.frames_dir / vid / f'{kf}.png').convert('RGB')
            for kf in kfs if (args.frames_dir / vid / f'{kf}.png').exists()}
    ground = {}
    for kf, img in imgs.items():
        reply, dt = ask(img, PROMPT_GROUND)
        ground[kf] = (*parse_boxes(reply), reply)
        raw_f.write(json.dumps({'vid': vid, 'kf': kf, 'prompt': 'GROUND', 'ratio': None, 's': round(dt, 3),
                                'status': ground[kf][1], 'n_boxes': ground[kf][2], 'box': ground[kf][0],
                                'reply': reply[:300]}) + '\n')
    for rk, ratio in RATIOS.items():
        z = np.load(args.cache / f'{vid}_{rk}.npz')
        reset = z['reset'].astype(bool)
        raw = z['raw']
        n = len(reset)
        w, h, axis = geometry(W, H, ratio)
        comp = 0 if axis == 0 else 1
        gt = load_gt(dev['annotations'], vid, rk)
        if gt.shape[1] != n:
            raise ValueError(f'{vid}_{rk} frame mismatch {gt.shape[1]} vs {n}')
        groundr = {}
        for kf, img in imgs.items():
            reply, dt = ask(img, prompt_groundr(*ratio))
            groundr[kf] = (*parse_boxes(reply), reply)
            raw_f.write(json.dumps({'vid': vid, 'kf': kf, 'prompt': 'GROUNDR', 'ratio': rk, 's': round(dt, 3),
                                    'status': groundr[kf][1], 'n_boxes': groundr[kf][2], 'box': groundr[kf][0],
                                    'reply': reply[:300]}) + '\n')
        raw_f.flush()
        qj = meta['ratios'][rk]
        obs = {'LFM_GROUND': [[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, False] if b else None
                              for b in (ground.get(kf, (None,))[0] for kf in kfs)],
               'LFM_GROUNDR': [[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, False] if b else None
                               for b in (groundr.get(kf, (None,))[0] for kf in kfs)],
               'QWEN_T': qj['points']}
        pol = {'B0': np.asarray(z['b0']),
               'CENTER': np.asarray(to_crops(W, H, ratio, centre_to_offset(np.full(n, .5), W, H, ratio)))}
        fb = {}
        for name, pts in obs.items():
            qc, src = qwen_centres(pts, kfs, reset, raw[:, comp], comp)
            pol[name] = np.asarray(to_crops(W, H, ratio, ema_offsets(np.asarray(qc), reset, W, H, ratio)))
            qi, _ = qwen_centres_interp(pts, kfs, reset, raw[:, comp], comp)
            pol[name + '+INTERP'] = np.asarray(to_crops(W, H, ratio, ema_offsets(np.asarray(qi), reset, W, H, ratio)))
            fb[name] = float((src == 'b0').mean())
        # per-keyframe localisation diagnostic on the free axis (vs mean GT crop centre)
        gtc = (gt[:, :, comp] + gt[:, :, comp + 2]) / 2 / (W if comp == 0 else H)
        gt_lo = gt[:, :, comp].mean(0) / (W if comp == 0 else H)
        gt_hi = gt[:, :, comp + 2].mean(0) / (W if comp == 0 else H)
        loc = {}
        for name, pts in list(obs.items()) + [('B0_RAW', [[raw[k][0], raw[k][1], False] for k in kfs])]:
            err, hit = [], []
            for kf, p in zip(kfs, pts):
                if p is None or (len(p) > 2 and p[2]) or kf >= n:
                    continue
                c = p[comp]
                err.append(abs(c - gtc[:, kf].mean()))
                hit.append(gt_lo[kf] <= c <= gt_hi[kf])
            loc[name] = {'n': len(err), 'abs_err': float(np.mean(err)) if err else None,
                         'hit': float(np.mean(hit)) if hit else None}
        np.savez_compressed(out_dir / 'crops' / f'{vid}_{rk}.npz',
                            **{k.replace('+', '_'): v for k, v in pol.items()},
                            kfs=np.asarray(kfs),
                            ground_boxes=np.asarray([ground.get(kf, (None,))[0] or [np.nan] * 4 for kf in kfs], float),
                            groundr_boxes=np.asarray([groundr.get(kf, (None,))[0] or [np.nan] * 4 for kf in kfs], float))
        for name, c in pol.items():
            rec = {'vid': vid, 'ratio': rk, 'axis': 'x' if comp == 0 else 'y', 'policy': name,
                   'iou': float(iou(boxes(c, ratio)[None], gt).mean()), 'legal': legal(c, W, H, ratio),
                   'layer': m['layer'], 'motion_px': m['motion_px'], 'shot_rate': m['shot_rate'],
                   'face_rate': m['face_rate'], 'n_frames': n, 'n_kf': len(kfs)}
            base = name.split('+')[0]
            if base in fb:
                rec['fallback_rate'] = fb[base]
            if base in loc:
                rec['loc'] = loc[base]
            rows_f.write(json.dumps(rec) + '\n')
        rows_f.write(json.dumps({'vid': vid, 'ratio': rk, 'policy': 'B0_RAW', 'loc': loc['B0_RAW']}) + '\n')
        rows_f.flush()
    print(f'{vid} done  avg {tq / max(nq, 1):.3f}s/query  n={nq}', flush=True)

summ = {'shard': args.shard, 'n_queries': nq, 's_queries': round(tq, 1), 'avg_s_per_query': round(tq / max(nq, 1), 3),
        'peak_mem_gib': round(torch.npu.max_memory_allocated() / 2**30, 3), 'load_s': round(load_s, 1),
        'prompts': {'GROUND': PROMPT_GROUND, 'GROUNDR_1-3': prompt_groundr(1, 3), 'GROUNDR_3-1': prompt_groundr(3, 1)},
        'dtype': args.dtype, 'attn': args.attn, 'jit_compile': False, 'max_new': args.max_new,
        't_done': time.strftime('%Y-%m-%d %H:%M:%S')}
(out_dir / f'summary_s{args.shard}.json').write_text(json.dumps(summ, indent=1) + '\n')
print('SUMMARY', json.dumps(summ), flush=True)
