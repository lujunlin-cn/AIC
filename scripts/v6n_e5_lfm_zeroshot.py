"""V6-NEXT E5: LFM2.5-VL-450M zero-shot spatial protocol on V100 (GPU4).

Prereg: configs/V6_NEXT10H_PREREG.json (e5_lfm_zero_shot).
Manifest: configs/LFM450_SPATIAL_DEV_MANIFEST_V1.json (RV dev2 sources 031-100,
layers nonperson/fast_motion/shot_cuts/regular, target ratios 1:3 / 3:1).

Arms:
  S0  center crop at the target ratio (pure geometry, no model)
  S1  LFM bbox -> legal crop window (clamped, ratio-exact)
No human boxes exist -> geometry validity + parse health only, NO fabricated IoU.

Isolation: transformers>=5.1 lives in /data/aic/tools/hf51 (PYTHONPATH
prepended; cv env untouched).  V100: fp16, eager/sdpa attention only.

Stages:
  smoke   10 frames, raw replies printed, health verdict
  sweep   <=6 stratified keyframes per manifest video -> lfm_zero_shot.csv
"""
import argparse, csv, json, re, time
from pathlib import Path

import numpy as np
import torch

MODEL = '/data/aic/pretrained/lfm2_5_vl_450m_v100'
MANIFEST = Path('/home/supie/AIC/configs/LFM450_SPATIAL_DEV_MANIFEST_V1.json')
KEYFRAMES = Path('/data/aic/experiments/QWEN_RV200_KEYFRAMES/d1')

PROMPTS = {
    'P1': 'Locate the primary subject that should stay inside the crop. '
          'Reply with exactly one bounding box in the form <box>x1 y1 x2 y2</box> '
          'using integers normalized to 0-1000. No other text.',
    'P2': 'What is the main subject of this image? Answer in one short sentence, '
          'then output its bounding box as <box>x1 y1 x2 y2</box> with integers '
          'normalized to 0-1000.',
}
BOX_RE = re.compile(r'<box>\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*</box>')
NUM_RE = re.compile(r'(\d{1,4})\D+(\d{1,4})\D+(\d{1,4})\D+(\d{1,4})')


def legal_window(W, H, ratio, cx, cy):
    """Ratio-exact crop window centred near (cx, cy), clamped inside the frame."""
    rw, rh = ratio
    w = min(W, int(round(H * rw / rh)))
    h = int(round(w * rh / rw))
    if w > W:
        w = W
        h = int(round(W * rh / rw))
    x = int(round(cx - w / 2))
    y = int(round(cy - h / 2))
    x = max(0, min(W - w, x))
    y = max(0, min(H - h, y))
    return [x, y, x + w, y + h]


def center_window(W, H, ratio):
    return legal_window(W, H, ratio, W / 2, H / 2)


def load_model(max_tiles=6):
    from transformers import AutoProcessor, Lfm2VlForConditionalGeneration
    proc = AutoProcessor.from_pretrained(MODEL, max_tiles=max_tiles)
    model = Lfm2VlForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.float16, low_cpu_mem_usage=True,
        attn_implementation='sdpa').to('cuda').eval()
    return proc, model


def run_frame(proc, model, image, prompt_key, max_new=64):
    from PIL import Image
    im = Image.open(image).convert('RGB')
    W, H = im.size
    msgs = [{'role': 'user', 'content': [
        {'type': 'image', 'image': im},
        {'type': 'text', 'text': PROMPTS[prompt_key]}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = proc(text=[text], images=[im], return_tensors='pt').to('cuda')
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    dt = time.time() - t0
    reply = proc.batch_decode(out[:, inputs['input_ids'].shape[1]:],
                              skip_special_tokens=True)[0].strip()
    m = BOX_RE.search(reply) or NUM_RE.search(reply)
    parse_ok = m is not None and BOX_RE.search(reply) is not None
    oob = False
    bbox = None
    if m:
        x1, y1, x2, y2 = (int(v) / 1000.0 for v in m.groups())
        if not (0 <= x1 < x2 <= 1.001 and 0 <= y1 < y2 <= 1.001):
            oob = True
        else:
            bbox = (x1, y1, x2, y2)
    return {'W': W, 'H': H, 'reply': reply, 'parse_ok': parse_ok,
            'bbox': bbox, 'oob': oob, 'latency_s': round(dt, 3),
            'vram_peak_mb': round(torch.cuda.max_memory_allocated() / 2**20, 1)}


def stratified_frames(manifest, per_video=6):
    """Spread the keyframe list evenly; cover every layer at least once.

    Ratio is NOT a model input (bbox prompting is ratio-agnostic): both target
    ratios from the manifest are evaluated on the same single inference.
    """
    out = []
    for s in manifest['videos']:
        kfs = s['keyframes'][::max(1, len(s['keyframes']) // per_video)][:per_video]
        for f in kfs:
            out.append({'video': s['vid'], 'frame': f, 'layer': s['layer']})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=('smoke', 'sweep'))
    ap.add_argument('--prompts', nargs='*', default=['P1'])
    ap.add_argument('--per-video', type=int, default=6)
    ap.add_argument('--max-tiles', type=int, default=6)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text())
    proc, model = load_model(a.max_tiles)
    if a.stage == 'smoke':
        rows = []
        for s in manifest['videos'][:4]:
            for f in s['keyframes'][:1]:
                for pk in a.prompts:
                    img = KEYFRAMES / s['vid'] / f'{f}.png'
                    r = run_frame(proc, model, img, pk)
                    r.update({'video': s['vid'], 'frame': f, 'prompt': pk})
                    rows.append(r)
                    print(json.dumps({k: r[k] for k in ('video', 'frame', 'prompt', 'parse_ok', 'oob', 'latency_s', 'reply')}), flush=True)
        ok = sum(r['parse_ok'] for r in rows)
        print(json.dumps({'smoke': {'n': len(rows), 'parse_ok': ok,
                                    'mean_latency': round(float(np.mean([r['latency_s'] for r in rows])), 3)}}))
        return
    frames = stratified_frames(manifest, a.per_video)
    with a.out.open('w', newline='') as fcsv:
        w = csv.writer(fcsv)
        w.writerow(['video', 'frame', 'layer', 'ratio', 'prompt', 'arm', 'raw_output',
                    'parse_ok', 'oob', 'bbox', 'window', 'fallback_center', 'latency_s', 'vram_peak_mb'])
        n = 0
        for fr in frames:
            img = KEYFRAMES / fr['video'] / f"{fr['frame']}.png"
            if not img.exists():
                continue
            for pk in a.prompts:
                r = run_frame(proc, model, img, pk)
                for rk, rw in manifest['ratios'].items():
                    c = center_window(r['W'], r['H'], rw)
                    if r['bbox']:
                        x1, y1, x2, y2 = r['bbox']
                        win = legal_window(r['W'], r['H'], rw, (x1 + x2) / 2 * r['W'], (y1 + y2) / 2 * r['H'])
                        fb = False
                    else:
                        win, fb = c, True
                    w.writerow([fr['video'], fr['frame'], fr['layer'], rk, pk, 'S1',
                                json.dumps(r['reply']), int(r['parse_ok']), int(r['oob']),
                                json.dumps(r['bbox']), json.dumps(win), int(fb),
                                r['latency_s'], r['vram_peak_mb']])
                    w.writerow([fr['video'], fr['frame'], fr['layer'], rk, pk, 'S0',
                                '', 1, 0, '', json.dumps(c), 0, 0.0, 0])
                n += 1
                if n % 20 == 0:
                    print(json.dumps({'done': n, 'of': len(frames)}), flush=True)
    print(json.dumps({'sweep_frames': n, 'csv': str(a.out)}))


if __name__ == '__main__':
    main()
