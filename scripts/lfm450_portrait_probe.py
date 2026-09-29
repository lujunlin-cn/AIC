"""LFM2.5-VL-450M native-portrait -> 16:9 probe (qualitative only).

Native portrait sources with local media and no sealed/confirm status
(YouTubeHighlights 5, SA-V 1, LaSOT 3 frame folders).  None carries a human
16:9 crop label, so this probe reports only format / geometry validity and
saves overlays: QUANTITATIVE EFFECT UNVERIFIED.  LIVE / RetargetVid
landscape->portrait results must not be read as evidence for this direction.

Same frozen GROUND prompt / geometry rule as scripts/lfm450_eval_spatial.py:
largest box -> centre (y) -> 16:9 max window (full width) clipped to frame.
8 evenly spaced frames per source.
"""
import argparse, json, os, re, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='/data/aic/pretrained/lfm2_5_vl_450m')
ap.add_argument('--cards', default='4')
ap.add_argument('--n-frames', type=int, default=8)
ap.add_argument('--output-dir', type=Path, required=True)
args = ap.parse_args()
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = args.cards

import torch, torch_npu  # noqa: E402
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False
import numpy as np  # noqa: E402
import av  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from transformers import AutoProcessor, Lfm2VlForConditionalGeneration  # noqa: E402

SOURCES = [
    ('yth:gymnastics:KldEfRhv7H0', 'video', '/data/aic/external_datasets/YouTubeHighlights/raw/mirror/youtube_highlights_full/gymnastics/videos/KldEfRhv7H0.mp4', 'val (selection-exposed)'),
    ('yth:gymnastics:zr5LSPMBpiI', 'video', '/data/aic/external_datasets/YouTubeHighlights/raw/mirror/youtube_highlights_full/gymnastics/videos/zr5LSPMBpiI.mp4', 'val (selection-exposed)'),
    ('yth:parkour:oT6g0I2Ok9o', 'video', '/data/aic/external_datasets/YouTubeHighlights/raw/mirror/youtube_highlights_full/parkour/videos/oT6g0I2Ok9o.mp4', 'val (selection-exposed)'),
    ('yth:skating:lBVAo0MpsY0', 'video', '/data/aic/external_datasets/YouTubeHighlights/raw/mirror/youtube_highlights_full/skating/videos/lBVAo0MpsY0.mp4', 'train'),
    ('yth:skiing:YRO_RQ3aBgg', 'video', '/data/aic/external_datasets/YouTubeHighlights/raw/mirror/youtube_highlights_full/skiing/videos/YRO_RQ3aBgg.mp4', 'train'),
    ('sav:sav_000001', 'video', '/data/aic/external_datasets/SA_V/annotations/upstream_repo/sav_dataset/example/sav_000001.mp4', 'train'),
    ('lasot:mouse-1', 'frames', '/data/aic/external_datasets/LaSOT/raw/mouse/mouse-1/img', 'val'),
    ('lasot:mouse-15', 'frames', '/data/aic/external_datasets/LaSOT/raw/mouse/mouse-15/img', 'train'),
    ('lasot:electricfan-20', 'frames', '/data/aic/external_datasets/LaSOT/raw/electricfan/electricfan-20/img', 'val'),
]
PROMPT = ('Detect all instances of: main subject. Response must be a JSON array: '
          '[{"label": ..., "bbox": [x1, y1, x2, y2]}, ...]. Coordinates are normalized to [0,1].')
BOX_RE = re.compile(r'"bbox"\s*:\s*\[\s*([-\d.eE]+)\s*,\s*([-\d.eE]+)\s*,\s*([-\d.eE]+)\s*,\s*([-\d.eE]+)\s*\]')


def largest_box(reply):
    v = []
    for g in BOX_RE.findall(reply):
        try:
            b = [float(x) for x in g]
        except ValueError:
            continue
        if min(b) < -0.02 or max(b) > 1.02:
            continue
        b = [min(max(x, 0.), 1.) for x in b]
        if b[2] - b[0] > .005 and b[3] - b[1] > .005:
            v.append(b)
    return max(v, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])) if v else None


def frames_of(kind, path, n):
    if kind == 'frames':
        fs = sorted(Path(path).glob('*.jpg'))
        idx = np.round(np.linspace(0, len(fs) - 1, n)).astype(int)
        return [Image.open(fs[i]).convert('RGB') for i in idx], idx.tolist()
    with av.open(path) as c:
        st = c.streams.video[0]
        nb = st.frames or int(float(c.duration) / 1e6 * float(st.average_rate))
        idx = np.round(np.linspace(0, nb - 1, n)).astype(int)
        want = {int(i): k for k, i in enumerate(idx)}
        out = [None] * n
        for i, fr in enumerate(c.decode(st)):
            if i in want:
                out[want[i]] = fr.to_image()
            if i >= idx.max():
                break
    return out, idx.tolist()


proc = AutoProcessor.from_pretrained(args.model)
model = Lfm2VlForConditionalGeneration.from_pretrained(args.model, dtype=torch.float16, low_cpu_mem_usage=True,
                                                       attn_implementation='eager').eval().to('npu')
args.output_dir.mkdir(parents=True, exist_ok=True)
res = []
for uid, kind, path, split in SOURCES:
    imgs, idx = frames_of(kind, path, args.n_frames)
    W, H = imgs[0].size
    cw, ch = W, W * 9 / 16
    tiles, recs = [], []
    ema = None
    for im, fi in zip(imgs, idx):
        x = proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'image', 'image': im}, {'type': 'text', 'text': PROMPT}]}],
                                     tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt').to('npu')
        x['pixel_values'] = x['pixel_values'].to(model.dtype)
        t = time.time()
        with torch.no_grad():
            o = model.generate(**x, max_new_tokens=64, do_sample=False)
        torch.npu.synchronize()
        dt = time.time() - t
        reply = proc.batch_decode(o[:, x['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip()
        b = largest_box(reply)
        cy = (b[1] + b[3]) / 2 if b else .5
        y0 = float(np.clip(cy * H - ch / 2, 0, H - ch))
        crop = [0., y0, cw, ch]
        legal = y0 >= -1e-6 and y0 + ch <= H + 1e-6 and abs(cw / ch - 16 / 9) < 1e-6
        recs.append({'frame': fi, 'reply': reply[:200], 'box': b, 'crop_xywh': crop, 'legal': bool(legal), 's': round(dt, 3)})
        d = ImageDraw.Draw(im)
        if b:
            d.rectangle([b[0] * W, b[1] * H, b[2] * W, b[3] * H], outline=(255, 220, 0), width=3)
        d.rectangle([0, y0, cw - 1, y0 + ch], outline=(255, 40, 40), width=4)
        d.text((4, 4), f'f{fi}', fill=(255, 255, 255))
        tiles.append(im.resize((W * 360 // H, 360)))
    tw = tiles[0].size[0]
    sheet = Image.new('RGB', (tw * len(tiles) + 4 * (len(tiles) - 1), 360 + 30), (20, 20, 20))
    for i, t in enumerate(tiles):
        sheet.paste(t, (i * (tw + 4), 30))
    ImageDraw.Draw(sheet).text((6, 8), f'{uid} [{split}] {W}x{H} -> 16:9   yellow=LFM box  red=final 16:9 crop   QUANTITATIVE EFFECT UNVERIFIED (no GT)', fill=(255, 255, 255))
    f = args.output_dir / f"{uid.replace(':', '_')}.png"
    sheet.save(f)
    res.append({'uid': uid, 'split': split, 'W': W, 'H': H, 'overlay': str(f), 'frames': recs,
                'parse_ok': sum(r['box'] is not None for r in recs), 'all_legal': all(r['legal'] for r in recs),
                'crop_y_std_frac': float(np.std([r['crop_xywh'][1] for r in recs]) / H)})
    print(uid, 'parse', res[-1]['parse_ok'], '/', len(recs), 'legal', res[-1]['all_legal'], flush=True)
(args.output_dir / 'portrait_probe.json').write_text(json.dumps({'status': 'QUANTITATIVE_EFFECT_UNVERIFIED_NO_GT',
                                                                 'sources': res}, indent=1, ensure_ascii=False) + '\n')
print('SUMMARY', sum(r['parse_ok'] for r in res), '/', sum(len(r['frames']) for r in res), 'parsed; legal all', all(r['all_legal'] for r in res))
