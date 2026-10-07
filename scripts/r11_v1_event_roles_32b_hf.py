"""R11 V1: source-level event roles via Qwen3-VL-32B, transformers batch.

Fallback after the vllm-ascend 0.18.0 engine-init deadlock (four probes:
import OK, plugin activated, engine init hangs with zero weight IO on both
V1 and V0 engines -- recorded in reports/r11).  Uses the historically
verified teacher stack instead (0.70 qps recipe, scripts/
qwen_subject_point_batch.py): transformers + Qwen3VLForConditionalGeneration
+ device_map='auto' + eager attention, run under the MAIN venv
(/usr/local/python3.11.15/bin/python3) with ascend_teacher.env sourced.

One request per source (median 7 / p90 12 segments), 3 context thumbnails +
1 slot frame per segment, explicit segment-number JSON keys.  Output
compatible with scripts/r11_o1_v1_features.py (--e3-dir <out>).
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--model', default='/data/aic/pretrained/qwen3_vl_32b_instruct')
p.add_argument('--media-root', default='/data/aic/external_datasets/PHD2/raw/youtube')
p.add_argument('--audio-root', default='/data/aic/experiments_910a/LFM_V11/r10_audio')
p.add_argument('--manifest-dir', default='/data/aic/experiments_910a/LFM_V11/r8_npu/r9_audit')
p.add_argument('--out', default='/data/aic/experiments_910a/LFM_V11/r11_e3_roles_32b')
p.add_argument('--batch', type=int, default=4)
p.add_argument('--limit', type=int, default=0)
p.add_argument('--dtype', default='float16')
p.add_argument('--cards', default='2,3,4,5,6,7', help='physical NPU ids, visible via ASCEND_RT_VISIBLE_DEVICES')
a = p.parse_args()

import cv2  # noqa: E402
import av  # noqa: E402
from PIL import Image  # noqa: E402

os.environ['ASCEND_RT_VISIBLE_DEVICES'] = a.cards

SYSTEM = ('You are a professional video editor cutting a short highlight clip from a video. '
          'You first understand the event the video is about, then decide which moments the highlight clip needs.')
ROLES_BODY = ('For each segment choose its role in the event:\n'
              'MAIN = the main event or highlight action is happening, the moment a viewer is meant to watch;\n'
              'SETUP = build-up needed to understand or enjoy the main event;\n'
              'RESULT = the outcome or reaction right after the main event;\n'
              'IDLE = nothing of the event is happening (waiting, wandering, aimless footage);\n'
              'JUNK = black, blank, fading, title cards, credits, or unrelated content;\n'
              'UNSURE = you cannot tell (use only for unreadable frames).\n'
              'Return ONLY JSON: {{"roles":{{"<segment number>":"ROLE"}}}} with one entry for every segment. '
              'Use the segment numbers printed in the red tags as keys.')
ALL_R = {'MAIN', 'SETUP', 'RESULT', 'IDLE', 'JUNK', 'UNSURE'}
_JSON = re.compile(r'\{.*\}', re.DOTALL)


def parse_roles(text, segs):
    m = _JSON.search(text or '')
    if not m:
        return None, 'no_json'
    try:
        d = json.loads(m.group(0))
    except ValueError:
        return None, 'bad_json'
    r = d.get('roles', d)
    if not isinstance(r, dict):
        return None, 'bad_roles'
    out = {}
    for s in segs:
        v = r.get(str(s), r.get(s))
        v = str(v).strip().upper() if v is not None else None
        out[s] = v if v in ALL_R else 'UNSURE'
    if sum(1 for v in out.values() if v == 'UNSURE') > len(segs) / 2:
        return out, 'mostly_unsure'
    return out, 'ok'


man = []
for split in ('train_public', 'eval_public'):
    f = Path(a.manifest_dir) / f'{split}.jsonl'
    if f.exists():
        man += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
seen, by_src = set(), {}
for r in man:
    if r['fragment_id'] in seen:
        continue
    seen.add(r['fragment_id'])
    by_src.setdefault(r['source_id'], []).append(r['fragment_id'])
srcs = sorted(by_src)
if a.limit:
    srcs = srcs[:a.limit]
OUT = Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)
todo = [s for s in srcs if not (OUT / f'{s}.json').exists()]
print(f'{len(srcs)} sources, {len(todo)} to run', flush=True)

t_of = {}
for f in Path(a.audio_root).glob('p*/*.npz'):
    try:
        t_of[f.stem] = np.load(f)['t'].astype(np.float32)
    except Exception:
        pass

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402
torch_npu.npu.config.allow_internal_format = False
assert torch_npu.npu.device_count() > 0, 'no NPU visible'
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # noqa: E402

t0 = time.time()
proc = AutoProcessor.from_pretrained(a.model)
proc.tokenizer.padding_side = 'left'
model = Qwen3VLForConditionalGeneration.from_pretrained(
    a.model, dtype=getattr(torch, a.dtype), device_map='auto',
    low_cpu_mem_usage=True, attn_implementation='eager').eval()
print(f'model loaded {time.time() - t0:.0f}s devices='
      f'{sorted({str(p_.device) for p_ in model.parameters()})}', flush=True)


def decode_frames(vid_path, times):
    c = av.open(str(vid_path))
    st = c.streams.video[0]
    fps = float(st.average_rate)
    need = {max(int(round(t * fps)), 0): i for i, t in enumerate(times)}
    c.seek(max(int((min(need) - 2) / fps * av.time_base), 0), backward=True)
    imgs = {}
    for fr in c.decode(video=0):
        idx = int(round(fr.pts * fr.time_base * fps))
        if idx in need:
            imgs[need[idx]] = cv2.resize(fr.to_ndarray(format='rgb24'), (384, 384),
                                         interpolation=cv2.INTER_CUBIC)
        if idx >= max(need):
            break
    c.close()
    return [imgs.get(i) for i in range(len(times))]


def source_thumbs(vid_path, n=3, size=192):
    c = av.open(str(vid_path))
    st = c.streams.video[0]
    nfr = st.frames or 0
    if nfr <= 0:
        c.close()
        return []
    picks = sorted({int(nfr * k / (n + 1)) for k in range(1, n + 1)})
    out, want = [], set(picks)
    c.seek(0, backward=True)
    for fr in c.decode(video=0):
        idx = int(round(fr.pts * fr.time_base * float(st.average_rate)))
        if idx in want:
            im = fr.to_ndarray(format='rgb24')
            out.append(Image.fromarray(cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)))
            want.discard(idx)
            if not want:
                break
    c.close()
    return out


def build_request(media, fids):
    """Contact-sheet form: ALL segments become ONE gridded image with the
    segment number burned into each cell (validated envelope on this host =
    single-image short-prompt inference; multi-image long prompts hang)."""
    segmap = {}
    for f in fids:
        if f not in t_of:
            continue
        for t in t_of[f].tolist():
            segmap.setdefault(int(t // 2), float(t))
    if not segmap or not media.exists():
        return None
    segs = sorted(segmap)
    seg_ims = decode_frames(media, [segmap[s] + 0.5 for s in segs])
    cells = [(s, im) for s, im in zip(segs, seg_ims) if im is not None]
    if not cells:
        return None
    from PIL import ImageDraw
    cols = 4
    cw, ch = 320, 180
    rows = (len(cells) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * cw, rows * ch + 22), (16, 16, 16))
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 4), f'{len(cells)} segments of one video, numbered:', fill=(255, 255, 80))
    for k, (s, im) in enumerate(cells):
        x, y = (k % cols) * cw, 22 + (k // cols) * ch
        sheet.paste(Image.fromarray(cv2.resize(im, (cw, ch))), (x, y))
        draw.rectangle([x, y, x + 74, y + 20], fill=(200, 30, 30))
        draw.text((x + 4, y + 3), f'seg={s}', fill=(255, 255, 255))
    text = ('The image is a contact sheet of a short video: every cell shows one consecutive '
            '2-second segment, with its segment number printed in the red corner tag.\n' + ROLES_BODY)
    content = [{'type': 'image', 'image': sheet}, {'type': 'text', 'text': text}]
    msgs = [{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]},
            {'role': 'user', 'content': content}]
    return msgs, segs


n_ok = n_bad = 0
t0 = time.time()
for c0 in range(0, len(todo), a.batch):
    batch = todo[c0:c0 + a.batch]
    reqs = []
    for vid in batch:
        r = build_request(Path(a.media_root) / f'{vid}.mp4', by_src[vid])
        if r:
            reqs.append((vid, *r))
    if not reqs:
        continue
    texts = []
    for r in reqs:
        t = proc.apply_chat_template([r[1]], tokenize=False, add_generation_prompt=True)
        # transformers 5.x may return list[str]; proc(text=) needs flat str
        texts.append(t[0] if isinstance(t, list) else t)
    imgs = [[c['image'] for c in r[1][1]['content'] if c.get('type') == 'image'] for r in reqs]
    inputs = proc(text=texts, images=[im for lst in imgs for im in lst], padding=True,
                  return_tensors='pt').to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=400, do_sample=False)
    for k, (vid, _m, segs) in enumerate(reqs):
        reply = proc.decode(out[k][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        roles, status = parse_roles(reply, segs)
        if roles is None:
            roles, status = {}, status
        (OUT / f'{vid}.json').write_text(json.dumps(
            {'roles': {str(x): v for x, v in roles.items()}, 'status': status, 'raw': reply[:300]}))
        if status == 'ok':
            n_ok += 1
        else:
            n_bad += 1
    done = c0 + len(batch)
    if done % 20 < a.batch:
        print(f'{done}/{len(todo)} ok={n_ok} bad={n_bad} {time.time() - t0:.0f}s '
              f'({done / (time.time() - t0):.2f} src/s)', flush=True)
print(f'DONE {len(todo)} sources ok={n_ok} bad={n_bad} {time.time() - t0:.0f}s', flush=True)
