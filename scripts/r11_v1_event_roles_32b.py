"""R11 V1: source-level event roles via Qwen3-VL-32B on vllm-ascend (kmsp05).

User-approved arm (2026-10-07).  Design follows the V4-E3 task shape, rebuilt
for the PHD2 pool with the lessons from the collapsed 7B arm:
  - ONE request per source (median 7 / p90 12 / max 15 segments per source --
    measured over the slot-pool time axes), so the model always sees the
    whole source: no isolated-window degeneracy.
  - 3 whole-source context thumbnails + 1 frame per segment (slot frames from
    the R10 audio npz time axes).
  - Segment-number keys are demanded explicitly (the 7B copy-the-K bug).
Engine: vllm-ascend v0.18.0 on the balanced venv (/data/aic/venvs/vllm18,
vllm 0.18.0 + torch 2.9.0 + torch_npu 2.9.x -- see the ABI repair notes).
Probe gate: only run after /tmp/vllm_probe.log shows `GEN OK`.

Output: <out>/<source_id>.json {"roles": {...}, "status", "raw"} --
consumed unchanged by scripts/r11_o1_v1_features.py (--e3-dir <out>).
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('VLLM_LOGGING_LEVEL', 'WARNING')

import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--model', default='/data/aic/pretrained/qwen3_vl_32b_instruct')
p.add_argument('--media-root', default='/data/aic/external_datasets/PHD2/raw/youtube')
p.add_argument('--audio-root', default='/data/aic/experiments_910a/LFM_V11/r10_audio')
p.add_argument('--manifest-dir', default='/data/aic/experiments_910a/LFM_V11/r8_npu/r9_audit')
p.add_argument('--out', default='/data/aic/experiments_910a/LFM_V11/r11_e3_roles_32b')
p.add_argument('--tp', type=int, default=4)
p.add_argument('--chunk', type=int, default=64, help='sources per llm.chat batch')
p.add_argument('--limit', type=int, default=0)
a = p.parse_args()

import cv2  # noqa: E402
import av  # noqa: E402
from PIL import Image  # noqa: E402

SYSTEM = ('You are a professional video editor cutting a short highlight clip from a video. '
          'You first understand the event the video is about, then decide which moments the highlight clip needs.')
ROLES = ('First {nctx} context frames from elsewhere in this source are shown, then {nseg} frames, each '
         'representing one consecutive 2-second segment of THIS source, labelled [seg=K t=A-Bs]. '
         'For each segment choose its role in the event:\n'
         'MAIN = the main event or highlight action is happening, the moment a viewer is meant to watch;\n'
         'SETUP = build-up needed to understand or enjoy the main event;\n'
         'RESULT = the outcome or reaction right after the main event;\n'
         'IDLE = nothing of the event is happening (waiting, wandering, aimless footage);\n'
         'JUNK = black, blank, fading, title cards, credits, or unrelated content;\n'
         'UNSURE = you cannot tell (use only for unreadable frames).\n'
         'Return ONLY JSON: {{"roles":{{"<segment number>":"ROLE"}}}} with one entry for every listed segment. '
         'Use the actual segment numbers from the [seg=...] labels as keys.')
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


t0 = time.time()
done = err = 0
for c0 in range(0, len(todo), a.chunk):
    batch = todo[c0:c0 + a.chunk]
    convs, metas = [], []
    for vid in batch:
        fids = sorted(by_src[vid], key=lambda f: float(t_of[f][0]) if f in t_of else 0)
        segmap = {}
        for f in fids:
            if f not in t_of:
                continue
            for t in t_of[f].tolist():
                segmap.setdefault(int(t // 2), float(t))
        if not segmap:
            continue
        segs = sorted(segmap)
        media = Path(a.media_root) / f'{vid}.mp4'
        if not media.exists():
            continue
        seg_times = [segmap[s] + 0.5 for s in segs]  # mid-segment frame
        seg_ims = decode_frames(media, seg_times)
        ctx_ims = source_thumbs(media)
        pil = [Image.fromarray(im) for im in seg_ims if im is not None]
        content = [{'type': 'text', 'text': f'{len(ctx_ims)} context frames:'}]
        content += [{'type': 'image_pil', 'image_pil': im} for im in ctx_ims]
        content.append({'type': 'text', 'text': f'{len(pil)} segment frames:'})
        for s, im in zip(segs, seg_ims):
            if im is not None:
                content.append({'type': 'text', 'text': f'[seg={s} t={2 * s}-{2 * s + 2}s]'})
                content.append({'type': 'image_pil', 'image_pil': Image.fromarray(im)})
        content.append({'type': 'text', 'text': ROLES.format(nctx=len(ctx_ims), nseg=len(pil))})
        convs.append([{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': content}])
        metas.append((vid, segs))

    if not convs:
        continue
    from vllm import LLM, SamplingParams
    if c0 == 0:
        llm = LLM(model=a.model, dtype='float16', tensor_parallel_size=a.tp,
                  gpu_memory_utilization=0.85, max_model_len=16384, max_num_seqs=16,
                  limit_mm_per_prompt={'image': 24}, enforce_eager=True)
        params = SamplingParams(temperature=0.0, max_tokens=400)
        print('ENGINE UP', f'{time.time() - t0:.0f}s', flush=True)
    res = llm.chat(convs, params, use_tqdm=False)
    for (vid, segs), o in zip(metas, res):
        roles, status = parse_roles(o.outputs[0].text, segs)
        if roles is None:
            roles, status = {}, status
        (OUT / f'{vid}.json').write_text(json.dumps(
            {'roles': {str(k): v for k, v in roles.items()}, 'status': status,
             'raw': o.outputs[0].text[:300]}))
        done += 1
        if done % 20 == 0:
            print(f'{done}/{len(todo)} sources {time.time() - t0:.0f}s '
                  f'({done / (time.time() - t0):.2f} src/s)', flush=True)
print(f'DONE {done} sources, err={err}, {time.time() - t0:.0f}s', flush=True)
