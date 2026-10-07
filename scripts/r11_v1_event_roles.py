"""R11 V1: event roles for the PHD2 slot pool via generative Qwen2.5-VL-7B.

Why not scripts/qwen_event_segments.py (the V4-E3 chain): it is vLLM-based
(dead on this host family) and bound to the YTH anchors cache.  This script
keeps E3's task shape (2 s segments, MAIN/SETUP/RESULT/IDLE/JUNK/UNSURE,
JSON-only reply) but rebuilds the pipeline on the R9 PHD2 stack:
media locator, slot time axis from the R10 audio npz, transformers fp16
eager generate on NPU (kmsp05-proven), sharded incremental writes.

Prompt adaptation: no TEMP summary exists for PHD2 -> the summary line is
dropped; each frag contributes its 8 slot frames as the 8 segment images
(t is source-absolute, so segment ids are globally consistent and multiple
frags of one source merge into one <source_id>.json).

Output: <out>/<source_id>.json  {"roles": {"<seg>": "ROLE", ...}}
(resume-safe; atomic tmp+rename; same source may be touched by several
shards -- last writer wins per segment, roles are identical across frags
of the same segment because they read the same slot frame).
Run on the NPU venv: /usr/local/python3.11.15/bin/python3
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--model-path', default='/data/aic/pretrained/Qwen2.5-VL-7B-Instruct')
p.add_argument('--media-root', default='/data/aic/external_datasets/PHD2/raw/youtube')
p.add_argument('--audio-root', default='/data/aic/experiments_910a/LFM_V11/r10_audio')
p.add_argument('--manifest-dir', default='/data/aic/experiments_910a/LFM_V11/r8_npu/r9_audit')
p.add_argument('--out', default='/data/aic/experiments_910a/LFM_V11/r11_e3_roles')
p.add_argument('--shard', type=int, default=0)
p.add_argument('--nshards', type=int, default=1)
p.add_argument('--limit', type=int, default=0)
p.add_argument('--device', default='npu')
a = p.parse_args()

import cv2  # noqa: E402
import av  # noqa: E402
if a.device == 'npu':
    os.environ.setdefault('ASCEND_RT_VISIBLE_DEVICES', str(a.shard % 2))
    import torch_npu  # noqa: F401,E402
    # NOTE 2026-10-04 rebuild: DO NOT set_compile_mode(jit_compile=False) on
    # CANN 9.0.0 + torch_npu 2.9.0.post2 (kernel-parse bug) -- see
    # /data/aic/tools/ascend_teacher.env.note; run via `source ascend_teacher.env`.
    torch_npu.npu.config.allow_internal_format = False
    assert torch_npu.npu.device_count() > 0, 'no NPU visible - refusing CPU fallback'
import torch  # noqa: E402
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # noqa: E402

SYSTEM = ('You are a professional video editor cutting a short highlight clip from a video. '
          'You first understand the event the video is about, then decide which moments the highlight clip needs.')
ROLES = ('The images cover {lo:.1f}-{hi:.1f}s of this {dur:.1f}s video as consecutive 2-second segments; '
         'each segment is shown by one frame labelled [seg=K t=A-Bs]. '
         'For each segment choose its role in the event:\n'
         'MAIN = the main event or highlight action is happening, the moment a viewer is meant to watch;\n'
         'SETUP = build-up needed to understand or enjoy the main event (approach, run-up, the moment right before it);\n'
         'RESULT = the outcome or reaction right after the main event (landing, catch, score, celebration);\n'
         'IDLE = nothing of the event is happening: waiting, wandering, repeated or aimless footage with no progress;\n'
         'JUNK = black, blank or fading frames, title cards, credits, logos, or content unrelated to the main event;\n'
         'UNSURE = you cannot tell.\n'
         'Use UNSURE only when the frame is unreadable (black, blurred beyond recognition). '
         'If a segment shows any part of the main activity, choose SETUP, MAIN or RESULT instead of UNSURE. '
         'A quiet or static moment is not IDLE if it belongs to the event. '
         'A moment without people is not JUNK if it shows the event or its setting in use.\n'
         'Return ONLY JSON: {{"roles":{{"<segment number>":"ROLE"}}}} with one entry for '
         'every listed segment. Use the actual segment numbers from the [seg=...] labels '
         'as keys (for example: {{"roles":{{"40":"MAIN","41":"SETUP"}}}}).')
ALL_R = {'MAIN', 'SETUP', 'RESULT', 'IDLE', 'JUNK', 'UNSURE'}
_JSON = re.compile(r'\{.*\}', re.DOTALL)


def parse_roles(text, segs):
    m = _JSON.search(text or '')
    if not m:
        return {s: 'UNSURE' for s in segs}, 'no_json'
    try:
        d = json.loads(m.group(0))
    except ValueError:
        return {s: 'UNSURE' for s in segs}, 'bad_json'
    r = d.get('roles', d)
    if not isinstance(r, dict):
        return {s: 'UNSURE' for s in segs}, 'bad_roles'
    out = {}
    for s in segs:
        v = r.get(str(s), r.get(s))
        v = str(v).strip().upper() if v is not None else None
        out[s] = v if v in ALL_R else 'UNSURE'
    return out, 'ok'


model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    a.model_path, torch_dtype=torch.float16, attn_implementation='eager').to(a.device)
proc = AutoProcessor.from_pretrained(a.model_path, min_pixels=448 * 28 * 28, max_pixels=448 * 448)
model.eval()
print(f'{a.model_path} on {a.device}', flush=True)

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
mine = [s for i, s in enumerate(srcs) if i % a.nshards == a.shard]
if a.limit:
    mine = mine[:a.limit]
OUT = Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)
print(f'shard {a.shard}/{a.nshards}: {len(mine)} sources', flush=True)

t_of = {}
for f in Path(a.audio_root).glob('p*/*.npz'):
    try:
        t_of[f.stem] = np.load(f)['t'].astype(np.float32)
    except Exception:
        pass


def slot_frames(vid_path, t8):
    c = av.open(str(vid_path))
    st = c.streams.video[0]
    fps = float(st.average_rate)
    need = {}
    for si, t in enumerate(t8.tolist()):
        fr = max(int(round(t * fps)), 0)
        need.setdefault(fr, []).append(si)
    c.seek(max(int((min(need) - 2) / fps * av.time_base), 0), backward=True)
    imgs = {}
    for fr in c.decode(video=0):
        idx = int(round(fr.pts * fr.time_base * fps))
        if idx in need:
            imgs[idx] = cv2.resize(fr.to_ndarray(format='rgb24'), (448, 448),
                                   interpolation=cv2.INTER_CUBIC)
        if idx >= max(need):
            break
    c.close()
    out = []
    for t in t8.tolist():
        fr = max(int(round(t * fps)), 0)
        out.append(imgs.get(fr))
    return out


def atomic_json(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj))
    tmp.rename(path)


def source_thumbs(vid_path, n=2, size=160):
    """3 evenly spaced frames across the WHOLE source (context only)."""
    c = av.open(str(vid_path))
    st = c.streams.video[0]
    nfr = st.frames or int(float(st.average_rate) * float(st.duration or 0) / st.time_base)
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
            out.append(cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA))
            want.discard(idx)
            if not want:
                break
    c.close()
    return out


n_ok, n_err, t0 = 0, 0, time.time()
for vid in mine:
    outp = OUT / f'{vid}.json'
    if outp.exists():
        continue
    fids = sorted(by_src[vid], key=lambda f: t_of.get(f, np.zeros(8))[0])
    try:
        t8 = t_of[fids[0]]
        media = Path(a.media_root) / f'{vid}.mp4'
        if not media.exists() or fids[0] not in t_of:
            n_err += 1
            continue
        frames = slot_frames(media, t8)
        frames = [cv2.resize(im, (384, 384), interpolation=cv2.INTER_AREA) if im is not None else None
                  for im in frames]
        # source-level context: 3 thumbnails (start/mid/end of the WHOLE source)
        # so the model knows what the video is about -- the E3 chain had TEMP's
        # summary for this; we approximate it with frames.
        ctx_ims = source_thumbs(media)
        segs = [int(t // 2) for t in t8.tolist()]
        content = [{'type': 'text', 'text':
                    'First, 3 context frames from elsewhere in this video, so you know what it is about:'}]
        for im in ctx_ims:
            content.append({'type': 'image', 'image': im})
        content.append({'type': 'text', 'text':
                        f'Now {len(segs)} consecutive 2-second segments to label, each shown by one frame:'})
        for s, im in zip(segs, frames):
            content.append({'type': 'text', 'text': f'[seg={s} t={2 * s}-{2 * s + 2}s]'})
            if im is not None:
                content.append({'type': 'image', 'image': im})
        lo, hi = 2 * segs[0], 2 * segs[-1] + 2
        content.append({'type': 'text', 'text': ROLES.format(lo=lo, hi=hi, dur=float(t8[-1] + 2), ids=','.join(map(str, segs)))})
        conv = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': content}]
        text = proc.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        imgs = [c['image'] for c in content if c.get('type') == 'image']
        inputs = proc(text=[text], images=imgs, return_tensors='pt').to(a.device)
        with torch.inference_mode():
            try:
                out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            except Exception as e:  # noqa: BLE001
                if 'memory' not in str(e).lower():
                    raise
                # OOM fallback: drop the context thumbnails, retry once
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if hasattr(torch, 'npu'):
                    torch.npu.empty_cache()
                keep = [(c, im) for c, im in zip(content, imgs) if c.get('type') == 'image'][:len(frames) + 1]
                drop = {id(im) for c, im in zip(content, imgs) if c.get('type') == 'image'} - \
                       {id(im) for _, im in keep[1:]}
                content2 = [c for c in content if c.get('type') != 'image' or id(c.get('image')) not in drop]
                text2 = proc.apply_chat_template([conv[0], {'role': 'user', 'content': content2}],
                                                 tokenize=False, add_generation_prompt=True)
                inputs = proc(text=[text2], images=[c['image'] for c in content2 if c.get('type') == 'image'],
                              return_tensors='pt').to(a.device)
                out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        reply = proc.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        roles, status = parse_roles(reply, segs)
        prev = json.loads(outp.read_text())['roles'] if outp.exists() else {}
        prev.update({str(k): v for k, v in roles.items()})
        atomic_json(outp, {'roles': prev, 'status': status,
                           'raw': reply[:400], 'n_img': len(imgs)})
        n_ok += 1
        if n_ok % 10 == 0:
            print(f'{n_ok} sources {time.time() - t0:.0f}s', flush=True)
    except Exception as e:  # noqa: BLE001
        n_err += 1
        print(f'ERR {vid}: {str(e)[:160]}', flush=True)
print(f'DONE ok={n_ok} err={n_err} {time.time() - t0:.0f}s', flush=True)
