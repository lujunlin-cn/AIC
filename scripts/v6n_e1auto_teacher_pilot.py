"""V6-NEXT E1-AUTO: teacher pseudo-label pilot on native portrait video (910A).

User override 2026-09-29: no human annotation anywhere.  Reuse public labels
(PM-400 clip_context, context_only_not_crop_gt=true -> weak temporal channel
only) + local Qwen3-VL-32B subject points as crop preference / temporal
pseudo-labels.  20-40 video pilot first: throughput + retention before scale.
Contest test videos are never touched.

Stages (run inside aic-batch container, /root/AIC + /data/aic mounts):
  select   --n 32  train-split pilot list, clip_context first, duration spread
  frames            cv2 1 fps decode, evenly capped <=20 ff, JPEG long-side 1280
  teacher           batch eager fp16 4-NPU subject_point (prompt sha asserted)
  qc                parse -> legal windows (3:1 / 16:9 / 1:3) -> preference
                    ranks + keep weights + smoothness filter -> retention stats
"""
import argparse, collections, json, statistics, time
from pathlib import Path

import numpy as np

VIDEOS = Path('/data/aic/external_datasets/PM400/videos')
MANIFEST = Path('/data/aic/external_datasets/PM400/pilot/sampling_manifest_v1.json')
WORK = Path('/data/aic/experiments/V6N_E1A')
RATIOS = {'3-1': (3, 1), '16-9': (16, 9), '1-3': (1, 3)}
SMOOTH_TOL = 0.15          # px distance to the per-video median trajectory
MIN_PARSE_RATE = 0.8       # video kept only if >=80% frames parsed
MAX_UNCERTAIN_RATE = 0.5


def legal_window(W, H, ratio, px, py):
    """Ratio-exact window covering (px,py), clamped; returns [x,y,w,h] + slide axis."""
    rw, rh = ratio
    w = min(W, int(round(H * rw / rh)))
    h = int(round(w * rh / rw))
    if w > W:
        w, h = W, int(round(W * rh / rw))
    x = int(round(min(max(px * W - w / 2, 0), W - w)))
    y = int(round(min(max(py * H - h / 2, 0), H - h)))
    axis = 'x' if h >= H else 'y'
    return [x, y, w, h], axis


def stage_select(n):
    man = json.loads(MANIFEST.read_text())
    # extracted clips are named by clip_context.sample_id (int stems)
    train = {}
    for s in man['sources']:
        if s['pilot_split'] == 'train' and s.get('clip_context'):
            train[str(s['clip_context']['sample_id'])] = s
    have = sorted(p.stem for p in VIDEOS.glob('*.mp4'))
    pool = [v for v in have if v in train]
    with_ctx = pool  # every mapped clip has clip_context by construction
    with_ctx.sort(key=lambda v: train[v]['source_duration_s'])
    picked, stride = [], max(1, len(with_ctx) // n)
    for i in range(0, len(with_ctx), stride):
        picked.append(with_ctx[i])
        if len(picked) >= n:
            break
    rows = [{'pm_video_id': v, 'pilot_split': 'train',
             'source_duration_s': train[v]['source_duration_s'],
             'category_name': train[v].get('category_name'),
             'has_clip_context': True,
             'clip_context': train[v].get('clip_context')} for v in picked]
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / 'pilot_list.json').write_text(json.dumps(rows, indent=1) + '\n')
    print(json.dumps({'picked': len(picked), 'with_ctx': sum(r['has_clip_context'] for r in rows),
                      'pool': len(pool), 'train_manifest': len(train)}), flush=True)


def stage_frames(max_frames=20, long_side=1280):
    import cv2
    pilot = json.loads((WORK / 'pilot_list.json').read_text())
    meta_f = (WORK / 'frames_meta.jsonl').open('w')
    for rec in pilot:
        vid = rec['pm_video_id']
        out = WORK / 'frames' / vid
        if out.is_dir() and len(list(out.glob('*.jpg'))) >= 2:
            continue
        cap = cv2.VideoCapture(str(VIDEOS / f'{vid}.mp4'))
        if not cap.isOpened():
            print(json.dumps({'vid': vid, 'error': 'open_failed'}), flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        scale = min(1.0, long_side / max(W, H))
        idxs = sorted(set(int(round(i * (n - 1) / (max_frames - 1))) for i in range(max_frames))) if n > 1 else [0]
        kept = []
        want, wi = 0, 0
        fi = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if wi < len(idxs) and fi == idxs[wi]:
                t = fi / fps
                if scale < 1.0:
                    frame = cv2.resize(frame, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(out / f'{fi:05d}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                kept.append({'f': fi, 't': round(t, 3), 'file': f'{fi:05d}.jpg'})
                wi += 1
            fi += 1
        cap.release()
        meta_f.write(json.dumps({'pm_video_id': vid, 'W': int(W * scale), 'H': int(H * scale),
                                 'fps': round(fps, 3), 'n_src_frames': n,
                                 'frames': kept, 'scale': round(scale, 4)}) + '\n')
        meta_f.flush()
    meta_f.close()
    rows = [json.loads(l) for l in (WORK / 'frames_meta.jsonl').read_text().splitlines() if l.strip()]
    print(json.dumps({'videos': len(rows), 'frames_total': sum(len(r['frames']) for r in rows)}), flush=True)


def stage_teacher(batch=4, cards='2,3,4,5', canonical=(720, 1280)):
    import sys
    sys.path.insert(0, '/root/AIC')
    import torch, torch_npu  # noqa: F401
    from PIL import Image
    from scripts.qwen_subject_point import SYSTEM, USER, parse, prompt_sha
    assert prompt_sha('point') == '918f0730f5407c7186a3208a9f76662c3121c8ea25475655d8ce3f5dc8acb53d', 'point prompt changed'
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model_dir = '/data/aic/pretrained/qwen3_vl_32b_instruct'
    proc = AutoProcessor.from_pretrained(model_dir)
    proc.tokenizer.padding_side = 'left'
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.float16, device_map='auto',
        low_cpu_mem_usage=True, attn_implementation='eager').eval()
    print(json.dumps({'load_s': True, 'devices': sorted({str(p.device) for p in model.parameters()})}), flush=True)

    # one canonical (W,H) for every query: mixed portrait resolutions would each
    # trigger a fresh NPU operator compile (~minutes per shape).  Normalised
    # 0..1000 points are invariant to the uniform resize.
    W, H = canonical
    rw, rh = RATIOS['3-1']
    orient = 'portrait, narrower than the frame' if rw / rh < W / H else 'landscape, wider than the frame'
    text = USER.format(w=W, h=H, rw=rw, rh=rh, orient=orient)
    metas = {json.loads(l)['pm_video_id']: json.loads(l)
             for l in (WORK / 'frames_meta.jsonl').read_text().splitlines() if l.strip()}
    out_f = (WORK / 'teacher_replies.jsonl').open('a')
    done = set()
    if (WORK / 'teacher_replies.jsonl').exists():
        done = {(r['pm_video_id'], r['f']) for r in
                (json.loads(l) for l in (WORK / 'teacher_replies.jsonl').read_text().splitlines() if l.strip())}
    pool = []
    for vid, m in metas.items():
        for fr in m['frames']:
            if (vid, fr['f']) not in done:
                pool.append((vid, fr['f'], text, WORK / 'frames' / vid / fr['file']))
    n = (len(pool) + batch - 1) // batch * batch
    while len(pool) < n:          # pad to full batches: a short last batch is
        pool.append(pool[0])      # another shape -> another compile
    print(json.dumps({'pool': len(pool), 'already_done': len(done)}), flush=True)
    t_all = time.time(); n_done = 0
    for i in range(0, len(pool), batch):
        chunk = pool[i:i + batch]
        imgs = [Image.open(p).convert('RGB').resize((W, H), Image.BICUBIC) for _, _, _, p in chunk]
        msgs = [[{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]},
                 {'role': 'user', 'content': [{'type': 'image', 'image': im}, {'type': 'text', 'text': t}]}]
                for (_, _, t, _), im in zip(chunk, imgs)]
        inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                          return_dict=True, return_tensors='pt', padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        torch.npu.synchronize()
        replies = proc.batch_decode(out[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        for (vid, f, _, _), reply in zip(chunk, replies):
            if (vid, f) in done:
                continue    # padding duplicate, already answered earlier
            done.add((vid, f))
            p, st = parse(reply)
            out_f.write(json.dumps({'pm_video_id': vid, 'f': f, 'W': W, 'H': H,
                                    'raw': reply, 'status': st,
                                    'point': None if p is None else [p['x'], p['y']],
                                    'uncertain': None if p is None else p['uncertain']}) + '\n')
        out_f.flush()
        n_done += len(chunk)
        el = time.time() - t_all
        print(json.dumps({'done': n_done, 'of': len(pool), 'qps_running': round(n_done / el, 3),
                          'eta_min': round((len(pool) - n_done) / max(n_done / el, 1e-6) / 60, 1)}), flush=True)
    out_f.close()
    rows = [json.loads(l) for l in (WORK / 'teacher_replies.jsonl').read_text().splitlines() if l.strip()]
    print(json.dumps({'teacher_done': len(rows), 'wall_min': round((time.time() - t_all) / 60, 1)}), flush=True)


def stage_qc():
    metas = {json.loads(l)['pm_video_id']: json.loads(l)
             for l in (WORK / 'frames_meta.jsonl').read_text().splitlines() if l.strip()}
    replies = collections.defaultdict(dict)
    for l in (WORK / 'teacher_replies.jsonl').read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        replies[r['pm_video_id']][r['f']] = r
    pilot = {r['pm_video_id']: r for r in json.loads((WORK / 'pilot_list.json').read_text())}
    lab_f = (WORK / 'pseudo_labels.jsonl').open('w')
    stats = collections.Counter()
    per_video = []
    for vid, m in sorted(metas.items()):
        rr = replies.get(vid, {})
        W, H = m['W'], m['H']
        ts, xs, ys = [], [], []
        for fr in m['frames']:
            r = rr.get(fr['f'])
            if r and r['point'] is not None:
                ts.append(fr['t']); xs.append(r['point'][0]); ys.append(r['point'][1])
        stats['frames_total'] += len(m['frames'])
        stats['frames_parsed'] += len(ts)
        parse_rate = len(ts) / max(1, len(m['frames']))
        err_rate = 1.0
        if len(ts) >= 5:
            mx, my = statistics.median(xs), statistics.median(ys)
            d = np.hypot(np.array(xs) - mx, np.array(ys) - my)
            err_rate = float((d > SMOOTH_TOL).mean())
        uncertain_rate = (sum(1 for f in rr.values() if f.get('uncertain')) / len(ts)) if ts else 1.0
        keep_video = parse_rate >= MIN_PARSE_RATE and err_rate <= 0.3 and uncertain_rate <= MAX_UNCERTAIN_RATE
        ctx = pilot.get(vid, {}).get('clip_context')
        frames = []
        for fr in m['frames']:
            r = rr.get(fr['f'])
            if not r or r['point'] is None:
                frames.append({'t': fr['t'], 'status': r['status'] if r else 'missing',
                               'keep_weight': 0.0})
                continue
            px, py = r['point']
            wins = {}
            for rk, ratio in RATIOS.items():
                wins[rk], _ = legal_window(W, H, ratio, px, py)
            weight = 0.0 if r.get('uncertain') else 1.0
            if err_rate > 0.15:
                weight *= 0.5
            frames.append({'t': fr['t'], 'status': r['status'],
                           'point': [px, py], 'uncertain': bool(r.get('uncertain')),
                           'windows': wins, 'keep_weight': weight,
                           'pref': {'3-1': ['point', 'center', 'anti'],
                                    '16-9': ['point', 'center', 'anti'],
                                    '1-3': ['point', 'center', 'anti']}})
        rec = {'pm_video_id': vid, 'source': 'PM400:pilot_v1:train', 'label_kind': 'teacher_pseudo_v1',
               'W': W, 'H': H, 'frames': frames,
               'event_context': (dict(ctx, context_only_not_crop_gt=True) if ctx else None),
               'qc': {'parse_rate': round(parse_rate, 3), 'smooth_err_rate': round(err_rate, 3),
                      'uncertain_rate': round(uncertain_rate, 3), 'kept': keep_video}}
        lab_f.write(json.dumps(rec) + '\n')
        stats['videos'] += 1
        stats['videos_kept'] += int(keep_video)
        per_video.append({'vid': vid, **rec['qc']})
    lab_f.close()
    rep = {'frames_total': stats['frames_total'], 'frames_parsed': stats['frames_parsed'],
           'parse_rate': round(stats['frames_parsed'] / max(1, stats['frames_total']), 4),
           'videos': stats['videos'], 'videos_kept': stats['videos_kept'],
           'retention': round(stats['videos_kept'] / max(1, stats['videos']), 3),
           'per_video': per_video}
    (WORK / 'qc_report.json').write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps({k: v for k, v in rep.items() if k != 'per_video'}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=('select', 'frames', 'teacher', 'qc'))
    ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--max-frames', type=int, default=20)
    ap.add_argument('--batch', type=int, default=4)
    a = ap.parse_args()
    if a.stage == 'select':
        stage_select(a.n)
    elif a.stage == 'frames':
        stage_frames(a.max_frames)
    elif a.stage == 'teacher':
        stage_teacher(a.batch)
    else:
        stage_qc()


if __name__ == '__main__':
    main()
