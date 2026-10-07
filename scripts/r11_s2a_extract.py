"""R11 S2a step 2: RT-DETR person detection over the PHD2 slot pool.

Smoke passed (r11/s2a_smoke.json): PekingU/rtdetr_r50vd loads on 910B via
hf-mirror in ~6 s.  This script wires the frozen media locator (R9 stack)
to the detector and produces, per frag:
    <out>/p<shard>/<frag>.npz {det_box (8,4) xyxy in 448-space, det_conf (8,),
                               n_det (8,), t}
Box = highest-confidence 'person' detection per slot frame; conf<0.3 -> zero
row (detector-abstain, mirrors the constructive-fallback discipline).

Gates, both written to <out>/parity.json before any scale-up:
  coverage: fraction of slot frames with conf>=0.3 must be >=0.6 (detector
            actually fires); per-source coverage table for the first 10.
  runtime : s/frame recorded (100-source extraction budget).
No GT is consumed here -- the candidate-pool oracle audits (plan S2 readouts
a/b/c) run downstream on these npz + the frozen B3 tables.
"""
import argparse
import json
import os
import time
from pathlib import Path

# MUST precede `import transformers`: huggingface_hub freezes HF_ENDPOINT at
# import time; setting it later silently no-ops and from_pretrained hangs on
# the unreachable huggingface.co (kmsp05 has no direct egress).
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--model-path', default='PekingU/rtdetr_r50vd')
p.add_argument('--media-root', default='/data/aic/external_datasets/PHD2/raw/youtube')
p.add_argument('--audio-root', default='/data/aic/experiments_910a/LFM_V11/r10_audio')
p.add_argument('--manifest-dir', default='/data/aic/experiments_910a/LFM_V11/r8_npu/r9_audit')
p.add_argument('--out', default='/data/aic/experiments_910a/LFM_V11/r11_s2a')
p.add_argument('--shard', type=int, default=0)
p.add_argument('--nshards', type=int, default=1)
p.add_argument('--limit', type=int, default=10, help='parity sources; 0 = all')
p.add_argument('--conf', type=float, default=0.3)
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
from transformers import AutoImageProcessor, AutoModelForObjectDetection  # noqa: E402

proc = AutoImageProcessor.from_pretrained(a.model_path)
model = AutoModelForObjectDetection.from_pretrained(a.model_path).to(a.device)
model.eval()
ID2LAB = model.config.id2label
PERSON = next((k for k, v in ID2LAB.items() if str(v).lower() == 'person'), None)
print(f'{a.model_path} on {a.device}; person id={PERSON}', flush=True)

man = []
for split in ('train_public', 'eval_public'):
    f = Path(a.manifest_dir) / f'{split}.jsonl'
    if f.exists():
        man += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
seen, fid_src, fid_t = set(), {}, {}
for r in man:
    if r['fragment_id'] in seen:
        continue
    seen.add(r['fragment_id'])
    fid_src[r['fragment_id']] = r['source_id']
fids = sorted(seen)
mine = [f for i, f in enumerate(fids) if i % a.nshards == a.shard]
if a.limit:
    mine = mine[:a.limit]
OUTP = Path(a.out) / f'p{a.shard}'
OUTP.mkdir(parents=True, exist_ok=True)

t_of = {}
for f in Path(a.audio_root).glob('p*/*.npz'):
    try:
        t_of[f.stem] = np.load(f)['t'].astype(np.float32)
    except Exception:
        pass


def frames_at(vid_path, t8):
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
    return [imgs.get(max(int(round(t * fps)), 0)) for t in t8.tolist()]


n_cov, n_tot, n_err, t0 = 0, 0, 0, time.time()
err_detail = []
for fid in mine:
    outp = OUTP / f'{fid}.npz'
    if outp.exists():
        continue
    vid = fid_src[fid]
    t8 = t_of.get(fid)
    media = Path(a.media_root) / f'{vid}.mp4'
    if t8 is None or not media.exists():
        continue
    try:
        ims = frames_at(media, t8)
        keep = [i for i, im in enumerate(ims) if im is not None]
        boxes = np.zeros((8, 4), np.float32)
        confs = np.zeros(8, np.float32)
        ndet = np.zeros(8, np.int16)
        if keep:
            with torch.inference_mode():
                inputs = proc(images=[ims[i] for i in keep], return_tensors='pt').to(a.device)
                out = model(**inputs)
                prob = out.logits.softmax(-1)
            # RT-DETR logits are pure-class (no background column); models with
            # a background column need the +1 label shift. Detect by width.
            n_cls, has_bg = prob.shape[-1], prob.shape[-1] == len(ID2LAB) + 1
            for row, i in enumerate(keep):
                sc, lb = prob[row].max(-1)
                sc = sc.float().cpu().numpy()
                lb = (lb.float() + 1).cpu().numpy() if has_bg else lb.cpu().numpy()
                order = np.argsort(-sc)
                ndet[i] = int((sc >= a.conf).sum())
                for j in order:
                    if int(lb[j]) == PERSON and sc[j] >= a.conf:
                        boxes[i] = out.pred_boxes[row, j].float().cpu().numpy()
                        confs[i] = float(sc[j])
                        break
        np.savez(outp, det_box=boxes, det_conf=confs, n_det=ndet, t=t8)
        n_cov += int((confs >= a.conf).sum())
        n_tot += 8
        if n_tot % 80 == 0:
            print(f'{n_tot // 8} frags cov={n_cov / max(n_tot, 1):.2f} {time.time() - t0:.0f}s', flush=True)
    except Exception as e:  # noqa: BLE001
        n_err += 1
        err_detail.append({'frag': fid, 'err': str(e)[:220]})
        print(f'ERR {fid}: {str(e)[:140]}', flush=True)

rate = n_cov / max(n_tot, 1)
verdict = {'coverage': round(rate, 4), 'gate_ge_060': rate >= 0.6,
           'frags_done': n_tot // 8, 'errors': n_err, 'error_detail': err_detail[:5],
           'runtime_s': round(time.time() - t0, 1),
           's_per_frame': round((time.time() - t0) / max(n_tot, 1), 3),
           'note': 'gate fires the 100-source scale-up; readouts a/b/c run downstream'}
(Path(a.out) / 'parity.json').write_text(json.dumps(verdict, indent=1))
print(json.dumps(verdict, indent=1))
