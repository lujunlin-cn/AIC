"""R11 S2c: Light-ASD active-speaker scoring over the PHD2 slot pool.

Per 2 s slot: detect the dominant face on the slot midpoint (YuNet), crop
1 s of continuous frames inside that box (25 frames, gray 112), take the
matching 1 s of 16 kHz audio (MFCC 13, winstep 10 ms -> 100 frames = 25
after the audio encoder's 4x temporal stride), run Light-ASD (CVPR23,
AVA pretrain) and record the raw speaker logit.  Writes the R10 readout
contract: <out>/<frag>.npz with s2c_asd (8, 3) = [asd_logit, n_faces,
face_conf] plus avalid.  CPU only, sharded, resume-safe.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--repo', default='/data/aic/tools/Light-ASD')
p.add_argument('--weight', default='/data/aic/tools/Light-ASD/weight/pretrain_AVA_CVPR.model')
p.add_argument('--yunet', default='/data/aic/tools/yunet.onnx')
p.add_argument('--media-root', default='/data/aic/external_datasets/PHD2/raw/youtube')
p.add_argument('--manifest-dir', default='/data/aic/experiments_910a/LFM_V11/r8_npu/r9_audit')
p.add_argument('--audio-root', default='/data/aic/experiments_910a/LFM_V11/r10_audio')
p.add_argument('--out', default='/data/aic/experiments_910a/LFM_V11/r11_s2c_asd')
p.add_argument('--shard', type=int, default=0)
p.add_argument('--nshards', type=int, default=8)
p.add_argument('--limit', type=int, default=0)
a = p.parse_args()

sys.path.insert(0, a.repo)
os.chdir(a.repo)  # S3FD-style relative paths inside the repo are unused, but keep cwd sane
import numpy as np
import av
import cv2
import torch
from python_speech_features import mfcc as psf_mfcc
from model.Model import ASD_Model

torch.set_grad_enabled(False)
model = ASD_Model()
state = torch.load(a.weight, map_location='cpu', weights_only=False)
state = {k[len('model.'):]: v for k, v in state.items() if k.startswith('model.')}
model.load_state_dict(state)
model.eval()
# the real score head lives in the checkpoint's lossAV.FC (Linear 128->2)
score_fc = torch.nn.Linear(128, 2)
score_fc.load_state_dict({'weight': torch.load(a.weight, map_location='cpu', weights_only=False)['lossAV.FC.weight'],
                          'bias': torch.load(a.weight, map_location='cpu', weights_only=False)['lossAV.FC.bias']})
score_fc.eval()
det = cv2.FaceDetectorYN_create(a.yunet, '', (320, 320), score_threshold=0.35)
print('model + yunet loaded', flush=True)

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
todo = [s for s in mine if not (OUT / f'{s}.json.npz').exists() and not (OUT / f'{s}.npz').exists()]
print(f'{len(mine)} sources, {len(todo)} to run', flush=True)

t_of = {}
for f in Path(a.audio_root).glob('p*/*.npz'):
    try:
        t_of[f.stem] = np.load(f)['t'].astype(np.float32)
    except Exception:
        pass


def decode_segment(vid_path, t0, t1, box, want=100):
    """Continuous frames in [t0, t1] cropped to box (xywh, expanded), gray 112."""
    c = av.open(str(vid_path))
    st = c.streams.video[0]
    fps = float(st.average_rate) or 25.0
    c.seek(max(int((t0 - 0.3) / av.time_base), 0), backward=True)
    x, y, w, h = box
    frames = []
    for fr in c.decode(video=0):
        ts = float(fr.pts * fr.time_base)
        if ts < t0:
            continue
        if ts > t1:
            break
        im = cv2.cvtColor(fr.to_ndarray(format='bgr24'), cv2.COLOR_BGR2GRAY)
        H, W = im.shape
        xa, xb = max(int(x), 0), min(int(x + w), W)
        ya, yb = max(int(y), 0), min(int(y + h), H)
        if xb - xa < 8 or yb - ya < 8:
            continue
        crop = cv2.resize(im[ya:yb, xa:xb], (112, 112))
        frames.append(crop)
    c.close()
    if not frames:
        return None
    if len(frames) > want:  # uniform downsample to 25
        idx = np.linspace(0, len(frames) - 1, want).astype(int)
        frames = [frames[i] for i in idx]
    return np.stack(frames)


def slot_audio(full_audio, t0, t1):
    if full_audio is None:
        return None
    seg = full_audio[int(t0 * 16000):int(t1 * 16000)]
    if seg.size < 8000:
        return None
    return psf_mfcc(seg, 16000, numcep=13, winlen=0.025, winstep=0.010)


def face_on_frame(im, det_net):
    H, W = im.shape[:2]
    det_net.setInputSize((W, H))
    _, faces = det_net.detect(im)
    if faces is None or len(faces) == 0:
        return None
    # largest face wins (tiny far-field faces carry no mouth signal)
    areas = faces[:, 2] * faces[:, 3]
    faces = faces[np.argsort(areas)[::-1]]
    if faces[0, 2] < 24 or faces[0, 3] < 24:
        return None
    return faces[0]  # x,y,w,h,...,conf(14)


n_ok = n_noface = 0
t_start = time.time()
for vi, vid in enumerate(todo):
    outp = OUT / f'{vid}.npz'
    fids = by_src[vid]
    if fids[0] not in t_of:
        continue
    t8 = t_of[fids[0]]
    media = Path(a.media_root) / f'{vid}.mp4'
    if not media.exists():
        continue
    feat = np.zeros((8, 3), np.float32)
    try:
        # whole audio track once, in its own container (av seek is demuxer-global)
        full_audio = None
        try:
            ca = av.open(str(media))
            if len(ca.streams.audio):
                resampler = av.AudioResampler(format='s16', rate=16000, layout='mono')
                chunks = []
                for fr in ca.decode(audio=0):
                    chunks.extend(resampler.resample(fr))
                if chunks:
                    full_audio = np.concatenate(
                        [k.to_ndarray().reshape(-1) for k in chunks]).astype(np.float32) / 32768.0
            ca.close()
        except Exception:  # noqa: BLE001
            full_audio = None
        c = av.open(str(media))
        for si in range(8):
            t_mid = float(t8[si]) + 1.0
            face = None
            mid = None
            for dts in (0.0, 0.25, -0.25, 0.5, -0.5):  # black/empty frames are common right after seek
                tt = t_mid + dts
                c.seek(max(int((tt - 0.4) / av.time_base), 0), backward=True)
                got = None
                for fr in c.decode(video=0):
                    ts = float(fr.pts * fr.time_base)
                    if ts >= tt:
                        got = fr.to_ndarray(format='bgr24')
                        break
                if got is None or float(got.mean()) < 5:  # black frame guard
                    continue
                face = face_on_frame(got, det)
                if face is not None:
                    mid = got
                    break
            if face is None:
                n_noface += 1
                continue
            x, y, w, h = face[0], face[1], face[2], face[3]
            conf = float(face[14])
            ex = 0.15 * max(w, h)
            box = (x - ex, y - ex, w + 2 * ex, h + 2 * ex)
            vis = decode_segment(media, t_mid - 2.0, t_mid + 2.0, box, want=100)
            if vis is None:
                continue
            aud = slot_audio(full_audio, t_mid - 2.0, t_mid + 2.0)
            if aud is None:
                continue
            Ta = min(aud.shape[0], 400)
            a_in = torch.tensor(aud[:Ta][None], dtype=torch.float32)
            v_in = torch.tensor(vis[None], dtype=torch.float32)
            ae = model.forward_audio_frontend(a_in)
            ve = model.forward_visual_frontend(v_in)
            out = model.forward_audio_visual_backend(ae, ve)
            logit = float(score_fc(out).squeeze(1)[:, 1].mean())
            feat[si] = (logit, 1.0, conf)
        c.close()
    except Exception as e:  # noqa: BLE001
        print(f'ERR {vid}: {str(e)[:120]}', flush=True)
        continue
    tmp = OUT / f'{vid}.tmp.npz'
    np.savez(tmp, s2c_asd=feat, avalid=np.ones(8, np.float32), t=t8)
    tmp.rename(outp)
    n_ok += 1
    if n_ok % 25 == 0:
        el = __import__('time').time() - t_start
        print(f'{n_ok}/{len(todo)} noface={n_noface} {el:.0f}s ({n_ok / el:.2f} src/s)', flush=True)
print(f'DONE {n_ok} noface={n_noface}', flush=True)
