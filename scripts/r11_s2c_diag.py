"""S2c one-slot diagnostic: input stats + logit spread across window lengths."""
import sys

sys.path.insert(0, '/data/aic/tools/Light-ASD')
import numpy as np
import av
import cv2
import torch
from python_speech_features import mfcc as psf_mfcc
from model.Model import ASD_Model

torch.set_grad_enabled(False)
model = ASD_Model()
state = torch.load('/data/aic/tools/Light-ASD/weight/pretrain_AVA_CVPR.model',
                   map_location='cpu', weights_only=False)
state = {k[len('model.'):]: v for k, v in state.items() if k.startswith('model.')}
model.load_state_dict(state)
model.eval()
det = cv2.FaceDetectorYN_create('/data/aic/tools/yunet.onnx', '', (320, 320), score_threshold=0.35)
media = '/data/aic/external_datasets/PHD2/raw/youtube/--uhWLthgsU.mp4'
t_mid = 82.77

ca = av.open(media)
resampler = av.AudioResampler(format='s16', rate=16000, layout='mono')
chunks = []
for fr in ca.decode(audio=0):
    chunks.extend(resampler.resample(fr))
pcm = np.concatenate([k.to_ndarray().reshape(-1) for k in chunks]).astype(np.float32) / 32768.0
ca.close()
print('pcm', pcm.shape, 'dur', round(pcm.shape[0] / 16000, 1))

c = av.open(media)
c.seek(max(int((t_mid - 0.4) / av.time_base), 0), backward=True)
got = None
for fr in c.decode(video=0):
    if float(fr.pts * fr.time_base) >= t_mid:
        got = fr.to_ndarray(format='bgr24')
        break
det.setInputSize((got.shape[1], got.shape[0]))
_, faces = det.detect(got)
face = faces[0]
x, y, w, h = face[0], face[1], face[2], face[3]
ex = 0.15 * max(w, h)
box = (x - ex, y - ex, w + 2 * ex, h + 2 * ex)
print('face conf', round(float(face[14]), 2), 'box', np.round(face[:4]).astype(int))

# continuous frames over a 4 s window around t_mid
c.seek(max(int((t_mid - 2.2) / av.time_base), 0), backward=True)
frames = []
for fr in c.decode(video=0):
    ts = float(fr.pts * fr.time_base)
    if ts < t_mid - 2.0:
        continue
    if ts > t_mid + 2.0:
        break
    im = cv2.cvtColor(fr.to_ndarray(format='bgr24'), cv2.COLOR_BGR2GRAY)
    H, W = im.shape
    xa, xb = max(int(box[0]), 0), min(int(box[0] + box[2]), W)
    ya, yb = max(int(box[1]), 0), min(int(box[1] + box[3]), H)
    if xb - xa < 8 or yb - ya < 8:
        continue
    frames.append(cv2.resize(im[ya:yb, xa:xb], (112, 112)))
print('window frames:', len(frames))

seg = pcm[int((t_mid - 2.0) * 16000):int((t_mid + 2.0) * 16000)]
fps_vis = 30.0  # this source
aud = psf_mfcc(seg, 16000, numcep=13, winlen=0.025 * 25 / fps_vis, winstep=0.010 * 25 / fps_vis)
print('mfcc:', aud.shape)

for T in (25, 50, 100):
    if len(frames) < T:
        continue
    idx = np.linspace(0, len(frames) - 1, T).astype(int)
    vis = np.stack([frames[i] for i in idx])
    Ta = min(aud.shape[0], T * 4)
    ae = model.forward_audio_frontend(torch.tensor(aud[:Ta][None], dtype=torch.float32))
    ve = model.forward_visual_frontend(torch.tensor(vis[None], dtype=torch.float32))
    out = model.forward_audio_visual_backend(ae, ve)
    lo = out.squeeze(1)[:, 1].numpy()
    print(f'T={T}: vis.mean={vis.mean():.1f} aud[:{Ta}].std={aud[:Ta].std():.3f} '
          f'logit[min={lo.min():.2f} max={lo.max():.2f} mean={lo.mean():.3f} std={lo.std():.3f}]')
