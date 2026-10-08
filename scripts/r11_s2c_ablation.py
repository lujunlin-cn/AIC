"""S2c ablation: does the model consume the inputs at all?"""
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
W = torch.load('/data/aic/tools/Light-ASD/weight/pretrain_AVA_CVPR.model',
               map_location='cpu', weights_only=False)
state = {k[len('model.'):]: v for k, v in W.items() if k.startswith('model.')}
model.load_state_dict(state)
model.eval()
fc = torch.nn.Linear(128, 2)
fc.load_state_dict({'weight': W['lossAV.FC.weight'], 'bias': W['lossAV.FC.bias']})
fc.eval()
det = cv2.FaceDetectorYN_create('/data/aic/tools/yunet.onnx', '', (320, 320), score_threshold=0.35)

media = '/data/aic/external_datasets/PHD2/raw/youtube/-6p1cTfcWEo.mp4'
t_mid = 60.0
ca = av.open(media)
resampler = av.AudioResampler(format='s16', rate=16000, layout='mono')
chunks = []
for fr in ca.decode(audio=0):
    chunks.extend(resampler.resample(fr))
pcm = np.concatenate([k.to_ndarray().reshape(-1) for k in chunks]).astype(np.float32) / 32768.0
ca.close()

c = av.open(media)
c.seek(max(int((t_mid - 0.4) / av.time_base), 0), backward=True)
got = None
for fr in c.decode(video=0):
    if float(fr.pts * fr.time_base) >= t_mid:
        got = fr.to_ndarray(format='bgr24')
        break
det.setInputSize((got.shape[1], got.shape[0]))
_, faces = det.detect(got)
face = faces[np.argmax(faces[:, 2] * faces[:, 3])]
x, y, w, h = face[:4]
ex = 0.15 * max(w, h)
box = (x - ex, y - ex, w + 2 * ex, h + 2 * ex)
print('face box', np.round(face[:4]).astype(int), 'conf', round(float(face[14]), 2))

c.seek(max(int((t_mid - 2.2) / av.time_base), 0), backward=True)
frames = []
for fr in c.decode(video=0):
    ts = float(fr.pts * fr.time_base)
    if ts < t_mid - 2.0:
        continue
    if ts > t_mid + 2.0:
        break
    im = cv2.cvtColor(fr.to_ndarray(format='bgr24'), cv2.COLOR_BGR2GRAY)
    H, Wd = im.shape
    xa, xb = max(int(box[0]), 0), min(int(box[0] + box[2]), Wd)
    ya, yb = max(int(box[1]), 0), min(int(box[1] + box[3]), H)
    if xb - xa < 8 or yb - ya < 8:
        continue
    frames.append(cv2.resize(im[ya:yb, xa:xb], (112, 112)))
idx = np.linspace(0, len(frames) - 1, 100).astype(int)
vis = np.stack([frames[i] for i in idx])
seg = pcm[int((t_mid - 2.0) * 16000):int((t_mid + 2.0) * 16000)]
aud = psf_mfcc(seg, 16000, numcep=13, winlen=0.025, winstep=0.010)[:400]
print('vis', vis.shape, 'aud', aud.shape)


def run(v, a, tag):
    ae = model.forward_audio_frontend(torch.tensor(a[None], dtype=torch.float32))
    ve = model.forward_visual_frontend(torch.tensor(v[None], dtype=torch.float32))
    out = model.forward_audio_visual_backend(ae, ve)
    s = fc(out).squeeze(1)[:, 1].numpy()
    print(f'{tag}: logit[min={s.min():.3f} max={s.max():.3f} std={s.std():.4f}]')


run(vis, aud, 'real/real    ')
run(vis, np.zeros_like(aud), 'real/zero-aud')
run(vis[::-1].copy(), aud[::-1].copy(), 'reversed     ')
run(np.zeros_like(vis), np.zeros_like(aud), 'zero/zero    ')
