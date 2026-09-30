"""V6-NEXT E3: raw MP4 -> YouTube-8M-compatible features -> frozen temporal head.

Official contract (fetched 2026-09-29 from google/youtube-8m feature_extractor
+ extract_tfrecords_main.py):
  RGB   frames at 1 fps -> InceptionV3 GraphDef (classify_image_graph_def.pb,
        2015-12-05) -> pool_3/_reshape 2048-d -> PCA (mean.npy / eigenvals.npy
        / eigenvecs.npy from yt8m_pca.tgz; whiten sqrt(val+1e-4)) -> 1024-d
        -> official quantize clip[-2,2] (f+2)*255/4 -> uint8
  audio VGGish embeddings (torchaudio official AudioSet weights), 0.96 s
        windows centered on each 1 fps frame tick -> 128-d -> same quantize
  model input  frozen release dequant v1 (q+0.5)*2/255-1 -- affine-equivalent
        to the official grid (absorbed by the head's first Linear), applied so
        cached-feature-trained heads stay in-distribution.

Stages:
  selftest    graph/PCA/VGGish load + feature sanity on one frame
  extract     --videos list-of-mp4 --out dir -> <video_stem>.npz
  head-eval   --features dir --weights t0/t1 pt -> per-second scores npz +
              summary (transfer evidence for E4; label mapping done downstream)
"""
import argparse, json, math, subprocess, time
from pathlib import Path

import numpy as np
import torch

TOOLS = Path('/data/aic/tools/yt8m_extract')
FEATS = 1024 + 128


def quantize_official(f):
    q = np.clip(f, -2.0, 2.0)
    return np.round((q + 2.0) * (255.0 / 4.0)).astype(np.uint8)


def dequant_v1(q):
    return ((q.astype('float32') + 0.5) * (2.0 / 255.0) - 1.0)


class InceptionExtractor:
    """TF-CPU InceptionV3 GraphDef + YouTube-8M PCA (official math)."""

    def __init__(self, model_dir):
        import tensorflow as tf
        self.tf = tf
        pb = model_dir / 'classify_image_graph_def.pb'
        graph_def = tf.compat.v1.GraphDef.FromString(pb.read_bytes())
        g = tf.compat.v1.Graph()
        with g.as_default():
            tf.compat.v1.import_graph_def(graph_def, name='')
            self.sess = tf.compat.v1.Session(graph=g)
        self.inp = self.sess.graph.get_tensor_by_name('DecodeJpeg:0')
        self.pool = self.sess.graph.get_tensor_by_name('pool_3/_reshape:0')
        self.pca_mean = np.load(model_dir / 'mean.npy')[:, 0]
        self.pca_vals = np.load(model_dir / 'eigenvals.npy')[:1024, 0]
        self.pca_vecs = np.load(model_dir / 'eigenvecs.npy').T[:, :1024]

    def frame_feature(self, rgb_u8):
        pool = self.sess.run(self.pool, {self.inp: rgb_u8})[0]
        f = (pool - self.pca_mean).reshape(1, 2048) @ self.pca_vecs
        return (f / np.sqrt(self.pca_vals + 1e-4)).reshape(1024)


class VGGishAudio:
    """torchvggish (official-converted AudioSet weights) + official PCA quantize.

    forward() with preprocess+postprocess returns the SAME quantization family
    as YouTube-8M stores (PCA whitening -> clip[-2,2] -> 8-bit 0..255); values
    are returned as a float tensor over the uint8 grid, cast on receipt.
    """

    SR = 16000
    WIN = 0.96

    def __init__(self, device='cpu'):
        import sys
        sys.path.insert(0, str(TOOLS / 'pydeps'))
        sys.path.insert(0, str(TOOLS))
        from tvpkg.vggish import VGGish as TVV
        # pca as an upstream-style torch dict (npz would go through torch.load)
        self.model = TVV({'vggish': (TOOLS / 'vggish-10086976.pth').as_uri(),
                          'pca': (TOOLS / 'vggish_pca_torch/vggish_pca_params-97013be8.pth').as_uri()},
                         device=torch.device(device), pretrained=True,
                         preprocess=True, postprocess=True)
        self.model.eval()

    def embed_windows(self, wav_16k, n_steps):
        """One 0.96 s VGGish window centred on each 1 s frame tick.

        Fed as 1.0 s of audio: 0.96 s yields 94 mel frames, 4 short of the
        96-frame example window, and the vendored frame() does not pad the
        tail (the extra 2 frames fall in the 40 ms margin, official-parity).
        """
        out = np.zeros((n_steps, 128), np.float32)
        have = np.zeros(n_steps, bool)
        for i in range(n_steps):
            c = int(i * self.SR + self.SR // 2)
            a = max(0, c - self.SR // 2)
            b = a + self.SR
            if b > len(wav_16k):
                b = len(wav_16k); a = max(0, b - self.SR)
            if b - a < self.SR // 4:
                continue
            with torch.inference_mode():
                e = self.model(wav_16k[a:b], self.SR)  # postprocessed, uint8 grid
            out[i] = e.reshape(-1).cpu().numpy().astype(np.float32)[:128]
            have[i] = True
        return out, have


def probe_streams(path):
    p = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
                        'stream=codec_type,duration', '-of', 'json', str(path)],
                       capture_output=True, text=True, timeout=120)
    info = {'video': None, 'audio': None}
    try:
        for s in json.loads(p.stdout).get('streams', []):
            k = s.get('codec_type')
            if k in info and info[k] is None:
                info[k] = float(s.get('duration') or 0) or None
    except Exception:
        pass
    return info


def read_frames_1fps(path, max_frames=300):  # official frame_iterator caps at 300
    """Decode 1 fps RGB frames via ffmpeg rawvideo; returns list of uint8 HWC."""
    p = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-vf', 'fps=1',
                        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                       capture_output=True, timeout=600)
    buf = p.stdout
    frames = []
    W = H = None
    # frame size unknown here; use ffprobe width/height
    q = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                        '-show_entries', 'stream=width,height', '-of', 'csv=p=0', str(path)],
                       capture_output=True, text=True, timeout=120)
    W, H = (int(x) for x in q.stdout.strip().split(','))
    step = W * H * 3
    for off in range(0, len(buf) - step + 1, step):
        a = np.frombuffer(buf[off:off + step], np.uint8).reshape(H, W, 3)
        frames.append(a)
        if len(frames) >= max_frames:
            break
    return frames, (W, H)


def read_audio_16k(path):
    p = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-vn',
                        '-ac', '1', '-ar', '16000', '-f', 's16le', '-'],
                       capture_output=True, timeout=600)
    if not p.stdout:
        return None
    return np.frombuffer(p.stdout, np.int16)


def stage_extract(videos, out_dir, device='cpu', fps_cap=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    import tensorflow as tf  # noqa: F401  (fail fast before per-video work)
    inc = InceptionExtractor(TOOLS)
    vgg = VGGishAudio(device)
    for vid_path in videos:
        vid = Path(vid_path); dest = out_dir / f'{vid.stem}.npz'
        if dest.exists():
            continue
        t0 = time.time()
        info = probe_streams(vid)
        frames, (W, H) = read_frames_1fps(vid)
        T = len(frames)
        rgb_q = np.zeros((T, 1024), np.uint8)
        for i, fr in enumerate(frames):
            rgb_q[i] = quantize_official(inc.frame_feature(fr))
        wav = read_audio_16k(vid) if info['audio'] else None
        if wav is not None and len(wav) > 16000 // 4:
            aud_q, have = vgg.embed_windows(wav, T)  # already on the uint8 grid
            aud_q = np.clip(np.round(aud_q), 0, 255).astype(np.uint8)
        else:
            aud_q, have = np.zeros((T, 128), np.uint8), np.zeros(T, bool)
        np.savez(dest, rgb=rgb_q, audio=aud_q, audio_present=have,
                 steps=T, width=W, height=H,
                 video_duration_s=info['video'],
                 wall_seconds=round(time.time() - t0, 1),
                 contract='yt8m_official_quantize_v1; feed=dequant_v1')
        print(json.dumps({'video': vid.stem, 'steps': T, 'audio_present_frac': round(float(have.mean()), 3),
                          'seconds': round(time.time() - t0, 1)}), flush=True)


def stage_head_eval(feat_dir, weights, out_dir, device='cuda:0'):
    import sys
    sys.path.insert(0, '/home/supie/AIC')
    from scripts.v6n_e4_temporal import build_model, dequant_v1  # noqa: E402
    ck = torch.load(weights, map_location='cpu', weights_only=False)
    model = build_model(ck['config']).to(device).eval()
    model.load_state_dict(ck['state_dict'])
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for f in sorted(feat_dir.glob('*.npz')):
        d = np.load(f)
        feats = np.concatenate([dequant_v1(d['rgb']), dequant_v1(d['audio'])], 1)
        x = torch.from_numpy(feats).unsqueeze(0).to(device)
        with torch.inference_mode():
            s = model(x).squeeze(0).cpu().numpy()
        np.savez(out_dir / f.name.replace('.npz', '_scores.npz'), scores=s,
                 audio_present=d['audio_present'], steps=d['steps'])
        summary.append({'video': f.stem, 'steps': int(d['steps']),
                        'score_mean': float(s.mean()), 'score_std': float(s.std())})
    (out_dir / 'head_eval_summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps({'scored': len(summary)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=('selftest', 'extract', 'head-eval'))
    ap.add_argument('--videos', nargs='*')
    ap.add_argument('--list-file')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--features', type=Path)
    ap.add_argument('--weights', type=Path)
    ap.add_argument('--device', default='cpu')
    a = ap.parse_args()
    if a.stage == 'selftest':
        import tensorflow as tf
        inc = InceptionExtractor(TOOLS)
        rng = np.random.default_rng(0)
        f = inc.frame_feature(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
        q = quantize_official(f)
        assert q.dtype == np.uint8 and q.shape == (1024,)
        vgg = VGGishAudio(a.device)
        e, have = vgg.embed_windows(np.random.default_rng(1).integers(-5000, 5000, 16000 * 3, dtype=np.int16), 3)
        assert e.shape == (3, 128) and bool(have.all())
        print(json.dumps({'selftest': 'OK', 'rgb_q_range': [int(q.min()), int(q.max())],
                          'audio_embed_absmean': float(np.abs(e).mean())}))
    elif a.stage == 'extract':
        vids = list(a.videos or [])
        if a.list_file:
            vids += [l.strip() for l in Path(a.list_file).read_text().splitlines() if l.strip()]
        stage_extract(vids, a.out, a.device)
    else:
        stage_head_eval(a.features, a.weights, a.out, a.device)


if __name__ == '__main__':
    main()
