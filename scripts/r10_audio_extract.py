"""R10 audio line: AST (AudioSet-finetuned) embeddings for the PHD2 slot pool.

Family A (learned audio) of the R10 census; checkpoint = MIT/ast-finetuned-
audioset-10-10-0.4593 (transformers-native, AudioSet mAP 0.459; PANNs GitHub
release is unreachable from this network - the swap stays inside the R10
pre-registered encoder table, AST was R2's encoder).

Media protocol: VERBATIM from r9_iv2_slot_feats.py - the frozen pool 't' axis
is absolute seconds inside the source video, PyAV seek to 2 s before the first
needed sample, sequential decode.  Audio stream instead of video: resample to
16 kHz mono s16, per slot take [t-1, t+1] (2 s window), AST fbank auto-pads
to 1024 frames (fixed protocol: every window forwarded independently).

Readouts per slot (pre-registered, <=2 kept later by the head stage):
  ast_pool  pooler_output (768)   - the checkpoint's own classification path
  ast_mean  mean(non-CLS tokens)  - norm-free readout
  ast_527   527 AudioSet class logits (coverage statistics: speech/music/
            crowd/cheering - doubles as the R10 4.4 soft-coverage audit)
Failure handling (R10 Q3.4): missing stream / undecodable window is NEVER
padded silently - zeros + 'avalid' flags + 'nshort' count; the head stage
falls back to prior on avalid=0 slots.  DOMAIN FACT (official 174-video audit,
user 2026-10-07): the official test set is 77% with a real audio track / 23%
with NO audio stream at all, while this pool is 96.7% - the availability gap
is handled by the CONSTRUCTIVE fallback in the head (residual gated to zero
on avalid=0), never by learning it from the ~3% missing slots here.

Sharded by frag index, incremental npz per frag (resume-safe), CPU only.
"""
import argparse, json, os, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '8')
from pathlib import Path
import numpy as np
import torch
import av

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='/data/aic/pretrained/ast_audioset')
ap.add_argument('--media-root', default='/data/aic/external_datasets/PHD2/raw/youtube')
ap.add_argument('--pool-feats', default='/data/aic/experiments_910a/LFM_V10/pool_feats')
ap.add_argument('--manifest-dir', default='/data/aic/experiments_910a/LFM_V11/r8_npu/r9_audit')
ap.add_argument('--sources', default='', help='optional file with the pilot source list')
ap.add_argument('--out', default='/data/aic/experiments_910a/LFM_V11/r10_audio')
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshards', type=int, default=1)
ap.add_argument('--limit', type=int, default=0)
a = ap.parse_args()

from transformers import ASTModel, ASTForAudioClassification
fe = __import__('transformers').ASTFeatureExtractor.from_pretrained(a.ckpt)
m = ASTModel.from_pretrained(a.ckpt).eval()
full = ASTForAudioClassification.from_pretrained(a.ckpt).eval()
w_head = full.classifier
print('AST loaded', flush=True)

man = []
for split in ('train_public', 'eval_public'):
    for l in (Path(a.manifest_dir) / f'{split}.jsonl').read_text().splitlines():
        if l.strip():
            man.append(json.loads(l))
srcs = {r['fragment_id']: r['source_id'] for r in man}
if a.sources:
    keep = set(Path(a.sources).read_text().split())
    man = [r for r in man if r['source_id'] in keep]
fids = [r['fragment_id'] for r in man]
mine = [f for i, f in enumerate(fids) if i % a.nshards == a.shard]
if a.limit:
    mine = mine[:a.limit]
OUTP = Path(a.out) / f'p{a.shard}'
OUTP.mkdir(parents=True, exist_ok=True)
print(f'shard {a.shard}/{a.nshards}: {len(mine)} frags', flush=True)

MEDIA = Path(a.media_root)
POOL = Path(a.pool_feats)
SR = 16000


def decode_audio(vid_path, t8):
    """Per-slot 2 s windows as float32 arrays; returns (wins, avalid, has_audio).

    Seeking and per-sample indexing follow the R9 video rule: absolute seconds
    from the frozen 't' axis, seek 2 s early, decode sequentially, index by
    pts*time_base.  Short windows at source edges are zero-padded and counted.
    """
    c = av.open(str(vid_path))
    try:
        asts = c.streams.audio
        if not asts:
            return (np.zeros((8, SR*2), np.float32), np.zeros(8, bool), 0)
        st = asts[0]
        res = av.AudioResampler(format='s16', layout='mono', rate=SR)
        t_min, t_max = float(t8.min()) - 1.0, float(t8.max()) + 1.0
        # NO stream= kwarg: Container.seek interprets the timestamp in the
        # given stream's time_base when stream is passed (microseconds would
        # be read as 1/44100 s - smoke probe: 25.7 s request landed at
        # 582.75 s).  Bare seek uses av.time_base, same as the R9 video path.
        c.seek(max(int((t_min - 2.0) * av.time_base), 0), backward=True)
        chunks = []          # (start_sample, np.int16 mono)
        for fr in c.decode(audio=0):
            ts = fr.pts * fr.time_base if fr.pts is not None else None
            for r in res.resample(fr):
                # to_ndarray is (planes, samples) e.g. (1, 355); a mean over
                # the wrong axis collapses 355 samples to 1 - the resampler
                # already mixes to mono, so flat-view is always correct.
                arr = r.to_ndarray().reshape(-1)
                if arr.dtype != np.int16:
                    arr = arr.astype(np.int16)
                start = int(round((r.pts * r.time_base) * SR)) if r.pts is not None else 0
                chunks.append((start, arr))
            if ts is not None and ts > t_max + 2.0:
                break
    finally:
        c.close()
    if not chunks:
        return (np.zeros((8, SR*2), np.float32), np.zeros(8, bool), 1)
    # merge into a sparse sample axis
    s0 = min(s for s, _ in chunks)
    s1 = max(s + len(x) for s, x in chunks)
    buf = np.zeros(s1 - s0, np.int16)
    cov = np.zeros(s1 - s0, bool)
    for s, x in chunks:
        b = s - s0
        buf[b:b+len(x)] = x
        cov[b:b+len(x)] = True
    wins = np.zeros((8, SR*2), np.float32)
    avalid = np.zeros(8, bool)
    for si, t in enumerate(t8.tolist()):
        cs = int(round((t - 1.0) * SR)) - s0
        ce = cs + SR*2
        lo, hi = max(cs, 0), min(ce, len(buf))
        if hi > lo:
            seg = buf[lo:hi].astype(np.float32) / 32768.0
            wins[si, lo-cs:hi-cs] = seg
            need = SR*2
            avalid[si] = cov[lo:hi].mean() > 0.95 and (hi - lo) == need
    return (wins, avalid, 1)


t0 = time.time(); done = 0; errs = 0; short = 0
for w in mine:
    outp = OUTP / f'{w}.npz'
    if outp.exists():
        done += 1
        continue
    try:
        t8 = np.load(POOL / f'{w}.npz')['t'].astype(np.float64)
        wins, avalid, has_a = decode_audio(MEDIA / f'{srcs[w]}.mp4', t8)
        inp = fe(list(wins), sampling_rate=SR, return_tensors='pt')['input_values']
        with torch.inference_mode():
            out = m(inp)
            logits = w_head(out.pooler_output)
        nh = out.last_hidden_state.shape[1]
        mean_tok = out.last_hidden_state[:, 1:, :].mean(1)
        np.savez(outp,
                 ast_pool=out.pooler_output.numpy().astype(np.float16),
                 ast_mean=mean_tok.numpy().astype(np.float16),
                 ast_527=logits.numpy().astype(np.float16),
                 avalid=avalid.astype(np.int8),
                 has_audio=np.int8(has_a),
                 t=t8.astype(np.float32))
        short += int((~avalid).sum())
        done += 1
        if done % 100 == 0:
            el = time.time() - t0
            print(f'{done}/{len(mine)} {done/el:.2f}/s eta {(len(mine)-done)/(done/el)/60:.0f}min '
                  f'invalid_slots {short}', flush=True)
    except Exception as e:
        errs += 1
        print(f'ERR {w}: {type(e).__name__} {e}', flush=True)
print(f'DONE shard {a.shard}: {done}/{len(mine)} errs {errs} invalid_slots {short} '
      f'in {(time.time()-t0)/60:.1f}min', flush=True)
