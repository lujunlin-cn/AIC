"""R10 ASR coverage probe (family S, R10 4.3): transcribe the FROZEN fragment
audio (never whole sources) with whisper-base, and audit how much of the pool
actually carries intelligible speech.

Per fragment: decode [t_min-1, t_max+1] (~10 s) mono 16 kHz with the exact R9
seek discipline, run whisper-base with timestamps, and store per-slot word
counts (a word belongs to the slot containing its midpoint).  Empty/whisper-
hallucination guard: skip leading/closing lines like 'Thank you' etc.?  NO -
first pass is raw: text, n_words, n_segments, per-slot counts, language.
Coverage verdict = share of fragments with >=4 words inside the 8 s window.

Sharded, one json per fragment (resume-safe).  CPU only.
"""
import argparse, json, os, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '10')
from pathlib import Path
import numpy as np
import av
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='/data/aic/pretrained/whisper-base')
ap.add_argument('--media-root', default='/data/aic/external_datasets/PHD2/raw/youtube')
ap.add_argument('--pool-feats', default='/data/aic/experiments_910a/LFM_V10/pool_feats')
ap.add_argument('--manifest-dir', default='/data/aic/experiments_910a/LFM_V11/r8_npu/r9_audit')
ap.add_argument('--sources', default='/data/aic/experiments_910a/LFM_V11/r10_audio/pilot_sources_64.txt')
ap.add_argument('--limit', type=int, default=32)
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshards', type=int, default=1)
ap.add_argument('--out', default='/data/aic/experiments_910a/LFM_V11/r10_asr')
a = ap.parse_args()

from transformers import WhisperForConditionalGeneration, WhisperProcessor
proc = WhisperProcessor.from_pretrained(a.ckpt)
m = WhisperForConditionalGeneration.from_pretrained(a.ckpt).eval()
print('whisper-base loaded', flush=True)

man = []
for split in ('train_public', 'eval_public'):
    for l in (Path(a.manifest_dir) / f'{split}.jsonl').read_text().splitlines():
        if l.strip():
            man.append(json.loads(l))
srcs = {r['fragment_id']: r['source_id'] for r in man}
keep = set(Path(a.sources).read_text().split())
man = [r for r in man if r['source_id'] in keep]
fids = [r['fragment_id'] for r in man][:a.limit] if a.limit else [r['fragment_id'] for r in man]
mine = [f for i, f in enumerate(fids) if i % a.nshards == a.shard]
OUTP = Path(a.out) / f'p{a.shard}'
OUTP.mkdir(parents=True, exist_ok=True)
print(f'shard {a.shard}/{a.nshards}: {len(mine)} frags', flush=True)
MEDIA = Path(a.media_root)
POOL = Path(a.pool_feats)
SR = 16000


def decode_frag_audio(vid_path, t8):
    """Whole-window mono float32 for the fragment + slot boundary seconds."""
    c = av.open(str(vid_path))
    try:
        if not c.streams.audio:
            return None, t8
        res = av.AudioResampler(format='s16', layout='mono', rate=SR)
        t_min, t_max = float(t8.min()) - 1.0, float(t8.max()) + 1.0
        # bare seek = av.time_base (microseconds); stream= would reinterpret
        # the timestamp in stream time_base (see r10_audio_extract.py note)
        c.seek(max(int((t_min - 2.0) * av.time_base), 0), backward=True)
        chunks = []
        for fr in c.decode(audio=0):
            ts = fr.pts * fr.time_base if fr.pts is not None else None
            for r in res.resample(fr):
                x = r.to_ndarray().reshape(-1).astype(np.int16)
                start = int(round((r.pts * r.time_base) * SR)) if r.pts is not None else 0
                chunks.append((start, x))
            if ts is not None and ts > t_max + 2.0:
                break
    finally:
        c.close()
    if not chunks:
        return None, t8
    s0 = min(s for s, _ in chunks)
    s1 = max(s + len(x) for s, x in chunks)
    buf = np.zeros(s1 - s0, np.int16)
    for s, x in chunks:
        buf[s - s0:s - s0 + len(x)] = x
    cs = int(round(t_min * SR)) - s0
    seg = buf[max(cs, 0):cs + int(round((t_max - t_min) * SR))]
    return (seg.astype(np.float32) / 32768.0), t8


t0 = time.time(); done = 0; errs = 0; naudio = 0
for w in mine:
    outp = OUTP / f'{w}.json'
    if outp.exists():
        done += 1
        continue
    try:
        t8 = np.load(POOL / f'{w}.npz')['t'].astype(np.float64)
        wav, _ = decode_frag_audio(MEDIA / f'{srcs[w]}.mp4', t8)
        if wav is None or len(wav) < SR:
            naudio += 1
            outp.write_text(json.dumps({'fragment_id': w, 'no_audio': True}))
            done += 1
            continue
        inp = proc(wav.tolist(), sampling_rate=SR, return_tensors='pt').input_features
        with torch.inference_mode():
            ids = m.generate(inp, return_timestamps=True, task='transcribe')
        text = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        # batch_decode returns PLAIN STRINGS; timestamps need decode(...,
        # output_offsets=True) (first pass checked isinstance(dec, dict) on a
        # str -> segments stayed empty).
        d = proc.tokenizer.decode(ids[0], skip_special_tokens=False,
                                  output_offsets=True)
        segs = []
        for off in d.get('offsets', []):
            ts = off.get('timestamp') or (None, None)
            if ts[0] is not None:
                segs.append({'t0': float(ts[0]), 't1': float(ts[1] or ts[0]),
                             'text': off['text'].strip()})
        words = text.split()
        # per-slot word counts: whisper timestamps are SECONDS FROM WINDOW
        # START; the decode window begins at t_lo = t8[0]-1, so slot i spans
        # window-relative [i+1, i+2).  (First pass wrongly subtracted the
        # absolute source time -> all-zero histogram.)
        t_lo = float(t8.min()) - 1.0
        per_slot = [0] * 8
        for sg in segs:
            mid = (sg['t0'] + sg['t1']) / 2
            si = int(np.floor(mid - 1.0))
            if 0 <= si < 8:
                per_slot[si] += len(sg['text'].split())
        outp.write_text(json.dumps({'fragment_id': w, 'text': text,
                                    'n_words': len(words), 'n_segments': len(segs),
                                    'segments': segs, 'per_slot': per_slot}))
        done += 1
        if done % 8 == 0:
            print(f'{done}/{len(mine)} {(time.time()-t0)/60:.1f}min', flush=True)
    except Exception as e:
        errs += 1
        print(f'ERR {w}: {type(e).__name__} {e}', flush=True)
print(f'DONE shard {a.shard}: {done}/{len(mine)} errs {errs} no_audio {naudio} '
      f'in {(time.time()-t0)/60:.1f}min', flush=True)
