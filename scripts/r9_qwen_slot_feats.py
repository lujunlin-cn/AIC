"""R9 third batch: Qwen2.5-VL-7B **visual tower only** features for the PHD2
slot pool (R9 report Q3 arm 3; static-semantic control - NOT a claim about
full 7B inference capability).

Protocol (recorded before extraction):
  STATIC 8-ANCHOR mode (R9 3.3 option 1): one image per slot, the frame at
  the frozen slot time t - NO temporal context, NO cross-slot information.
  The tower's own temporal patch merge is irrelevant at T=1, so all 8 slot
  anchors stay separate observations.
  Readouts = mean of pre-merger block tokens at blocks[31] (last of 32) and
  blocks[23] (the R9-named candidates); the post-merger projector path is
  NOT used (its 3584-d LLM-facing output mixes a different scale).
  Processor = AutoProcessor with the SFT-exercised pixel budget
  (448*28*28 .. 448*448), fp16, eager attention on NPU.
Schema: npz per frag {qwen_l32 (8,1280), qwen_l24 (8,1280), t, fps, nmiss}.
Decoder/slots/manifest identical to r9_iv2_slot_feats.py; no official media.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '4')

p = argparse.ArgumentParser()
p.add_argument('--model-path', default='/data/aic/pretrained/Qwen2.5-VL-7B-Instruct')
p.add_argument('--media-root', default='/data/aic/external_datasets/PHD2/raw/youtube')
p.add_argument('--pool-feats', default='/data/aic/experiments_910a/LFM_V10/pool_feats')
p.add_argument('--manifest-dir', default='/data/aic/experiments_910a/LFM_V11/r8_npu/r9_audit')
p.add_argument('--out', default='/data/aic/experiments_910a/LFM_V11/r9_qwen_slot')
p.add_argument('--shard', type=int, default=0)
p.add_argument('--nshards', type=int, default=1)
p.add_argument('--limit', type=int, default=0)
p.add_argument('--dump-frames', type=int, default=0)
a = p.parse_args()

import cv2  # noqa: E402
import av  # noqa: E402
os.environ.setdefault('ASCEND_RT_VISIBLE_DEVICES', str(0))  # logical 0 = physical 2 (aic-batch map 0-5 -> 2-7)
import torch_npu  # noqa: E402,F401
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False
assert torch.npu.device_count() > 0, 'no NPU visible - refusing CPU fallback'
from transformers import AutoImageProcessor, Qwen2_5_VLForConditionalGeneration  # noqa: E402

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    a.model_path, torch_dtype=torch.float16,
    attn_implementation='eager').to('npu')
# AutoImageProcessor, NOT AutoProcessor: the full processor's __call__
# walks the chat text for image tokens and crashes on text=None (observed
# transformers 5.x, processing_qwen2_5_vl.py:176).  The visual tower only
# needs pixel_values + image_grid_thw.
iproc = AutoImageProcessor.from_pretrained(a.model_path, min_pixels=448 * 28 * 28,
                                           max_pixels=448 * 448)
vis = model.model.visual
NB = len(vis.blocks)
assert NB >= 32, f'visual depth {NB} < 32 - protocol assumes 32 blocks'
L24, L32 = {}, {}
def _hk(idx, store):
    def hook(_m, _i, out):
        x = out[0] if isinstance(out, tuple) else out
        store['tok'] = x
    return hook
vis.blocks[23].register_forward_hook(_hk(23, L24))
vis.blocks[31].register_forward_hook(_hk(31, L32))
model.eval()
print(f'Qwen2.5-VL-7B visual tower on NPU ({NB} blocks, hooks 23/31)', flush=True)

man = []
for split in ('train_public', 'eval_public'):
    for l in (Path(a.manifest_dir) / f'{split}.jsonl').read_text().splitlines():
        if l.strip():
            man.append(json.loads(l))
fids = [r['fragment_id'] for r in man]
srcs = {r['fragment_id']: r['source_id'] for r in man}
mine = [f for i, f in enumerate(fids) if i % a.nshards == a.shard]
if a.limit:
    mine = mine[:a.limit]
OUTP = Path(a.out) / f'p{a.shard}'
OUTP.mkdir(parents=True, exist_ok=True)
print(f'shard {a.shard}/{a.nshards}: {len(mine)} frags', flush=True)

MEDIA = Path(a.media_root)
POOL = Path(a.pool_feats)


def read_center(vid_path, t8, dump_dir=None):
    """8 center-frame PIL-ready arrays; missing frames stay zero + counted."""
    c = av.open(str(vid_path))
    st = c.streams.video[0]
    fps = float(st.average_rate)
    need = {}
    for si, t in enumerate(t8.tolist()):
        fr = max(int(round(t * fps)), 0)
        need.setdefault(fr, []).append(si)
    minfr, maxfr = min(need), max(need)
    c.seek(max(int((minfr - 2) / fps * av.time_base), 0), backward=True)
    imgs = {}
    for fr in c.decode(video=0):
        idx = int(round(fr.pts * fr.time_base * fps))
        if idx in need:
            imgs[idx] = cv2.resize(fr.to_ndarray(format='rgb24'), (448, 448),
                                   interpolation=cv2.INTER_CUBIC)
        if idx >= maxfr:
            break
    c.close()
    if dump_dir:
        dump_dir.mkdir(parents=True, exist_ok=True)
        for si in (0, 4):
            fr = max(int(round(t8[si] * fps)), 0)
            if fr in imgs:
                cv2.imwrite(str(dump_dir / f'{vid_path.stem}_s{si}_t{t8[si]:.2f}_f{fr}.jpg'),
                            cv2.cvtColor(imgs[fr], cv2.COLOR_RGB2BGR))
    nmiss = len(need) - len(imgs)
    out = []
    for si, t in enumerate(t8.tolist()):
        fr = max(int(round(t * fps)), 0)
        out.append(imgs.get(fr))
    return out, fps, nmiss


t0 = time.time(); done = 0; errs = 0; miss_tot = 0
for w in mine:
    outp = OUTP / f'{w}.npz'
    if outp.exists():
        done += 1
        continue
    try:
        t8 = np.load(POOL / f'{w}.npz')['t'].astype(np.float64)
        arrs, mfps, nmiss = read_center(MEDIA / f'{srcs[w]}.mp4', t8,
                                        dump_dir=OUTP / '_frames' if a.dump_frames > done else None)
        from PIL import Image
        imgs = [Image.fromarray(x) if x is not None else Image.new('RGB', (448, 448))
                for x in arrs]
        px = iproc(images=imgs, return_tensors='pt')
        pixel_values = px['pixel_values'].half().npu()
        grid = px['image_grid_thw'].npu()
        with torch.inference_mode():
            out32 = vis(pixel_values, grid_thw=grid)     # post-merger, unused for readout
            h32 = L32['tok']                              # (sum_patches, 1280)
            h24 = L24['tok']
        # split back to 8 images by grid areas (merge factor 2x2)
        areas = (grid[:, 1] * grid[:, 2]).tolist()   # pre-merge patches (h32 is blocks output, NOT merged)
        assert sum(areas) == h32.shape[0], (sum(areas), h32.shape)
        l32, l24, o = [], [], 0
        for n in areas:
            l32.append(h32[o:o + n].mean(0).float())
            l24.append(h24[o:o + n].mean(0).float())
            o += n
        l32 = torch.stack(l32); l24 = torch.stack(l24)
        assert l32.shape[0] == 8, l32.shape
        np.savez(outp,
                 qwen_l32=l32.cpu().numpy().astype(np.float16),
                 qwen_l24=l24.cpu().numpy().astype(np.float16),
                 t=t8.astype(np.float32), fps=np.float32(mfps),
                 nmiss=np.int32(nmiss))
        miss_tot += nmiss
        done += 1
        if done % 200 == 0:
            el = time.time() - t0
            print(f'{done}/{len(mine)} {done/el:.2f}/s eta {(len(mine)-done)/(done/el)/60:.0f}min miss {miss_tot}',
                  flush=True)
    except Exception as e:
        errs += 1
        print(f'ERR {w}: {e}', flush=True)
print(f'DONE shard {a.shard}: {done}/{len(mine)} errs {errs} missed_frames {miss_tot} '
      f'in {(time.time()-t0)/60:.1f}min', flush=True)
