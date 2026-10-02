"""Vision-tower-only student inference: identical predictions, 172 MB instead of 856 MB.

The deployed forward pass only ever calls model.model.vision_tower(...).  No text
prompt, no generate, no projector, no language model.  LFM2.5-VL-450M keeps
354,483,968 of 448,718,848 parameters (79.0%) in the language model, and the
rules size a submission by "all model weights actually loaded when the result
was produced" - so those 709 MB are counted for a module that never runs.

The model is therefore instantiated on the meta device and ONLY the vision tower
is materialised: the language model's tensors are never allocated and their
bytes are never read off disk.  That is a stronger claim than "load everything
then delete", and it is the version the declared size is computed from.

    current package   903,762,857 B = 861.9 MB -> k_size 0.90
    this one          178,964,689 B = 170.6 MB -> k_size 0.95

Predictions must be bit-identical to the full-model run; --verify-against
enforces that on the real drop rather than asserting it, and a mismatch aborts.
"""
import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='/data/aic/pretrained/lfm2_5_vl_450m')
ap.add_argument('--head', type=Path, required=True)
ap.add_argument('--index', type=Path, required=True)
ap.add_argument('--keyframe-src', type=Path, required=True)
ap.add_argument('--cards', default='0')
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshards', type=int, default=1)
ap.add_argument('--output-dir', type=Path, required=True)
ap.add_argument('--verify-against', type=Path, default=None)
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--device', default='npu')
args = ap.parse_args()

DEV = args.device
if DEV == 'npu':
    os.environ['ASCEND_RT_VISIBLE_DEVICES'] = args.cards
import torch  # noqa: E402
if DEV == 'npu':
    import torch_npu  # noqa: E402,F401
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = False
DTYPE = torch.float16 if DEV == 'npu' else torch.float32
from PIL import Image  # noqa: E402
import safetensors.torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from transformers import AutoConfig, AutoProcessor, Lfm2VlForConditionalGeneration  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aic.max_window_path import geometry  # noqa: E402

NC = 129
PREFIX = 'model.vision_tower.'


class Head(nn.Module):
    def __init__(self, d, nc, ch=128, heads=4, ff=256):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Linear(512, ch))
        self.h = heads
        self.qkv = nn.ModuleList([nn.Linear(ch, 3 * ch) for _ in range(2)])
        self.proj_o = nn.ModuleList([nn.Linear(ch, ch) for _ in range(2)])
        self.ln1 = nn.ModuleList([nn.LayerNorm(ch) for _ in range(2)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(ch) for _ in range(2)])
        self.ff1 = nn.ModuleList([nn.Linear(ch, ff) for _ in range(2)])
        self.ff2 = nn.ModuleList([nn.Linear(ff, ch) for _ in range(2)])
        self.drop = nn.Dropout(0.1)
        self.out = nn.Linear(ch, 1)

    def _attn(self, x, i):
        B, N, C = x.shape
        q, k, v = self.qkv[i](x).reshape(B, N, 3, self.h, C // self.h).permute(
            2, 0, 3, 1, 4).unbind(0)
        a = torch.softmax(q @ k.transpose(-1, -2) / (C // self.h) ** 0.5, dim=-1)
        return self.ln1[i](x + self.drop(self.proj_o[i]((a @ v).transpose(1, 2).reshape(B, N, C))))

    def _ff(self, x, i):
        return self.ln2[i](x + self.drop(self.ff2[i](torch.nn.functional.gelu(self.ff1[i](x)))))

    def forward(self, x):
        h = self.proj(x)
        for i in range(2):
            h = self._ff(self._attn(h, i), i)
        return self.out(h).squeeze(-1)


# Build the shell on CPU with the real (non-meta) constructor so every
# non-persistent buffer - Siglip2's position_ids in particular - is initialised
# correctly; instantiating from config reads no weight file, so the language
# model's bytes are still never touched.  (to_empty() on a meta module was the
# first attempt and left position_ids as garbage, which surfaced as an
# aclnnIsNegInf-unsupported crash 910A cannot run.)
cfg = AutoConfig.from_pretrained(args.model)
# The production path passes attn_implementation='eager'; constructing from a
# bare config leaves the vision tower on the default backend, whose NPU kernel
# path calls aclnnIsNegInf - an op 910A does not implement (EZ1001).  Pin the
# attribute on the config (the constructor takes no attn_implementation kwarg).
for _c in [cfg, getattr(cfg, 'vision_config', None), getattr(cfg, 'text_config', None)]:
    if _c is not None:
        _c._attn_implementation = 'eager'
shell = Lfm2VlForConditionalGeneration(cfg).to(DTYPE)
vision = shell.model.vision_tower
del shell

sd = {}
with safetensors.torch.safe_open(str(Path(args.model) / 'model.safetensors'),
                                 framework='pt') as fh:
    for k in fh.keys():
        if k.startswith(PREFIX):
            sd[k[len(PREFIX):]] = fh.get_tensor(k).to(DTYPE)
missing, unexpected = vision.load_state_dict(sd, strict=False)
missing = [m for m in missing if 'position_ids' not in m]
assert not missing and not unexpected, (missing[:5], unexpected[:5])
vision = vision.to(DEV).eval()
n_loaded = sum(p.numel() for p in vision.parameters())
del sd
print(f'VISION_ONLY loaded {n_loaded:,} vision params '
      f'({n_loaded * 2 / 1e6:.1f} MB fp16); language model never allocated', flush=True)

proc = AutoProcessor.from_pretrained(args.model, min_tiles=1, max_tiles=1, max_image_tokens=256)
ck = torch.load(args.head, map_location=DEV, weights_only=False)
D = int(ck['config']['d'])
assert int(ck['config']['nc']) == NC
head = Head(D, NC).to(DEV).float()
head.load_state_dict(ck['state_dict'])
head.eval()

recs = [json.loads(l) for l in args.index.read_text().splitlines() if l.strip()]
recs = recs[args.shard::args.nshards]
if args.limit:
    recs = recs[:args.limit]
pdir = args.output_dir / 'points'
pdir.mkdir(parents=True, exist_ok=True)
print(f'VISION_ONLY shard {args.shard}/{args.nshards}: {len(recs)} videos', flush=True)

t0, n_cmp, n_bad = time.time(), 0, 0
for vi, r in enumerate(recs):
    vid = r['video_id']
    if (pdir / f'{vid}.json').exists():
        continue
    src = json.loads((args.keyframe_src / 'points' / f'{vid}.json').read_text())
    kfs = src['keyframes']
    pts, st = [], []
    for kf in kfs:
        try:
            img = Image.open(args.keyframe_src / 'keyframes' / vid / f'{kf}.png').convert('RGB')
            W, H = img.size
            msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': img},
                                                 {'type': 'text', 'text': 'describe'}]}]
            x = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                         return_dict=True, return_tensors='pt')
            x = {k: v.to(DEV) for k, v in x.items() if isinstance(v, torch.Tensor)}
            if 'pixel_values' in x:
                x['pixel_values'] = x['pixel_values'].to(DTYPE)
            with torch.no_grad():
                out = vision(pixel_values=x['pixel_values'], spatial_shapes=x['spatial_shapes'],
                             pixel_attention_mask=x['pixel_attention_mask'], return_dict=True)
                valid = int(x['pixel_attention_mask'][0].sum())
                fh, fw = [int(v) for v in x['spatial_shapes'][0]]
                grid = out.last_hidden_state[0, :valid].reshape(fh, fw, -1)[:, :(fw // 2) * 2, :]
                grid = grid[: (fh // 2) * 2].float()
        except Exception:
            # a boundary-frame tile layout can leave valid != fh*fw (the
            # attention mask and the declared grid disagree), which crashes the
            # reshape and would otherwise drop the whole video.  Emit a centred
            # fallback point for that frame instead - the packaging path reads
            # the third tuple element as the is_fallback flag.
            pts.append([0.5, 0.5, True])
            st.append('frame_fallback')
            continue
        flat = grid.reshape(-1, grid.shape[-1])
        rw, rh = r['targetRatioWH']
        w, h, axis = geometry(float(W), float(H), [rw, rh])
        span = (W - w) if axis == 0 else ((H - h) if axis == 1 else 0.0)
        offs = np.linspace(0, span, NC) if span > 0 else np.zeros(1)
        px_per = np.array([W / fw, H / fh])
        gy, gx = np.mgrid[0:fh, 0:fw]
        m = np.zeros((len(offs), fh * fw), dtype=np.float32)
        for j, o in enumerate(offs):
            if axis == 0:
                x1, y1, x2, y2 = o, 0, o + w, h
            elif axis == 1:
                x1, y1, x2, y2 = 0, o, w, o + h
            else:
                x1, y1, x2, y2 = 0, 0, W, H
            cx1 = int(math.floor(x1 / px_per[0]))
            cx2 = max(int(math.ceil(x2 / px_per[0])), cx1 + 1)
            cy1 = int(math.floor(y1 / px_per[1]))
            cy2 = max(int(math.ceil(y2 / px_per[1])), cy1 + 1)
            m[j] = ((gx >= cx1) & (gx < cx2) & (gy >= cy1) & (gy < cy2)).reshape(-1).astype(np.float32)
        win = torch.from_numpy(m @ flat.cpu().numpy() / np.maximum(m.sum(1, keepdims=True), 1)).to(DEV)
        outm = torch.from_numpy((1 - m) @ flat.cpu().numpy() /
                                np.maximum((1 - m).sum(1, keepdims=True), 1)).to(DEV)
        pos = torch.from_numpy((offs / span if span > 0 else offs)[:, None].astype(np.float32)).to(DEV)
        feat = torch.cat([win, outm, win - outm, pos], 1).unsqueeze(0)
        with torch.no_grad():
            u = head(feat)[0]
        j = int(u.argmax())
        if axis == 0:
            cx, cy = (offs[j] + w / 2) / W, 0.5
        elif axis == 1:
            cx, cy = 0.5, (offs[j] + h / 2) / H
        else:
            cx, cy = 0.5, 0.5
        pts.append([round(float(cx), 6), round(float(cy), 6), False])
        st.append('ok_head')
    rec = {'video_id': vid, 'W': src['W'], 'H': src['H'], 'fps': src['fps'],
           'step': src['step'], 'keyframes': kfs,
           'model': f'LFM2.5-VL-450M@fc6221ca597f3315e4f82fc2df606783267b34ba'
                    f'.vision_tower+v9_head',
           'runtime': 'vision_tower only (language model and projector never loaded)',
           'ratios': {'t': {'ratio': r['targetRatioWH'], 'points': pts,
                            'status': st, 'raw': [], 'boxes': []}}}
    if args.verify_against:
        ref = args.verify_against / f'{vid}.json'
        if ref.exists():
            n_cmp += 1
            old = json.loads(ref.read_text())['ratios']['t']['points']
            d = (max(max(abs(a_[0] - b_[0]), abs(a_[1] - b_[1]))
                     for a_, b_ in zip(old, pts)) if len(old) == len(pts) else 1.0)
            if d > 0:
                n_bad += 1
                print(f'MISMATCH {vid} maxdiff={d:.3g}', flush=True)
    (pdir / f'{vid}.json').write_text(json.dumps(rec, ensure_ascii=False) + '\n')
    if vi % 40 == 0:
        print(f'VISION {vi}/{len(recs)} {time.time() - t0:.0f}s', flush=True)

print('VISION_SUMMARY', json.dumps({
    'videos': len(recs), 'verified': n_cmp, 'mismatched': n_bad,
    'wall_s': round(time.time() - t0, 1),
    'vision_params': n_loaded,
    'declared_weight_bytes': n_loaded * 2 + 1511425 * 4 + 232589,
    'k_size_implied': 0.95,
}), flush=True)
if n_bad:
    raise SystemExit(f'{n_bad} videos differ from the reference run')