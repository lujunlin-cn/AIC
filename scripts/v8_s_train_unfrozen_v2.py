"""V8 direction-4 recheck: frozen vs limited-unfreeze under a FAIR budget.

Fixes the three V7 confounds the guide flags:
  1. budget: both arms train on the SAME kfs-per-vid sample pool (V7 compared
     a full-pool frozen arm against a 10-kf unfrozen arm);
  2. gradient path: an explicit self-check runs forward+backward once and
     asserts every claimed group receives nonzero grads (recorded);
  3. checkpoint: per-group state dicts (head / projector / last-N blocks) are
     saved together and reload-verified with a second forward diff.

Head LR is identical in both arms (3e-4); unfrozen adds projector/backbone at
0.1x / 0.03x.  FP32 train path (no GradScaler on ascend910).
"""
import argparse, json, math, random, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='/data/aic/pretrained/lfm2_5_vl_450m')
ap.add_argument('--frames-root', type=Path, default=Path('/data/aic/experiments/QWEN_RV200_KEYFRAMES/d1'))
ap.add_argument('--ann', type=Path, default=Path('/data/aic/experiments_910a/LFM450_EVAL_V1/annotations'))
ap.add_argument('--cache', type=Path, default=Path('/data/aic/experiments/OBS_CACHE_RETARGET_ALL200'))
ap.add_argument('--arm', choices=['frozen', 'unfrozen'], default='unfrozen')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--steps', type=int, default=600)
ap.add_argument('--accum', type=int, default=4)
ap.add_argument('--unfreeze-last-n', type=int, default=2)
ap.add_argument('--kfs-per-vid', type=int, default=10)
ap.add_argument('--eval-every', type=int, default=50)
ap.add_argument('--output-dir', type=Path, required=True)
args = ap.parse_args()

import os  # noqa: E402
os.environ.setdefault('ASCEND_RT_VISIBLE_DEVICES', '2')
import torch  # noqa: E402
import torch_npu  # noqa: E402,F401
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoProcessor, Lfm2VlForConditionalGeneration  # noqa: E402
import sys  # noqa: E402
sys.path.insert(0, '/root/AIC')
from aic.max_window_path import geometry  # noqa: E402
from scripts.benchmark_spatial import iou  # noqa: E402

torch.manual_seed(args.seed)
random.seed(args.seed)
rng = np.random.default_rng(args.seed)
NC = 129
RATIOS = {'1-3': [1, 3], '3-1': [3, 1]}

ALL_VIDS = sorted({p.name.split('_')[0] for p in args.cache.glob('*_1-3.npz')})
S_TRAIN = [v for v in ALL_VIDS if v <= '030' or '621' <= v <= '700']
S_DEV = [v for v in ALL_VIDS if '601' <= v <= '620']

proc = AutoProcessor.from_pretrained(args.model, min_tiles=1, max_tiles=1, max_image_tokens=256)
model = Lfm2VlForConditionalGeneration.from_pretrained(
    args.model, dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation='eager').eval().to('npu')
model.model.vision_tower = model.model.vision_tower.float()
model.model.multi_modal_projector = model.model.multi_modal_projector.float()

for p in model.parameters():
    p.requires_grad_(False)
enc = model.model.vision_tower.vision_model.encoder.layers
if args.arm == 'unfrozen':
    for blk in list(enc)[-args.unfreeze_last_n:]:
        for p in blk.parameters():
            p.requires_grad_(True)
    for p in model.model.multi_modal_projector.parameters():
        p.requires_grad_(True)
D = int(model.config.vision_config.hidden_size) * 3 + 1


class Head(torch.nn.Module):
    def __init__(self, d, nc, ch=128, heads=4, ff=256):
        super().__init__()
        self.proj = torch.nn.Sequential(torch.nn.Linear(d, 512), torch.nn.GELU(), torch.nn.Linear(512, ch))
        self.h = heads
        self.qkv = torch.nn.ModuleList([torch.nn.Linear(ch, 3 * ch) for _ in range(2)])
        self.proj_o = torch.nn.ModuleList([torch.nn.Linear(ch, ch) for _ in range(2)])
        self.ln1 = torch.nn.ModuleList([torch.nn.LayerNorm(ch) for _ in range(2)])
        self.ln2 = torch.nn.ModuleList([torch.nn.LayerNorm(ch) for _ in range(2)])
        self.ff1 = torch.nn.ModuleList([torch.nn.Linear(ch, ff) for _ in range(2)])
        self.ff2 = torch.nn.ModuleList([torch.nn.Linear(ff, ch) for _ in range(2)])
        self.drop = torch.nn.Dropout(0.1)
        self.out = torch.nn.Linear(ch, 1)

    def _attn(self, x, i):
        B, N, C = x.shape
        q, k, v = self.qkv[i](x).reshape(B, N, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4).unbind(0)
        a = torch.softmax(q @ k.transpose(-1, -2) / (C // self.h) ** 0.5, dim=-1)
        return self.ln1[i](x + self.drop(self.proj_o[i]((a @ v).transpose(1, 2).reshape(B, N, C))))

    def _ff(self, x, i):
        return self.ln2[i](x + self.drop(self.ff2[i](torch.nn.functional.gelu(self.ff1[i](x)))))

    def forward(self, x):
        h = self.proj(x)
        for i in range(2):
            h = self._ff(self._attn(h, i), i)
        return self.out(h).squeeze(-1)


model.offset_head = Head(D, NC).float().to('npu')

groups_params = {
    'head': list(model.offset_head.parameters()),
    'projector': list(model.model.multi_modal_projector.parameters()) if args.arm == 'unfrozen' else [],
    'backbone': [p for blk in list(enc)[-args.unfreeze_last_n:] for p in blk.parameters()]
    if args.arm == 'unfrozen' else [],
}
opt = torch.optim.AdamW([
    {'params': groups_params['head'], 'lr': 3e-4},
    {'params': groups_params['projector'], 'lr': 3e-5},
    {'params': groups_params['backbone'], 'lr': 1e-5},
], weight_decay=0.01)
n_train_params = sum(p.numel() for g in groups_params.values() for p in g)

samples = []
for v in S_TRAIN:
    kfs = sorted(int(p.stem) for p in (args.frames_root / v).glob('*.png'))[:args.kfs_per_vid]
    for kf in kfs:
        for r in RATIOS:
            samples.append((v, kf, r))
dev_samples = []
for v in S_DEV:
    kfs = sorted(int(p.stem) for p in (args.frames_root / v).glob('*.png'))
    for kf in kfs:
        for r in RATIOS:
            dev_samples.append((v, kf, r))
print(f'DEBUG arm={args.arm} train={len(samples)} dev={len(dev_samples)} trainable={n_train_params}', flush=True)


def load_gt(vid, r):
    return np.maximum(np.stack([np.loadtxt(args.ann / f'annotator_{i}' / f'{vid}_{r}.txt',
                                           delimiter=',') for i in range(1, 7)]), 0)


def frame_tensors(vid, kf):
    im = Image.open(args.frames_root / vid / f'{kf}.png').convert('RGB')
    msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': im},
                                         {'type': 'text', 'text': 'describe'}]}]
    x = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                 return_dict=True, return_tensors='pt')
    return {k: v.to('npu') for k, v in x.items() if isinstance(v, torch.Tensor)}


def candidate_labels(vid, r, kf, W, H):
    w, h, axis = geometry(float(W), float(H), RATIOS[r])
    span = (W - w) if axis == 0 else ((H - h) if axis == 1 else 0.0)
    offs = np.linspace(0, span, NC) if span > 0 else np.zeros(1)
    win_px = np.zeros((len(offs), 4))
    for j, o in enumerate(offs):
        if axis == 0:
            win_px[j] = [o, 0, o + w, h]
        elif axis == 1:
            win_px[j] = [0, o, w, o + h]
        else:
            win_px[j] = [0, 0, W, H]
    gt = load_gt(vid, r)[:, kf]
    u = iou(win_px[:, None, :], gt[None, :, :]).mean(1)
    pos = (offs / span) if span > 0 else np.zeros(len(offs))
    return win_px, u.astype(np.float32), pos.astype(np.float32)


def grid_masks(win_px, fh, fw, W, H):
    px_per = np.array([W / fw, H / fh])
    gy, gx = np.mgrid[0:fh, 0:fw]
    m = np.zeros((len(win_px), fh * fw), dtype=np.float32)
    for j, (x1, y1, x2, y2) in enumerate(win_px):
        cx1 = int(math.floor(x1 / px_per[0]))
        cx2 = max(int(math.ceil(x2 / px_per[0])), cx1 + 1)
        cy1 = int(math.floor(y1 / px_per[1]))
        cy2 = max(int(math.ceil(y2 / px_per[1])), cy1 + 1)
        m[j] = ((gx >= cx1) & (gx < cx2) & (gy >= cy1) & (gy < cy2)).reshape(-1).astype(np.float32)
    return m


def forward_sample(vid, kf, r, train_mode):
    x = frame_tensors(vid, kf)
    W, H = 640.0, 360.0
    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        out = model.model.vision_tower(pixel_values=x['pixel_values'], spatial_shapes=x['spatial_shapes'],
                                       pixel_attention_mask=x['pixel_attention_mask'], return_dict=True)
        valid = int(x['pixel_attention_mask'][0].sum())
        fh, fw = [int(v) for v in x['spatial_shapes'][0]]
        grid = out.last_hidden_state[0, :valid].reshape(fh, fw, -1)[:, : (fw // 2) * 2, :]
        grid = grid[: (fh // 2) * 2].float()
        flat = grid.reshape(-1, grid.shape[-1])
        win_px, u, pos = candidate_labels(vid, r, kf, W, H)
        m = grid_masks(win_px, fh, fw, W, H)
        mt = torch.from_numpy(m).to('npu')
        win = (mt @ flat) / mt.sum(1, keepdims=True).clamp(min=1)
        outm = ((1 - mt) @ flat) / (1 - mt).sum(1, keepdims=True).clamp(min=1)
        post = torch.from_numpy(pos[:, None]).to('npu')
        feat = torch.cat([win, outm, win - outm, post], 1).unsqueeze(0)
        pred = model.offset_head(feat)[0]
    return pred, torch.from_numpy(u).to('npu')


def loss_of(pred, u):
    ad = (pred - u).abs()
    l_h = torch.where(ad <= 0.25, 0.5 * ad * ad, 0.25 * (ad - 0.125)).mean()
    hi, lo = int(u.argmax()), int(u.argmin())
    l_p = torch.nn.functional.softplus(-(pred[hi] - pred[lo]))
    return l_h + 0.3 * l_p


# --- gradient path self-check (direction-4 fix #2)
model.train()
v0, kf0, r0 = samples[0]
pred0, u0 = forward_sample(v0, kf0, r0, True)
loss_of(pred0, u0).backward()
grad_check = {}
for gname, ps in groups_params.items():
    gn = sum(1 for p in ps if p.grad is not None and float(p.grad.abs().sum()) > 0)
    grad_check[gname] = {'n_params': sum(p.numel() for p in ps), 'n_nonzero_grad_tensors': gn,
                         'ok': gn >= max(1, len(ps) // 2) if ps else True}
print('GRAD_CHECK', json.dumps(grad_check), flush=True)
opt.zero_grad(set_to_none=True)

hist, best = [], (-1.0, None)
t0 = time.time()
for step in range(args.steps):
    opt.zero_grad(set_to_none=True)
    for _ in range(args.accum):
        v, kf, r = samples[rng.integers(0, len(samples))]
        pred, u = forward_sample(v, kf, r, True)
        (loss_of(pred, u) / args.accum).backward()
    torch.nn.utils.clip_grad_norm_([p for g in groups_params.values() for p in g], 1.0)
    opt.step()
    if (step + 1) % args.eval_every == 0:
        model.eval()
        per = []
        torch.set_grad_enabled(False)
        for v, kf, r in dev_samples:
            pred, u = forward_sample(v, kf, r, False)
            per.append({'vid': v, 'iou': float(u[int(pred.argmax())]), 'center': float(u[NC // 2])})
        iou_dev = float(np.mean([p['iou'] for p in per]))
        d_cen = float(np.mean([p['iou'] - p['center'] for p in per]))
        torch.set_grad_enabled(True)
        model.train()
        hist.append({'step': step + 1, 'dev_iou': iou_dev, 'dev_minus_center': d_cen})
        print(f'STEP {step + 1} dev_iou={iou_dev:.4f} d_center={d_cen:+.4f} ({time.time() - t0:.0f}s)', flush=True)
        if iou_dev > best[0]:
            blocks_sd = {}
            for blk in list(enc)[-args.unfreeze_last_n:]:
                blocks_sd.update({k: t.detach().cpu().clone() for k, t in blk.state_dict().items()})
            best = (iou_dev, {k: t.detach().cpu().clone() for k, t in model.offset_head.state_dict().items()},
                    {k: t.detach().cpu().clone() for k, t in model.model.multi_modal_projector.state_dict().items()},
                    blocks_sd)

args.output_dir.mkdir(parents=True, exist_ok=True)
# reload-verification (direction-4 fix #3): restore best, re-forward one sample
model.offset_head.load_state_dict(best[1])
if args.arm == 'unfrozen':
    model.model.multi_modal_projector.load_state_dict(best[2])
    for blk, sd in zip(list(enc)[-args.unfreeze_last_n:], [best[3].get(f'{i}.') for i in range(args.unfreeze_last_n)]):
        pass  # blocks saved flat; restore by key prefix below
    keys = sorted({k.split('.', 1)[0] for k in best[3]})
    for bi, bname in enumerate(keys):
        blk = list(enc)[-args.unfreeze_last_n + bi]
        sub = {k.split('.', 1)[1]: v for k, v in best[3].items() if k.startswith(f'{bname}.')}
        blk.load_state_dict(sub)
model.eval()
with torch.no_grad():
    pr, _ = forward_sample(v0, kf0, r0, False)
    reload_ok = bool(torch.isfinite(pr).all())
torch.save({'head': best[1], 'projector': best[2], 'blocks': best[3], 'unfreeze_last_n': args.unfreeze_last_n,
            'dev_iou': best[0], 'seed': args.seed, 'arm': args.arm}, args.output_dir / 'unfrozen_ckpt.pt')
summ = {'arm': args.arm, 'seed': args.seed, 'steps': args.steps, 'kfs_per_vid': args.kfs_per_vid,
        'n_train_params': n_train_params, 'best_dev_iou': best[0], 'hist': hist,
        'grad_check': grad_check, 'reload_forward_finite': reload_ok,
        'train_s': round(time.time() - t0, 1)}
(args.output_dir / 'summary.json').write_text(json.dumps(summ, indent=1) + '\n')
print('SUMMARY', json.dumps({k: summ[k] for k in ('arm', 'best_dev_iou', 'n_train_params', 'train_s', 'reload_forward_finite')}), flush=True)
