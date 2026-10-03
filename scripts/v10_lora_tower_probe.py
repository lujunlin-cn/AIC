"""LoRA probe: is the frozen Siglip2 tower the ranking wall?  (P3, last lever)

Everything above the tower has been eliminated by experiment: pooling
(mean/attn/topk), structure (single/multi-scale TCN), motion features
(+delta), label protocol (OR/vote/strict), training breadth (mixed-only vs
+allneg/+allpos), and both decision-layer routes (calibration, budget
utility).  Ten variants, all landing in heldout AP 0.677-0.696.  The wall sits
in the frozen representation itself - Siglip2 was aligned to image-text
similarity, not to "is this second worth a GIF".

This probe attaches LoRA adapters (r=8) to the vision tower's attention
projections and backpropagates the SAME temporal objective through the tower.
Everything else is held fixed: same pooled-mean head, same TCN, same mixed
fragments, same source split, same steps.  The single question:

    does heldout AP escape the 0.677-0.696 band?

An escape means the representation was the wall and reshaping it is the new
main line (with the deployed tower swapped for the LoRA'd one).  No escape
means the GIF-interval labels cap ANY image-derived representation near 0.69,
and only a video-native encoder could change that.
"""
import argparse, json, os, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '8')
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='/data/aic/pretrained/lfm2_5_vl_450m')
ap.add_argument('--pool-feats', type=Path, default=Path('/data/aic/experiments_910a/LFM_V10/pool_feats'),
                help='only used for the frame-time index; pixels are loaded fresh')
ap.add_argument('--frames-root', type=Path, default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/frames'))
ap.add_argument('--index', type=Path, default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/index_clean.jsonl'))
ap.add_argument('--eval-sources', type=Path, default=Path('/data/aic/experiments_910a/LFM_V10/eval_sources_50.json'))
ap.add_argument('--cards', default='0')
ap.add_argument('--steps', type=int, default=900)
ap.add_argument('--fragments-per-step', type=int, default=8)
ap.add_argument('--eval-every', type=int, default=300)
ap.add_argument('--eval-limit', type=int, default=500,
                help='mid-training evals score the first N held-out fragments; 0 = full. '
                     'A full pass is ~13 min and the probe does several.')
ap.add_argument('--lora-r', type=int, default=8)
ap.add_argument('--lr', type=float, default=3e-4)
ap.add_argument('--out', type=Path, default=Path('/data/aic/experiments_910a/LFM_V10/lora_probe.json'))
args = ap.parse_args()

os.environ['ASCEND_RT_VISIBLE_DEVICES'] = args.cards
import torch  # noqa: E402
import torch_npu  # noqa: E402,F401
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False
assert torch.npu.device_count() > 0, 'no NPU'
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoProcessor, Lfm2VlForConditionalGeneration  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402

DEV = 'npu'
sel = json.loads(Path('/data/aic/external_datasets/PHD2/annotations/selections/train.json').read_text())
eval_set = set(json.loads(args.eval_sources.read_text()))

proc = AutoProcessor.from_pretrained(args.model, min_tiles=1, max_tiles=1, max_image_tokens=256)
model = Lfm2VlForConditionalGeneration.from_pretrained(
    args.model, dtype=torch.float16, low_cpu_mem_usage=True,
    attn_implementation='eager').to(DEV)
# freeze everything, then hand the vision tower to peft
for p_ in model.parameters():
    p_.requires_grad_(False)
lcfg = LoraConfig(
    r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
    # names are RELATIVE to the tower: vision_model.encoder.layers.N.self_attn.q_proj
    target_modules=r'.*self_attn\.(q_proj|k_proj|v_proj|out_proj)$',
    bias='none')
model.model.vision_tower = get_peft_model(model.model.vision_tower, lcfg)
model.model.vision_tower.print_trainable_parameters()


class TCN(torch.nn.Module):
    def __init__(self, d_in=768, ch=128, dils=(1, 2, 4)):
        super().__init__()
        self.proj = torch.nn.Conv1d(d_in, ch, 1)
        self.convs = torch.nn.ModuleList([torch.nn.Conv1d(ch, ch, 3, padding=d, dilation=d) for d in dils])
        self.out = torch.nn.Conv1d(ch, 1, 1)

    def forward(self, x):
        h = self.proj(x.transpose(1, 2))
        for c in self.convs:
            h = torch.nn.functional.gelu(c(h) + h)
        return self.out(h).squeeze(1)


tcn = TCN().to(DEV).float()
rows = [json.loads(l) for l in args.index.read_text().splitlines() if l.strip()]
train_rows = [r for r in rows if r['src'] not in eval_set]
eval_rows = [r for r in rows if r['src'] in eval_set]
rng = np.random.default_rng(20261003)
rng.shuffle(train_rows)


def fragment_labels(r):
    t0, L = float(r['t0']), float(r['L'])
    ivs = []
    for recs in sel.get(r['src'], {}).values():
        for rec in recs:
            a_, b_ = float(rec['t0']) - t0, float(rec['t1']) - t0
            if b_ > a_:
                ivs.append((a_, b_))
    kdir = args.frames_root / r['video_id']
    jpgs = sorted(kdir.glob('*.jpg'), key=lambda p: float(p.stem))
    if len(jpgs) < 4:
        return None, None, None
    stems = [float(p.stem) for p in jpgs]
    times = np.array([s - t0 for s in stems], np.float32)
    half = ((stems[-1] - stems[0]) / max(len(stems) - 1, 1)) * 0.5
    y = np.zeros(len(jpgs), np.float32)
    for a_, b_ in ivs:
        y[(times + half >= a_) & (times - half < b_)] = 1.0
    return jpgs, y, (0 < y.sum() < len(y))   # mixed only


def tower_pooled(jpgs, grad):
    """Forward the (LoRA'd) tower over a fragment's frames; return mean-pooled
    [L,768] fp32, computed frame-by-frame (frames of one fragment share a size,
    so they could batch, but the processor is per-image and correctness first)."""
    feats = []
    for p_ in jpgs:
        im = Image.open(p_).convert('RGB')
        msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': im},
                                             {'type': 'text', 'text': 'describe'}]}]
        x = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                     return_dict=True, return_tensors='pt')
        x = {k: v.to(DEV) for k, v in x.items() if isinstance(v, torch.Tensor)}
        x['pixel_values'] = x['pixel_values'].to(torch.float16)
        with torch.set_grad_enabled(grad):
            out = model.model.vision_tower(pixel_values=x['pixel_values'],
                                           spatial_shapes=x['spatial_shapes'],
                                           pixel_attention_mask=x['pixel_attention_mask'],
                                           return_dict=True)
            valid = int(x['pixel_attention_mask'][0].sum())
            fh, fw = [int(v) for v in x['spatial_shapes'][0]]
            grid = out.last_hidden_state[0, :valid].reshape(fh, fw, -1)
        feats.append(grid.float().mean((0, 1)))
    return torch.stack(feats)   # [L,768] on DEV


opt = torch.optim.AdamW([p_ for p_ in model.model.vision_tower.parameters() if p_.requires_grad]
                        + list(tcn.parameters()), lr=args.lr, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)


def heldout_ap(tag, limit=None):
    model.model.vision_tower.eval(); tcn.eval()
    aps = []
    with torch.no_grad():
        for r in (eval_rows if not limit else eval_rows[:limit]):
            jpgs, y, mixed = fragment_labels(r)
            if jpgs is None or not mixed:
                continue
            s = torch.sigmoid(tcn(tower_pooled(jpgs, grad=False)[None]))[0].cpu().numpy()
            o = np.argsort(-s); ys = y[o]
            aps.append(float((np.cumsum(ys) / np.arange(1, len(ys) + 1) * ys).sum() / y.sum()))
    v = float(np.mean(aps))
    print(f'[{tag}] heldout_ap={v:.4f} ({len(aps)} fragments)', flush=True)
    return v


t0 = time.time()
hist = []
step = 0
best = (-1.0, None)
baseline_ap = None
while step < args.steps:
    model.model.vision_tower.train(); tcn.train()
    batch = [r for r in rng.choice(len(train_rows), size=args.fragments_per_step * 3)
             if (lambda r: fragment_labels(r)[2] is not None and fragment_labels(r)[2])(train_rows[r])][
        :args.fragments_per_step]
    if len(batch) < 2:
        continue
    opt.zero_grad()
    total_loss = 0.0
    for r in [train_rows[i] for i in batch]:
        jpgs, y, _ = fragment_labels(r)
        if jpgs is None:
            continue
        s = tcn(tower_pooled(jpgs, grad=True)[None])[0]
        loss = torch.nn.functional.mse_loss(torch.sigmoid(s),
                                            torch.from_numpy(y).to(DEV))
        (loss / len(batch)).backward()
        total_loss += float(loss.detach()) / len(batch)
    torch.nn.utils.clip_grad_norm_(
        [p_ for p_ in model.model.vision_tower.parameters() if p_.requires_grad] + list(tcn.parameters()), 1.0)
    opt.step(); sched.step()
    step += 1
    if step % 50 == 0:
        print(f'step {step} loss={total_loss:.4f} {time.time()-t0:.0f}s', flush=True)
    if step % args.eval_every == 0 or step == args.steps:
        v = heldout_ap(f'step{step}', None if step == args.steps else args.eval_limit)
        hist.append({'step': step, 'loss': round(total_loss, 4), 'heldout_ap': round(v, 4)})
        if baseline_ap is None and step == args.eval_every:
            baseline_ap = v
        if v > best[0]:
            best = (v, step)

out = {
    'baseline_band': '0.677-0.696 (ten frozen-tower variants)',
    'first_eval_ap': baseline_ap,
    'best_ap': round(best[0], 4), 'best_step': best[1],
    'history': hist,
    'escaped_band': bool(best[0] > 0.70),
    'note': ('an escape above 0.70 says the frozen tower was the ranking wall and '
             'LoRA is the new main line; no escape means the GIF labels cap any '
             'image-derived representation near 0.69')}
print(json.dumps({k: out[k] for k in ('first_eval_ap', 'best_ap', 'best_step', 'escaped_band')},
                 indent=1), flush=True)
args.out.write_text(json.dumps(out, indent=1))
torch.save({'lora_state': {k: v.cpu() for k, v in
                           model.model.vision_tower.state_dict().items() if 'lora' in k},
            'tcn_state': tcn.state_dict(), 'hist': hist}, args.out.with_suffix('.pt'))