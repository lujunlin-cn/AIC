"""V7 E0: real training smoke on 910A — forward, loss, backward, optimizer update,
save, reload, re-forward.  Verifies every planned-trainable group has non-zero
gradients and actually changes, and measures iter/s + peak memory.

Task for smoke only: predict the free-axis offset of the mean annotator crop on
3:1 tasks (GT files of the frozen dev manifest).  The production candidate-utility
head is a separate script; here we prove the training path works end to end.
"""
import argparse, json, time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='/data/aic/pretrained/lfm2_5_vl_450m')
ap.add_argument('--cards', default='3')
ap.add_argument('--steps', type=int, default=60)
ap.add_argument('--n-frames', type=int, default=16)
ap.add_argument('--unfreeze-last-n', type=int, default=2)
ap.add_argument('--lr-head', type=float, default=3e-4)
ap.add_argument('--lr-backbone', type=float, default=1e-5)
ap.add_argument('--output', type=Path, required=True)
args = ap.parse_args()

import os  # noqa: E402
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = args.cards
import torch  # noqa: E402
import torch_npu  # noqa: E402,F401
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False
from PIL import Image  # noqa: E402
from transformers import AutoProcessor, Lfm2VlForConditionalGeneration  # noqa: E402

FR = Path('/data/aic/experiments/QWEN_RV200_KEYFRAMES/d1')
ANN = Path('/data/aic/experiments_910a/LFM450_EVAL_V1/annotations')

proc = AutoProcessor.from_pretrained(args.model, min_tiles=1, max_tiles=1, max_image_tokens=256)
model = Lfm2VlForConditionalGeneration.from_pretrained(
    args.model, dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation='eager').eval().to('npu')

# ---- freeze everything, then open the planned groups ----
for p in model.parameters():
    p.requires_grad_(False)
vis = model.model.vision_tower  # Lfm2VlModel wraps vision_tower/projector/language
enc = vis.vision_model.encoder.layers
n_blocks = len(enc)
for blk in list(enc)[-args.unfreeze_last_n:]:
    for p in blk.parameters():
        p.requires_grad_(True)
        p.data = p.data.float()  # FP32 master weights for trainable backbone
for p in model.model.multi_modal_projector.parameters():
    p.requires_grad_(True)
    p.data = p.data.float()
# the smoke head (registered on the module so save/load covers it)
D = model.config.text_config.hidden_size


class OffsetHead(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d, 256), torch.nn.GELU(), torch.nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


model.offset_head = OffsetHead(D).float().to('npu')
for p in model.offset_head.parameters():
    p.requires_grad_(True)

trainable = [n for n, p in model.named_parameters() if p.requires_grad]
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

# ---- data: 3:1 frames with 6-annotator GT ----
frames = []
for vid in ['031', '032', '033', '034']:
    kfs = sorted(int(p.stem) for p in (FR / vid).glob('*.png'))
    for kf in kfs[:args.n_frames // 4]:
        gt = np.maximum(np.stack([np.loadtxt(ANN / f'annotator_{i}' / f'{vid}_3-1.txt',
                                             delimiter=',') for i in range(1, 7)]), 0)  # (6, n, 4)
        y1, y2 = gt[:, :, 1].mean(0), gt[:, :, 3].mean(0)
        H = 360.0
        target = float(((y1[kf] + y2[kf]) / 2 / H))
        frames.append((vid, kf, min(max(target, 0.0), 1.0)))
print(f'DEBUG frames={len(frames)} trainable_params={n_train} groups={len(trainable)}', flush=True)


def prep(vid, kf):
    im = Image.open(FR / vid / f'{kf}.png').convert('RGB')
    msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': im},
                                         {'type': 'text', 'text': 'describe'}]}]
    x = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                 return_dict=True, return_tensors='pt')
    return {k: v.to('npu') for k, v in x.items() if isinstance(v, torch.Tensor)}


def forward_features(x):
    out = model.model.vision_tower(pixel_values=x['pixel_values'], spatial_shapes=x['spatial_shapes'],
                             pixel_attention_mask=x['pixel_attention_mask'], return_dict=True)
    b = 0
    valid = int(x['pixel_attention_mask'][b].sum())
    fh, fw = [int(v) for v in x['spatial_shapes'][b]]
    feat = out.last_hidden_state[b, :valid].reshape(fh, fw, -1)
    feat = feat[:, : (fw // 2) * 2]
    feat = feat[: (fh // 2) * 2]
    proj = model.model.multi_modal_projector(feat.unsqueeze(0).permute(0, 3, 1, 2) if False else feat.unsqueeze(0))
    tok = proj.reshape(-1, proj.shape[-1])
    return tok


# AmpUpdateScale operator is unsupported on ascend910 -> pure FP32 training path
# (vision tower 85.8M + projector 8.4M + head in FP32; LM tower untouched).
model.model.vision_tower = model.model.vision_tower.float()
model.model.multi_modal_projector = model.model.multi_modal_projector.float()
model.offset_head = model.offset_head.float()
scaler = None

groups = [model.offset_head.parameters(), model.model.multi_modal_projector.parameters()] + \
         [list(blk.parameters()) for blk in list(enc)[-args.unfreeze_last_n:]]
opt = torch.optim.AdamW([
    {'params': list(model.offset_head.parameters()), 'lr': args.lr_head},
    {'params': list(model.model.multi_modal_projector.parameters()), 'lr': 3e-5},
    {'params': [p for blk in list(enc)[-args.unfreeze_last_n:] for p in blk.parameters()], 'lr': args.lr_backbone},
], weight_decay=0.01)

snap0 = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
model.train()
hist, t_data = [], 0.0
t0 = time.time()
for step in range(args.steps):
    vid, kf, target = frames[step % len(frames)]
    x = prep(vid, kf)
    tok = forward_features(x)
    pred = model.offset_head(tok.float().mean(0))
    loss = torch.nn.functional.huber_loss(pred.unsqueeze(0), torch.tensor([target], device='npu'))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if step == 0:
        grad_check = {}
        for gi, g in enumerate(groups):
            got = [p for p in g if p.grad is not None]
            grad_check[f'group{gi}'] = {'n_tensors': len(got),
                                        'nonzero_grad_tensors': sum(int(p.grad.abs().sum() > 0) for p in got)}
        print('GRAD_CHECK', json.dumps(grad_check), flush=True)
    gnorm_before = torch.nn.utils.clip_grad_norm_([p for g in groups for p in g], 1.0)
    opt.step()
    hist.append(float(loss))
    if step == 0 or (step + 1) % 20 == 0:
        nz = {gi: float(sum((p.grad.abs().sum() > 0).sum() for p in g if p.grad is not None)) for gi, g in enumerate(groups)}
        print(f'STEP {step} loss={float(loss):.4f} gnorm={float(gnorm_before):.3f} nonzero_grad_groups={nz}', flush=True)
train_s = time.time() - t0

# ---- verify parameters actually changed ----
changed = {n: float((snap0[n] - p.detach()).abs().max()) for n, p in model.named_parameters() if p.requires_grad}
n_changed = sum(1 for v in changed.values() if v > 0)

# ---- save / reload / re-forward ----
ckpt = {'trainable': {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}}
ck_path = args.output.parent / 'smoke_ckpt.pt'
torch.save(ckpt, ck_path)
before = {}
model.eval()
with torch.no_grad():
    x = prep(frames[0][0], frames[0][1])
    out_before = model.offset_head(forward_features(x).float().mean(0))
reloaded = torch.load(ck_path, weights_only=True)
for n, t in reloaded['trainable'].items():
    dict(model.named_parameters())[n].data.copy_(t.to('npu'))
with torch.no_grad():
    out_after = model.offset_head(forward_features(x).float().mean(0))
reload_diff = float((out_before - out_after).abs().max())

summ = {
    'trainable_params': n_train, 'trainable_tensors': len(trainable),
    'steps': args.steps, 'loss_first5': [round(h, 4) for h in hist[:5]],
    'loss_last5': [round(h, 4) for h in hist[-5:]],
    'params_changed_groups': f'{n_changed}/{len(changed)}',
    'max_abs_change_sample': dict(list(sorted(changed.items(), key=lambda kv: -kv[1]))[:4]),
    'grad_check': grad_check, 'reload_forward_max_diff': reload_diff,
    'train_s': round(train_s, 1), 's_per_iter': round(train_s / args.steps, 3),
    'peak_mem_gib': round(torch.npu.max_memory_allocated() / 2**30, 2),
    'unfreeze_last_n': args.unfreeze_last_n, 'n_encoder_blocks': n_blocks,
    'processor': 'min_tiles=1 max_tiles=1 max_image_tokens=256',
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(summ, indent=1) + '\n')
print('SUMMARY', json.dumps(summ), flush=True)
