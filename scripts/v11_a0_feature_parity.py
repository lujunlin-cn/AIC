"""A0: feature-contract parity audit (V10 33.85 review, section 1.2).

The deployment feature script pools ALL output tokens (mean over the last
hidden state, padding included, FP16 cache).  The probe training script
pools VALID tokens only (pixel-attention-mask sum, spatial grid mean).
The 33.85 package scored a probe-trained head on the all-token cache.

This script measures, on the same forward pass per frame:
  e_z       = ||z_valid - z_all|| / ||z_all||
  valid/total token counts
and propagates to heads: per-frame score deltas for the SHIPPED head
(tcn_s0) and the PROBE head, plus keep-0.80 mask Jaccard under both
feature contracts.

Run on NPU (vision tower).  One card.
"""
import argparse, json, os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '8')
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--kf-root', type=Path,
                default=Path('/data/aic/semifinal_20261001/keyframes/keyframes'))
ap.add_argument('--cache-root', type=Path,
                default=Path('/data/aic/semifinal_20261001/temporal_feats'))
ap.add_argument('--model', default='/data/aic/pretrained/lfm2_5_vl_450m')
ap.add_argument('--head-a', type=Path,
                default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/tcn_full/tcn_s0.pt'),
                help='shipped head (34.73 package)')
ap.add_argument('--head-b', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V10/probe_deploy_head.pt'),
                help='probe head (33.85 package)')
ap.add_argument('--n-videos', type=int, default=16)
ap.add_argument('--frames-per-video', type=int, default=6)
ap.add_argument('--keep', type=float, default=0.80)
ap.add_argument('--cards', default='0')
ap.add_argument('--out', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/a0_feature_parity.json'))
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

DEV = 'npu'
proc = AutoProcessor.from_pretrained(args.model, min_tiles=1, max_tiles=1,
                                     max_image_tokens=256)
model = Lfm2VlForConditionalGeneration.from_pretrained(
    args.model, dtype=torch.float16, low_cpu_mem_usage=True,
    attn_implementation='eager').eval().to(DEV)


class TCN(torch.nn.Module):
    def __init__(self, d_in=768, ch=128, dils=(1, 2, 4)):
        super().__init__()
        self.proj = torch.nn.Conv1d(d_in, ch, 1)
        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv1d(ch, ch, 3, padding=d, dilation=d) for d in dils])
        self.out = torch.nn.Conv1d(ch, 1, 1)

    def forward(self, x):
        h = self.proj(x.transpose(1, 2))
        for c in self.convs:
            h = torch.nn.functional.gelu(c(h) + h)
        return self.out(h).squeeze(1)


def load_head(p):
    ck = torch.load(p, map_location='cpu', weights_only=False)
    sd = ck.get('state_dict', ck.get('tcn_state', ck))
    h = TCN(ch=ck.get('ch', 128), dils=tuple(ck.get('dils', (1, 2, 4))))
    h.load_state_dict(sd)
    return h.eval().float().to(DEV)


head_a, head_b = load_head(args.head_a), load_head(args.head_b)

vids = sorted((d for d in args.kf_root.iterdir() if d.is_dir()), key=lambda p: p.name)
step = max(1, len(vids) // args.n_videos)
vids = vids[::step][:args.n_videos]

ezs, vt_counts, heads = [], [], {'a': head_a, 'b': head_b}
mask_jac = {'a': [], 'b': []}
score_corr = {'a': [], 'b': []}
n_frames = 0

with torch.no_grad():
    for vi, vd in enumerate(vids):
        pngs = sorted(p for p in vd.glob('*.png'))[:args.frames_per_video]
        if len(pngs) < 4:
            continue
        cache_d = args.cache_root / vd.name
        z_all_rows, z_val_rows = [], []
        for p in pngs:
            im = Image.open(p).convert('RGB')
            msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': im},
                                                 {'type': 'text', 'text': 'describe'}]}]
            x = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                         return_dict=True, return_tensors='pt')
            x = {k: v.to(DEV) for k, v in x.items() if isinstance(v, torch.Tensor)}
            x['pixel_values'] = x['pixel_values'].to(torch.float16)
            out = model.model.vision_tower(pixel_values=x['pixel_values'],
                                           spatial_shapes=x['spatial_shapes'],
                                           pixel_attention_mask=x['pixel_attention_mask'],
                                           return_dict=True)
            valid = int(x['pixel_attention_mask'][0].sum())
            total = int(out.last_hidden_state.shape[1])
            fh, fw = [int(v) for v in x['spatial_shapes'][0]]
            z_all = out.last_hidden_state[0].float().mean(0)
            z_val = out.last_hidden_state[0, :valid].float().mean(0)
            ezs.append(float((z_val - z_all).norm() / (z_all.norm() + 1e-12)))
            vt_counts.append((valid, total))
            z_all_rows.append(z_all.cpu().numpy())
            z_val_rows.append(z_val.cpu().numpy())
            n_frames += 1
        # heads on both contracts
        Xa = np.stack(z_all_rows).astype(np.float32)[None]
        Xv = np.stack(z_val_rows).astype(np.float32)[None]
        Ta = torch.from_numpy(Xa).to(DEV)
        Tv = torch.from_numpy(Xv).to(DEV)
        for k, h in heads.items():
            sa = h(Ta)[0].cpu().numpy()
            sv = h(Tv)[0].cpu().numpy()
            if sa.std() > 1e-9 and sv.std() > 1e-9:
                score_corr[k].append(float(np.corrcoef(sa, sv)[0, 1]))
            ka = set(np.argsort(-sa)[:max(1, int(round(len(sa) * args.keep)))].tolist())
            kv = set(np.argsort(-sv)[:max(1, int(round(len(sv) * args.keep)))].tolist())
            mask_jac[k].append(len(ka & kv) / max(1, len(ka | kv)))
        if (vi + 1) % 4 == 0:
            print(f'{vi + 1}/{len(vids)} videos {n_frames} frames mean_ez={float(np.mean(ezs)):.5f}', flush=True)

valids, totals = zip(*vt_counts)
res = {
    'videos': len(vids), 'frames': n_frames,
    'ez_mean': round(float(np.mean(ezs)), 6),
    'ez_p50': round(float(np.percentile(ezs, 50)), 6),
    'ez_p90': round(float(np.percentile(ezs, 90)), 6),
    'ez_max': round(float(np.max(ezs)), 6),
    'valid_tokens_mean': round(float(np.mean(valids)), 1),
    'total_tokens_mean': round(float(np.mean(totals)), 1),
    'pad_share_mean': round(1 - float(np.mean(valids)) / float(np.mean(totals)), 4),
    'heads': {k: {'score_pearson_mean': round(float(np.mean(v)), 5) if v else None,
                  'keep08_jaccard_mean': round(float(np.mean(mask_jac[k])), 4),
                  'keep08_jaccard_min': round(float(np.min(mask_jac[k])), 4)}
              for k, v in score_corr.items()},
    'note': 'z_all = deployment cache contract (all-token mean, cached FP16); z_val = probe training contract (valid-token mean). FP32 here both, so e_z isolates the pooling effect, not dtype.',
}
print(json.dumps(res, indent=1))
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(res, indent=1))
