"""Compare the shipped (tcn_s0) and probe (frozen) temporal heads under ONE
evaluation protocol, after the platform said -0.88 for +0.12 AP.

The platform result means "AP on PHD2 held-out fragments" is not the
deployment metric.  First question: was the 0.626 -> 0.746 comparison even
same-protocol?  The shipped head's 0.626 was a deployment-simulation F1 from
the V9 round, never scored under the probe protocol.  This script scores any
TCN checkpoint under the exact probe protocol (same 1,954 mixed held-out
fragments, features computed fresh from the frozen tower) so both heads get
one number each.

CPU-only mask overlap analysis lives in this file too (--stage overlap):
how much the two keep masks actually differ on the 426 videos.
"""
import argparse, json, os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '8')
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--stage', choices=['ap', 'overlap'], required=True)
ap.add_argument('--ckpt', type=Path, default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/tcn_full/tcn_s0.pt'))
ap.add_argument('--model', default='/data/aic/pretrained/lfm2_5_vl_450m')
ap.add_argument('--frames-root', type=Path, default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/frames'))
ap.add_argument('--index', type=Path, default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/index_clean.jsonl'))
ap.add_argument('--eval-sources', type=Path, default=Path('/data/aic/experiments_910a/LFM_V10/eval_sources_50.json'))
ap.add_argument('--selections', type=Path, default=Path('/data/aic/external_datasets/PHD2/annotations/selections/train.json'))
ap.add_argument('--cards', default='2')
ap.add_argument('--new-mask', type=Path, default=Path('/data/aic/semifinal_20261001/temporal_probe/predictions.jsonl'))
ap.add_argument('--old-mask', type=Path, default=Path('/data/aic/semifinal_20261001/submissions/LFM_V9_B3_VISIONONLY_SEMIFINAL/predictions.jsonl'))
ap.add_argument('--out', type=Path, default=Path('/data/aic/experiments_910a/LFM_V10/head_compare.json'))
args = ap.parse_args()

if args.stage == 'overlap':
    new = {json.loads(l)['video_id']: json.loads(l) for l in args.new_mask.read_text().splitlines() if l.strip()}
    old = {json.loads(l)['video_id']: json.loads(l) for l in args.old_mask.read_text().splitlines() if l.strip()}
    jac, changed, both = [], 0, 0
    for vid in sorted(new, key=int):
        nf = {p['frame'] for p in new[vid]['predictions']}
        of = {p['frame'] for p in old[vid]['predictions']}
        u = nf | of
        jac.append(len(nf & of) / len(u) if u else 1.0)
        changed += (nf != of)
        both += 1
    res = {'videos': both, 'mask_frameset_changed': changed,
           'share_changed': round(changed / both, 4),
           'jaccard_mean': round(float(sum(jac) / len(jac)), 4),
           'jaccard_p10': round(float(sorted(jac)[len(jac) // 10]), 4),
           'jaccard_min': round(min(jac), 4)}
    print(json.dumps(res, indent=1))
    args.out.write_text(json.dumps(res, indent=1))
    raise SystemExit

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
sel = json.loads(args.selections.read_text())
eval_set = set(json.loads(args.eval_sources.read_text()))
proc = AutoProcessor.from_pretrained(args.model, min_tiles=1, max_tiles=1, max_image_tokens=256)
model = Lfm2VlForConditionalGeneration.from_pretrained(
    args.model, dtype=torch.float16, low_cpu_mem_usage=True,
    attn_implementation='eager').eval().to(DEV)


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


ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
sd = ck.get('state_dict', ck.get('tcn_state', ck))
head = TCN()
head.load_state_dict(sd)
head.eval().float().to(DEV)

rows = [json.loads(l) for l in args.index.read_text().splitlines() if l.strip()]
eval_rows = [r for r in rows if r['src'] in eval_set]


def fragment_labels(r):
    t0 = float(r['t0'])
    ivs = []
    for recs in sel.get(r['src'], {}).values():
        for rec in recs:
            a_, b_ = float(rec['t0']) - t0, float(rec['t1']) - t0
            if b_ > a_:
                ivs.append((a_, b_))
    kdir = args.frames_root / r['video_id']
    jpgs = sorted(kdir.glob('*.jpg'), key=lambda p: float(p.stem))
    if len(jpgs) < 4 or not ivs:
        return None, None
    stems = [float(p.stem) for p in jpgs]
    times = np.array([s - t0 for s in stems], np.float32)
    half = ((stems[-1] - stems[0]) / max(len(stems) - 1, 1)) * 0.5
    y = np.zeros(len(jpgs), np.float32)
    for a_, b_ in ivs:
        y[(times + half >= a_) & (times - half < b_)] = 1.0
    return jpgs, y


def tower_feats(jpgs):
    feats = []
    with torch.no_grad():
        for p_ in jpgs:
            im = Image.open(p_).convert('RGB')
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
            fh, fw = [int(v) for v in x['spatial_shapes'][0]]
            feats.append(out.last_hidden_state[0, :valid].reshape(fh, fw, -1).float().mean((0, 1)))
    return torch.stack(feats)


aps, n_eval = [], 0
with torch.no_grad():
    for r in eval_rows:
        jpgs, y = fragment_labels(r)
        if jpgs is None or not (0 < y.sum() < len(y)):
            continue
        s = torch.sigmoid(head(tower_feats(jpgs)[None]))[0].cpu().numpy()
        o = np.argsort(-s); ys = y[o]
        aps.append(float((np.cumsum(ys) / np.arange(1, len(ys) + 1) * ys).sum() / y.sum()))
        n_eval += 1
        if n_eval % 200 == 0:
            print(f'{n_eval} fragments, running AP {float(np.mean(aps)):.4f}', flush=True)
res = {'ckpt': str(args.ckpt), 'protocol': 'probe: 1,954 mixed held-out fragments, fresh tower features',
       'n_fragments': n_eval, 'ap': round(float(np.mean(aps)), 4)}
print(json.dumps(res, indent=1), flush=True)
args.out.write_text(json.dumps(res, indent=1))
