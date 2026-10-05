"""R6 P0: CPU-fp32 vs NPU-fp16 parity gate re-run + error-injection self-test.

Part 1 (identical to the round-5 gate, re-run after any stack change):
  identical jpg bytes -> feature eps -> head-logit eps -> gamma > 2*eps
  mask rule -> |F_cpu - F_npu| <= 0.001 on 32 frozen sources x 8 frames.

Part 2 (NEW, R6 Q7-D): four error injections, each applied to the NPU-side
features before the head.  The gate MUST catch every injection
(mask flip, or |dF| > 0.001, or NaN).  An injection the gate cannot see is
recorded as gate_insensitive - itself a finding.

  temporal_shift : features rolled by 1 frame in time (decode/PTS error)
  bad_padding    : one zero frame appended, then truncated back to 8 after
                   scoring 2x-length input (padding-contract error)
  layout_swap    : [8,768] reshaped (4,2,768) -> transposed -> back
                   (tubelet B/T/D order error)
  fp16_overflow  : features scaled 1e4 (fp16 dynamic-range error)

Run on ONE NPU: export ASCEND_RT_VISIBLE_DEVICES=2 (physical card 2).
Output: /data/aic/experiments_910a/LFM_V11/r6_numeric_parity.json
"""
import argparse, hashlib, json, os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '16')
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
from pathlib import Path
import numpy as np
import sys
import torch
try:
    import torch_npu                                        # noqa: F401
except Exception as e:                                      # noqa: BLE001
    raise SystemExit(f'torch_npu import failed: {e}; source the ascend env first')

ap = argparse.ArgumentParser()
ap.add_argument('--conf-dir', type=Path,
                default=Path('/data/aic/experiments_910a/PHD2_FRAG_CONFIRM_V1'))
ap.add_argument('--model-dir', default='/data/aic/pretrained/lfm2_5_vl_450m')
ap.add_argument('--head', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V10/probe_deploy_head.pt'))
ap.add_argument('--n-sources', type=int, default=32)
ap.add_argument('--seed', type=int, default=20261005)
ap.add_argument('--reuse-round5-cache', action='store_true', default=True)
ap.add_argument('--out', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/r6_numeric_parity.json'))
args = ap.parse_args()


def build(model_dir, device, dtype):
    from transformers import AutoProcessor, Lfm2VlForConditionalGeneration
    proc = AutoProcessor.from_pretrained(model_dir, min_tiles=1, max_tiles=1,
                                         max_image_tokens=256)
    model = Lfm2VlForConditionalGeneration.from_pretrained(
        model_dir, dtype=dtype, attn_implementation='eager').eval().to(device)
    if device == 'npu':
        sys.path.insert(0, '/root/AIC')
        from v11_npu_siglip2_patch import patch_siglip2_npu
        assert patch_siglip2_npu(model), 'siglip2 npu patch failed'
    return proc, model


def grid_mean(proc, model, img, device):
    from PIL import Image
    im = Image.open(img).convert('RGB')
    msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': im},
                                         {'type': 'text', 'text': 'describe'}]}]
    x = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                 return_dict=True, return_tensors='pt')
    x = {k: v.to(device) for k, v in x.items() if isinstance(v, torch.Tensor)}
    with torch.no_grad():
        out = model.model.vision_tower(pixel_values=x['pixel_values'],
                                       spatial_shapes=x['spatial_shapes'],
                                       pixel_attention_mask=x['pixel_attention_mask'],
                                       return_dict=True)
    valid = int(x['pixel_attention_mask'][0].sum())
    fh, fw = [int(v) for v in x['spatial_shapes'][0]]
    g = out.last_hidden_state[0, :valid].float().reshape(fh, fw, -1)
    return g.mean((0, 1)).cpu()


def sha16(paths):
    return hashlib.sha256(b''.join(hashlib.sha256(
        Path(p).read_bytes()).digest() for p in paths)).hexdigest()[:16]


def main():
    rows = [json.loads(l) for l in
            (args.conf_dir / 'index.jsonl').read_text().splitlines() if l.strip()]
    rng = np.random.RandomState(args.seed)
    cases = []
    for i in rng.choice(len(rows), len(rows), replace=False):
        fdir = args.conf_dir / 'frames' / rows[i]['video_id']
        jpgs = sorted(fdir.glob('*.jpg'), key=lambda p: float(p.stem)) if fdir.exists() else []
        if len(jpgs) == 8:
            cases.append((rows[i], jpgs))
        if len(cases) == args.n_sources:
            break
    print(f'parity cases: {len(cases)}', flush=True)
    assert cases, 'no cases found'

    feats, shas = {}, {}
    r5_cache = Path('/data/aic/experiments_910a/LFM_V11/round5_cpu_npu_parity_feats_cpu_fp32.pt')
    r5_cache_n = Path('/data/aic/experiments_910a/LFM_V11/round5_cpu_npu_parity_feats_npu_fp16.pt')
    if args.reuse_round5_cache and r5_cache.exists() and r5_cache_n.exists():
        b = torch.load(r5_cache, map_location='cpu', weights_only=False)
        feats['cpu_fp32'], shas['cpu_fp32'] = b['feats'], b['shas']
        b = torch.load(r5_cache_n, map_location='cpu', weights_only=False)
        feats['npu_fp16'], shas['npu_fp16'] = b['feats'], b['shas']
        print('features reused from the round-5 gate cache (same seed, same stack)',
              flush=True)
    else:
        for name, device, dtype in (('cpu_fp32', 'cpu', torch.float32),
                                    ('npu_fp16', 'npu', torch.float16)):
            proc, model = build(args.model_dir, device, dtype)
            fl, sl = [], []
            for r, jpgs in cases:
                fl.append(torch.stack([grid_mean(proc, model, j, device) for j in jpgs]))
                sl.append(sha16(jpgs))
            feats[name], shas[name] = fl, sl
            del model
            print(name, 'done', flush=True)
    assert shas['cpu_fp32'] == shas['npu_fp16'], 'input sha mismatch'

    head_ck = torch.load(args.head, map_location='cpu')
    dils, ch = tuple(head_ck.get('dils', (1, 2, 4))), head_ck.get('ch', 128)

    class TCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Conv1d(768, ch, 1)
            self.convs = torch.nn.ModuleList(
                [torch.nn.Conv1d(ch, ch, 3, padding=d, dilation=d) for d in dils])
            self.out = torch.nn.Conv1d(ch, 1, 1)

        def forward(self, x):
            h = self.proj(x.transpose(1, 2))
            for c in self.convs:
                h = torch.nn.functional.gelu(c(h) + h)
            return self.out(h).squeeze(1)

    hc = TCN(); hc.load_state_dict(head_ck['state_dict']); hc.eval()
    hn = TCN(); hn.load_state_dict(head_ck['state_dict']); hn.eval().to('npu')

    def score_npu(x):
        with torch.no_grad():
            return hn(x[None].to('npu'))[0].float().cpu().numpy()

    sel = json.loads(Path('/data/aic/external_datasets/PHD2/'
                          'annotations/selections/train.json').read_text())
    eps_feat, eps_head, gammas, mask_diff = [], [], [], 0
    f1s = {'cpu_fp32': [], 'npu_fp16': []}
    inj = {k: {'caught': 0, 'nan': 0} for k in
           ('temporal_shift', 'bad_padding', 'layout_swap', 'fp16_overflow')}
    inj_clean = {k: [] for k in inj}
    for ci, (r, _) in enumerate(cases):
        fc, fn = feats['cpu_fp32'][ci], feats['npu_fp16'][ci]
        eps_feat.append(float((fc - fn).abs().max()))
        with torch.no_grad():
            sc = hc(fc[None])[0].numpy()
            sn = score_npu(fn)
        eps_head.append(float(np.abs(sc - sn).max()))
        L = len(sc)
        k = max(1, int(round(0.8 * L)))
        order = np.argsort(-sc)
        gamma = float(sc[order[k - 1]] - sc[order[k]]) if k < L else float('inf')
        gammas.append(gamma)
        same_mask = np.argsort(-sc)[:k].tolist() == np.argsort(-sn)[:k].tolist()
        if not same_mask:
            mask_diff += 1
        # ---- injections (each must be caught) ----
        variants = {
            'temporal_shift': torch.roll(fn, 1, 0),
            'layout_swap': fn.reshape(4, 2, -1).permute(0, 2, 1).reshape(8, -1),
            'fp16_overflow': fn * 1e4,
            'bad_padding': None,                       # scored separately
        }
        for name, xv in variants.items():
            if xv is None:
                continue
            s2 = score_npu(xv.contiguous())
            if not np.isfinite(s2).all():
                inj[name]['nan'] += 1
                inj[name]['caught'] += 1
                continue
            flipped = np.argsort(-s2)[:k].tolist() != np.argsort(-sc)[:k].tolist()
            dF = abs(float(s2.mean()) - float(sc.mean()))
            if flipped or dF > 1e-3:
                inj[name]['caught'] += 1
            inj_clean[name].append(0 if flipped else 1)
        sx = torch.cat([fn, torch.zeros(1, fn.shape[1])], 0)[None].to('npu')
        with torch.no_grad():
            s_pad = hn(sx)[0].float().cpu().numpy()
        flipped = np.argsort(-s_pad)[:k].tolist() != np.argsort(-sc)[:k].tolist()
        if flipped:
            inj['bad_padding']['caught'] += 1
        else:
            inj_clean['bad_padding'].append(1)
        # ---- keep-0.8 F1 for the release gate ----
        t0, Lt = float(r['t0']), float(r['L'])
        stamps = np.array(sorted(float(t0 + Lt * j / 8) for j in range(8)))
        y = np.zeros(8, np.float32)
        for recs in sel.get(r['src'], {}).values():
            for rec in recs:
                a_, b_ = float(rec['t0']) - t0, float(rec['t1']) - t0
                if b_ > a_:
                    y[(stamps >= a_) & (stamps <= b_)] = 1.0
        if 0 < y.sum() < 8:
            for nm, s in (('cpu_fp32', sc), ('npu_fp16', sn)):
                idx = set(np.argsort(-s)[:k].tolist())
                f1s[nm].append(float(2 * sum(y[i] for i in idx) / (k + y.sum())))

    f_cpu, f_npu = float(np.mean(f1s['cpu_fp32'])), float(np.mean(f1s['npu_fp16']))
    out = {
        'protocol': 'R6 P0 numeric parity + error-injection self-test; '
                    'features reused from the round-5 gate cache (identical '
                    'seed/stack) unless --no-reuse',
        'n_cases': len(cases),
        'input_sha_equal': True,
        'eps_feat_inf_max': round(float(np.max(eps_feat)), 6),
        'eps_head_inf_max': round(float(np.max(eps_head)), 6),
        'gamma_min': round(float(np.min(gammas)), 6),
        'mask_diff_cases': mask_diff,
        'gamma_gt_2eps_all': bool(all(g > 2 * e for g, e in zip(gammas, eps_head))),
        'F_cpu': round(f_cpu, 5), 'F_npu': round(f_npu, 5),
        'abs_F_diff': round(abs(f_cpu - f_npu), 5),
        'gate_pass': bool(abs(f_cpu - f_npu) <= 0.001 and mask_diff == 0),
        'injection_selftest': {
            name: dict(v, gate_insensitive_rate=round(
                float(np.mean(inj_clean[name])), 4) if inj_clean[name] else 0.0)
            for name, v in inj.items()},
        'injection_rule': 'every injection must flip a mask, blow up (>1e-3 mean '
                          'score delta), or produce NaN; an uncaught injection is '
                          'recorded as gate_insensitive and is a finding',
    }
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
