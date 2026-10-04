"""Round-5 B1: CPU-fp32 vs NPU-fp16 feature and mask parity (R5 section 7.3).

Contract copied VERBATIM from v11_round4_confirm_feats_cpu.pools_of (the
deployment feature path): processor min_tiles=1 max_tiles=1
max_image_tokens=256; chat-template input; model.model.vision_tower with
spatial_shapes + pixel_attention_mask; valid-token grid mean over the
[fh,fw] token grid; champion TCN head on the [8,768] matrix.

Checks (R5 order): identical jpg bytes -> feature eps -> head-logit eps ->
gamma > 2*eps mask rule -> |F_cpu - F_npu| <= 0.001 release gate.

Protocol: 32 frozen sources (seed 20261005) x 8 confirm-audit frames.
Output: /data/aic/experiments_910a/LFM_V11/round5_cpu_npu_parity.json
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
    import torch_npu                                        # noqa: F401  (npu backend)
except Exception:                                           # noqa: BLE001
    torch_npu = None

ap = argparse.ArgumentParser()
ap.add_argument('--conf-dir', type=Path,
                default=Path('/data/aic/experiments_910a/PHD2_FRAG_CONFIRM_V1'))
ap.add_argument('--model-dir', default='/data/aic/pretrained/lfm2_5_vl_450m')
ap.add_argument('--head', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V10/probe_deploy_head.pt'))
ap.add_argument('--n-sources', type=int, default=32)
ap.add_argument('--seed', type=int, default=20261005)
ap.add_argument('--out', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/round5_cpu_npu_parity.json'))
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
    """pools_of contract: single-image forward -> valid-token grid mean."""
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
    return g.mean((0, 1)).cpu()                                  # [D] fp32 out


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
    for name, device, dtype in (('cpu_fp32', 'cpu', torch.float32),
                                ('npu_fp16', 'npu', torch.float16)):
        cache = args.out.with_name(f'{args.out.stem}_feats_{name}.pt')
        if cache.exists():
            blob = torch.load(cache, map_location='cpu', weights_only=False)
            feats[name], shas[name] = blob['feats'], blob['shas']
            print(name, 'loaded from cache', flush=True)
            continue
        proc, model = build(args.model_dir, device, dtype)
        fl, sl = [], []
        for r, jpgs in cases:
            fl.append(torch.stack([grid_mean(proc, model, j, device)
                                   for j in jpgs]))              # [8, D]
            sl.append(sha16(jpgs))
        feats[name] = fl
        shas[name] = sl
        torch.save({'feats': fl, 'shas': sl}, cache)
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

    sel = json.loads(Path('/data/aic/external_datasets/PHD2/'
                          'annotations/selections/train.json').read_text())
    eps_feat, eps_head, gammas, mask_diff = [], [], [], 0
    f1s = {'cpu_fp32': [], 'npu_fp16': []}
    for ci, (r, _) in enumerate(cases):
        fc, fn = feats['cpu_fp32'][ci], feats['npu_fp16'][ci]
        eps_feat.append(float((fc - fn).abs().max()))
        with torch.no_grad():
            sc = hc(fc[None])[0].numpy()
            sn = hn(fn[None].to('npu'))[0].float().cpu().numpy()
        eps_head.append(float(np.abs(sc - sn).max()))
        L = len(sc)
        k = max(1, int(round(0.8 * L)))
        order = np.argsort(-sc)
        gamma = float(sc[order[k - 1]] - sc[order[k]]) if k < L else float('inf')
        gammas.append(gamma)
        if np.argsort(-sc)[:k].tolist() != np.argsort(-sn)[:k].tolist():
            mask_diff += 1
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
        'protocol': 'R5 7.3 parity, contract copied from confirm_feats_cpu.pools_of',
        'n_cases': len(cases),
        'input_sha_equal': True,
        'eps_feat_inf_max': round(float(np.max(eps_feat)), 6),
        'eps_head_inf_max': round(float(np.max(eps_head)), 6),
        'gamma_min': round(float(np.min(gammas)), 6),
        'mask_diff_cases': mask_diff,
        'gamma_gt_2eps_all': bool(all(g > 2 * e for g, e in zip(gammas, eps_head))),
        'F_cpu': round(f_cpu, 5), 'F_npu': round(f_npu, 5),
        'abs_F_diff': round(abs(f_cpu - f_npu), 5),
        'release_gate': 'abs_F_diff <= 0.001 on this frozen subset; masks must '
                        'not differ beyond the gamma>2eps float rule',
        'gate_pass': bool(abs(f_cpu - f_npu) <= 0.001 and mask_diff == 0),
    }
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
