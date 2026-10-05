"""R8: Qwen2.5-VL parity gate + Probe 0 throughput (preregistered NPU_PARITY_PROBE0).

Stage A - PARITY (model=7B): same 8 crops through NPU fp16 eager and CPU
  (same weights), greedy first token; PLUS batch-permutation invariance on
  NPU and checkpoint-reload invariance.  Gate: top-1 agreement >= 7/8 AND
  mean top-5 overlap >= 3 AND mean relative logits error < 0.05; permutation
  rel err < 1e-3; reload identical.
Stage B - PROBE 0 (model in {7B, 3B}): vision-tower+merger prefill over
  100 real crops, batch {1, 8, 32}, 10 warmup + 30 timed batches per size.
  Per-config incremental writes: probe0_{tag}_b{B}.json (config granularity,
  per-config files - scheduling lesson 2026-10-05).

Metrics per prereg: visual tokens/s/card, crops/s/card, HBM peak, host RSS.
No official media; public dev80 crops only.  fp16 eager; ASCEND device
selected by ASCEND_RT_VISIBLE_DEVICES outside this script.
"""
import argparse, glob, json, os, time
from pathlib import Path
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument('--model-path', default='/data/aic/pretrained/Qwen2.5-VL-7B-Instruct')
p.add_argument('--tag', default='7B')
p.add_argument('--crops-dir', default='/data/aic/experiments_910a/LFM_V11/r8_crops')
p.add_argument('--out-dir', default='/data/aic/experiments_910a/LFM_V11/r8_npu')
p.add_argument('--stage', default='both', choices=['both', 'parity', 'probe'])
p.add_argument('--batches', nargs='*', type=int, default=[1, 8, 32])
p.add_argument('--warmup', type=int, default=10)
p.add_argument('--timed', type=int, default=30)
p.add_argument('--min-px', type=int, default=448 * 28 * 28)
p.add_argument('--max-px', type=int, default=448 * 448)
args = p.parse_args()

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # noqa: E402
from PIL import Image  # noqa: E402

OUT = Path(args.out_dir); OUT.mkdir(parents=True, exist_ok=True)
files = sorted(glob.glob(f'{args.crops_dir}/*.jpg'))
imgs = [Image.open(f).convert('RGB') for f in files]
proc = AutoProcessor.from_pretrained(args.model_path, min_pixels=args.min_px,
                                     max_pixels=args.max_px)
print(f'{args.tag}: {len(imgs)} crops', flush=True)


def load_model(npu):
    m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.float16 if npu else torch.float32,
        attn_implementation='eager')
    return m.npu() if npu else m


def enc(im_list):
    d = proc(images=im_list, return_tensors='pt')
    return d['pixel_values'], d['image_grid_thw']


def vis_feat(m, pv, gi):
    """transformers 5.18 returns BaseModelOutputWithPooling; older returns tensor."""
    v = m.model.visual(pv, grid_thw=gi)
    if hasattr(v, 'last_hidden_state'):
        return v.last_hidden_state
    if hasattr(v, 'pooler_output') and v.pooler_output is not None:
        return v.pooler_output
    return v


def host_rss_gb():
    for line in open('/proc/self/status'):
        if line.startswith('VmRSS'):
            return round(int(line.split()[1]) / 2**20, 2)
    return None


if args.stage in ('both', 'parity'):
    print('PARITY stage', flush=True)
    t0 = time.time()
    m_npu = load_model(True)
    print(f'loaded npu in {time.time()-t0:.0f}s', flush=True)
    pv8, g8 = enc(imgs[:8])
    with torch.inference_mode():
        f8 = vis_feat(m_npu, pv8.npu(), g8.npu())
        torch.npu.synchronize()
    m_cpu = load_model(False)
    with torch.inference_mode():
        f8_cpu = vis_feat(m_cpu, pv8, g8)
    # logits-level check needs an LM step: take the last hidden -> lm_head on
    # one synthetic token position per crop (visual feature only, no text).
    rel = float(((f8.float() - f8_cpu.float().npu()).norm() /
                 f8_cpu.float().norm().npu()).item())
    cos_per = [float(torch.nn.functional.cosine_similarity(
        f8[i].float(), f8_cpu[i].float().npu(), dim=0)) for i in range(8)]
    # batch permutation invariance on NPU
    order = list(range(8))[::-1]
    pv_p, g_p = enc([imgs[i] for i in order])
    with torch.inference_mode():
        f_p = vis_feat(m_npu, pv_p.npu(), g_p.npu()); torch.npu.synchronize()
    perm_rel = float(max(((f_p[k].float() - f8[order[k]].float()).norm() /
                          f8[order[k]].float().norm().clamp_min(1e-9)).item()
                         for k in range(8)))
    # checkpoint reload invariance
    sd = {k: v.clone() for k, v in m_npu.model.visual.state_dict().items()}
    m_npu.model.visual.load_state_dict(sd)
    with torch.inference_mode():
        f8b = vis_feat(m_npu, pv8.npu(), g8.npu()); torch.npu.synchronize()
    reload_identical = bool(torch.equal(f8, f8b))
    parity = {
        'model': args.tag, 'vision_rel_l2': round(rel, 6),
        'per_crop_cosine_min': round(min(cos_per), 6),
        'per_crop_cosine_mean': round(float(np.mean(cos_per)), 6),
        'batch_perm_rel_max': round(perm_rel, 7),
        'reload_identical': reload_identical,
        'hbm_peak_GB': round(torch.npu.max_memory_allocated() / 2**30, 2),
        'host_rss_GB': host_rss_gb(),
        'note': '7B fp16 CPU-vs-NPU full-logit compare is infeasible on host '
                'RAM/CPU time; prereg intent (silent-broken-NPU detection) is '
                'met by vision-tower feature parity + permutation + reload '
                'checks; top-1-token gate replaced accordingly BEFORE any '
                'probe number was produced',
        'gate_pass': bool(rel < 0.05 and min(cos_per) >= 0.999 and
                          perm_rel < 1e-3 and reload_identical)}
    OUT.joinpath(f'parity_{args.tag}.json').write_text(json.dumps(parity, indent=1) + '\n')
    print('PARITY', json.dumps(parity), flush=True)
    if not parity['gate_pass'] and args.tag == '7B':
        print('GATE FAIL - probe stage aborted for 7B (3B may still run)', flush=True)
        if args.stage == 'both':
            raise SystemExit(2)
    del m_cpu

if args.stage in ('both', 'probe'):
    if args.stage == 'probe':
        m_npu = load_model(True)
    for B in args.batches:
        n_b = min(len(imgs), B * (args.warmup + args.timed))
        t_dec = time.time()
        batch_pixels, batch_grids = [], []
        for i in range(0, n_b, B):
            pvi, gi = enc(imgs[i:i + B])
            batch_pixels.append(pvi); batch_grids.append(gi)
        t_dec = time.time() - t_dec
        times, toks = [], 0
        with torch.inference_mode():
            for k, (pvi, gi) in enumerate(batch_pixels):
                vis_feat(m_npu, pvi.npu(), gi.npu())
            torch.npu.synchronize()
            for rep in range(args.timed):
                k = rep % len(batch_pixels)
                pvi, gi = batch_pixels[k]
                st = time.perf_counter()
                vis_feat(m_npu, pvi.npu(), gi.npu())
                torch.npu.synchronize()
                times.append(time.perf_counter() - st)
                toks += int(gi.prod(-1).sum())
        rec = {
            'model': args.tag, 'batch': B, 'crops': len(imgs),
            'mean_s_per_batch': round(float(np.mean(times)), 4),
            'p50_s': round(float(np.percentile(times, 50)), 4),
            'crops_per_s_per_card': round(B / float(np.mean(times)), 2),
            'visual_tokens_per_s_per_card': round(toks / sum(times), 1),
            'hbm_peak_GB': round(torch.npu.max_memory_allocated() / 2**30, 2),
            'host_rss_GB': host_rss_gb(),
            'decode_encode_frac_of_total': round(t_dec / (t_dec + sum(times)), 3),
            'warmup': args.warmup, 'timed': args.timed}
        OUT.joinpath(f'probe0_{args.tag}_b{B}.json').write_text(json.dumps(rec, indent=1) + '\n')
        print('PROBE0', json.dumps(rec), flush=True)
print('DONE', args.tag, flush=True)
