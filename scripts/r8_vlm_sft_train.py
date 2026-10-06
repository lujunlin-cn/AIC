"""R8 VLM_SFT formal training (preregistered: reports/r8/preregistration.yaml
VLM_SFT; all prereqs passed: NPU_PARITY_PROBE0 PASS, LABEL_PROVENANCE clean,
C_M_E closed, stability probe DONE 8.69 s/step no NaN).

Two lr configs from the preregistered grid run IN PARALLEL on separate cards
(parallel-first rule): launch one process per lr with --card 2 / --card 3.
Everything else identical, single seed 510 per config (no seed grid
preregistered for this arm; stability probe already used this stack).

Frozen protocol (recorded before any run):
  data     full train240 pool = 1,920 rows (public LIVE release, single
           annotator raw_single_box labels); dev80 = 640 rows for readout
  model    Qwen2.5-VL-7B, LoRA r=8 LANGUAGE q/v only, vision tower frozen
  step     micro-batch 1, grad accum 8 (single-image forward: multi-image
           concat is quadratic on this stack - Probe 0), AdamW, clip 1.0
  epochs   4 (decision recorded 2026-10-06 before launch: ~2.9 h per config,
           both configs inside the 12 NPU-hour cap; per-epoch dev curve kept,
           epoch selection is a dev-selection to be re-confirmed on the
           fresh pool, never claimed raw)
  readout  dev80 constrained candidate-ID over the 36-shortlist, greedy
           2-token decode, IoU = u[sl[pick]] (u IS the per-candidate IoU,
           same audited shards as C/M/E), invalid/unparsable pick = IoU 0
           and counted; video-macro aggregation identical to C/M/E
  control  content-permutation control after the LAST epoch: shortlist
           order re-shuffled by a fixed rng; prediction must follow content
           (perm[p_new] == p_old) well above the 1/36 chance rate
  writes   out JSON rewritten after every epoch (incremental); LoRA adapter
           saved per epoch (no resume support: a crashed config reruns,
           completed epochs stay in the JSON)

Gate (computed offline in the merge script, NOT here): dev macro IoU -
B3 CPU baseline 0.53108 >= +0.02 with source-level paired CI lower > 0 and
the permutation control passing.
"""
import argparse, csv, glob, json, os, random, re, sys, time
from pathlib import Path
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--model-path', default='/data/aic/pretrained/Qwen2.5-VL-7B-Instruct')
ap.add_argument('--freeze-csv', default='/data/aic/experiments_910a/LFM_V11/r6_s_pool_freeze_rows.csv')
ap.add_argument('--shard-glob', default='/data/aic/experiments_910a/LFM_V11/r6_s_encode_shard*.pt')
ap.add_argument('--frames-dir', default='/data/aic/experiments_910a/LFM_V11/r8_sft_frames')
ap.add_argument('--out', required=True)
ap.add_argument('--adapter-dir', required=True)
ap.add_argument('--card', default=None, help='set ASCEND_RT_VISIBLE_DEVICES before torch_npu import')
ap.add_argument('--epochs', type=int, default=4)
ap.add_argument('--accum', type=int, default=8)
ap.add_argument('--lr', type=float, required=True)
ap.add_argument('--lora-r', type=int, default=8)
ap.add_argument('--seed', type=int, default=510)
args = ap.parse_args()

for _v in ('OMP_NUM_THREADS',):
    os.environ.setdefault(_v, '8')
if args.card:
    os.environ['ASCEND_RT_VISIBLE_DEVICES'] = args.card
import torch  # noqa: E402
import torch_npu  # noqa: E402,F401
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False
assert torch.npu.device_count() > 0, 'no NPU visible - refusing CPU fallback'
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402

# ---------------- data: frames + shortlist boxes + targets ----------------
u_map, sl_map, box_map = {}, {}, {}
for sp in sorted(glob.glob(args.shard_glob)):
    d = torch.load(sp, map_location='cpu', weights_only=False)
    u_map.update(d['u']); sl_map.update(d['shortlist']); box_map.update(d['crop_boxes'])
FD = Path(args.frames_dir)

rows_tr, rows_dv = [], []
for r in csv.DictReader(open(args.freeze_csv)):
    key = f"{r['vid']}|{r['frame']}|{r['ratio']}"
    if key not in u_map:
        continue
    fp = FD / f"{r['vid']}_f{r['frame']}.jpg"
    if not fp.exists():
        continue
    sl = [int(c) for c in sl_map[key]]
    boxes = box_map[key]
    if len(boxes) != len(sl):
        continue
    u = np.asarray(u_map[key], np.float32)
    if r['pool'] == 'train240':
        rows_tr.append({'img': str(fp), 'boxes': [boxes[j] for j in range(len(sl))],
                        'target': int(np.argmax(u[sl]))})
    elif r['pool'] == 'dev80':
        rows_dv.append({'img': str(fp), 'boxes': [boxes[j] for j in range(len(sl))],
                        'u': u, 'sl': np.asarray(sl, int),
                        'key': key, 'vid': r['vid']})
print(f'train {len(rows_tr)} dev {len(rows_dv)}', flush=True)
assert len(rows_tr) >= 1900 and len(rows_dv) >= 630, 'frame prep incomplete - abort'


def build_prompt(boxes):
    parts = [f'{i}:({b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f})'
             for i, b in enumerate(boxes)]
    return ('This frame lists candidate crop windows: ' + ', '.join(parts) +
            '. Which window best frames the subject? Reply with one number.')


# ---------------- model: LoRA r8 language q/v only ----------------
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.model_path, torch_dtype=torch.float16,
    attn_implementation='eager').to('npu')
proc = AutoProcessor.from_pretrained(args.model_path, min_pixels=448 * 28 * 28,
                                     max_pixels=448 * 448)
lcfg = LoraConfig(
    r=args.lora_r, lora_alpha=16, lora_dropout=0.0, bias='none',
    target_modules=r'.*language_model.*(q_proj|v_proj)$',
    task_type='CAUSAL_LM')
model.enable_input_require_grads()
model = get_peft_model(model, lcfg)
model.print_trainable_parameters()
train_names = [n for n, p_ in model.named_parameters() if p_.requires_grad]
assert not [n for n in train_names if 'visual' in n], 'LoRA leaked into vision tower'
opt = torch.optim.AdamW([p_ for p_ in model.parameters() if p_.requires_grad], lr=args.lr)
torch.manual_seed(args.seed)


def forward_inputs(sample, boxes=None):
    msgs = [{'role': 'user', 'content': [
        {'type': 'image', 'image': sample['img']},
        {'type': 'text', 'text': build_prompt(sample['boxes'] if boxes is None else boxes)}]}]
    x = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                 return_dict=True, return_tensors='pt')
    return {k: (v.to('npu') if isinstance(v, torch.Tensor) else v) for k, v in x.items()}


def one_step(sample):
    x = forward_inputs(sample)
    ans = proc.tokenizer(str(sample['target']), return_tensors='pt').input_ids.to('npu')
    input_ids = torch.cat([x['input_ids'], ans], 1)
    labels = torch.cat([torch.full_like(x['input_ids'], -100), ans], 1)
    x['input_ids'] = input_ids
    if 'attention_mask' in x:
        x['attention_mask'] = torch.ones_like(input_ids)
    # processor-derived length-coupled fields must not outlive the concat
    # (get_rope_index indexes them with attention_mask)
    for stale in ('token_type_ids', 'mm_token_type_ids'):
        x.pop(stale, None)
    x['labels'] = labels
    return model(**x).loss


def parse_pick(txt):
    m = re.findall(r'\d+', txt)
    return int(m[-1]) if m else -1


@torch.no_grad()
def decode_pick(sample, boxes=None):
    x = forward_inputs(sample, boxes)   # generation keeps ALL processor fields
    out = model.generate(**x, max_new_tokens=2, do_sample=False)
    txt = proc.tokenizer.decode(out[0, x['input_ids'].shape[1]:], skip_special_tokens=True)
    return parse_pick(txt)


def macro_per_vid(iou_rows, vids):
    per = defaultdict(list)
    for x, v in zip(iou_rows, vids):
        per[v].append(float(x))
    vs = sorted(per)
    return np.array([np.mean(per[v]) for v in vs]), vs


@torch.no_grad()
def eval_dev(tag):
    model.eval()
    picks = [decode_pick(s) for s in rows_dv]
    iou_rows, n_bad = [], 0
    for s, pk in zip(rows_dv, picks):
        if 0 <= pk < len(s['sl']):
            iou_rows.append(float(s['u'][int(s['sl'][pk])]))
        else:
            iou_rows.append(0.0); n_bad += 1
    macro, vids = macro_per_vid(iou_rows, [s['vid'] for s in rows_dv])
    hist = defaultdict(int)
    for pk in picks:
        hist[pk] += 1
    model.train()
    return ({'tag': tag, 'dev_macro_iou': round(float(macro.mean()), 5),
             'per_vid_iou': [round(float(x), 5) for x in macro],
             'vids': vids, 'n_invalid_pick': n_bad,
             'pick_hist': {str(k): v for k, v in sorted(hist.items())}}, picks)


@torch.no_grad()
def perm_control(base_picks):
    """Re-shuffle shortlist order (fixed rng); prediction must follow content."""
    rng = np.random.RandomState(88)
    model.eval()
    agree = []
    for s, p_old in zip(rows_dv, base_picks):
        perm = rng.permutation(len(s['boxes']))
        p_new = decode_pick(s, boxes=[s['boxes'][j] for j in perm])
        if 0 <= p_old < len(perm) and 0 <= p_new < len(perm):
            agree.append(int(perm[p_new] == p_old))
    model.train()
    return {'agree_rate': round(float(np.mean(agree)), 4) if agree else None,
            'n': len(agree), 'chance': round(1.0 / 36, 4)}


rec = {'protocol': 'R8 VLM_SFT formal; LoRA r8 language q/v; vision frozen; '
                   'single-image forward accum 8; dev80 candidate-ID readout; '
                   'IoU = u[sl[pick]] video-macro (C/M/E aggregation)',
       'model': 'Qwen2.5-VL-7B', 'lr': args.lr, 'seed': args.seed,
       'epochs': args.epochs, 'accum': args.accum,
       'n_train': len(rows_tr), 'n_dev': len(rows_dv),
       'card': args.card, 'epochs_log': [], 'status': 'RUNNING'}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.adapter_dir).mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(rec, indent=1) + '\n')

t0 = time.time()
for ep in range(1, args.epochs + 1):
    te = time.time()
    model.train()
    order = list(range(len(rows_tr)))
    random.Random(args.seed * 100 + ep).shuffle(order)
    k, losses = 0, []
    steps = len(order) // args.accum
    for st in range(steps):
        opt.zero_grad(set_to_none=True)
        ls = 0.0
        for _ in range(args.accum):
            s = rows_tr[order[k]]; k += 1
            loss = one_step(s) / args.accum
            loss.backward()
            ls += float(loss.detach())
        gn = torch.nn.utils.clip_grad_norm_(
            [p_ for p_ in model.parameters() if p_.requires_grad], 1.0)
        opt.step()
        losses.append(ls)
        if not torch.isfinite(gn):
            rec['nan_step'] = {'epoch': ep, 'step': st}
        if (st + 1) % 20 == 0:
            el = time.time() - te
            print(f'ep{ep} step {st+1}/{steps} loss {ls:.4f} gn {float(gn):.2f} '
                  f'{el/(st+1):.2f}s/step', flush=True)
    ap_dir = Path(args.adapter_dir) / f'ep{ep}'
    model.save_pretrained(str(ap_dir))
    ev, picks = eval_dev(f'ep{ep}')
    ev.update({'epoch': ep, 'adapter': str(ap_dir),
               'loss_mean': round(float(np.mean(losses)), 4),
               'loss_last10': round(float(np.mean(losses[-10:])), 4),
               'train_min': round((time.time() - te) / 60, 1)})
    rec['epochs_log'].append(ev)
    rec['status'] = 'RUNNING'
    Path(args.out).write_text(json.dumps(rec, indent=1) + '\n')
    print(f"EPOCH {ep} DONE dev_macro_iou {ev['dev_macro_iou']} "
          f"loss {ev['loss_mean']} {ev['train_min']}min", flush=True)

last = rec['epochs_log'][-1]
rec['perm_control'] = perm_control(picks)
rec['status'] = 'DONE'
Path(args.out).write_text(json.dumps(rec, indent=1) + '\n')
print('WROTE', args.out, 'perm_control', rec['perm_control'], flush=True)
