"""R8 VLM_SFT stability probe (preregistered: 50 stable steps before any claim).

Question: can a LoRA r=8 (q/v, LANGUAGE LAYERS ONLY, vision tower frozen)
Qwen2.5-VL-7B train stably on the candidate-ID readout task, and what does
one epoch cost?  This probe measures stability and throughput only - no
quality claim, no gate.

Architecture decisions forced by Probe 0:
  - multi-image concatenated batches have quadratic attention cost on this
    stack (batch 8 did not finish in ~25 min while batch 1 takes 0.58 s), so
    training uses ONE image per forward + gradient accumulation 8
    (effective batch 8, per preregistration design).

Data: 64 rows sampled from the frozen train240 table (public LIVE release,
single-annotator raw_single_box labels).  Prompt shows the frame and the
36-shortlist candidate boxes; the target is the argmax-utility candidate
ID inside the shortlist.  Frames are extracted once to disk.

Incremental writes: stability_probe.json is rewritten every 10 steps.
"""
import argparse, csv, glob, json, os, random, time
from pathlib import Path
import numpy as np
import torch

for _v in ('OMP_NUM_THREADS',):
    os.environ.setdefault(_v, '8')
p = argparse.ArgumentParser()
p.add_argument('--model-path', default='/data/aic/pretrained/Qwen2.5-VL-7B-Instruct')
p.add_argument('--freeze-csv', default='/data/aic/experiments_910a/LFM_V11/r6_s_pool_freeze_rows.csv')
p.add_argument('--shard-glob', default='/data/aic/experiments_910a/LFM_V11/r6_s_encode_shard*.pt')
p.add_argument('--t5-index', default='/data/aic/experiments/T5_CROPHEAD_V3/live_train_index.jsonl')
p.add_argument('--frames-dir', default='/data/aic/experiments_910a/LFM_V11/r8_sft_frames')
p.add_argument('--out', default='/data/aic/experiments_910a/LFM_V11/r8_npu/vlm_sft_stability.json')
p.add_argument('--steps', type=int, default=50)
p.add_argument('--accum', type=int, default=8)
p.add_argument('--lr', type=float, default=5e-6)
p.add_argument('--n-train', type=int, default=64)
p.add_argument('--lora-r', type=int, default=8)
args = p.parse_args()

os.environ.setdefault('ASCEND_RT_VISIBLE_DEVICES', '2')
import torch_npu  # noqa: E402,F401
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False
assert torch.npu.device_count() > 0, 'no NPU visible - refusing CPU fallback'
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from PIL import Image  # noqa: E402
import cv2  # noqa: E402

sys_path = None
sys_eps = None

# ---------- data: frames + shortlist boxes + target id ----------
FD = Path(args.frames_dir)
FD.mkdir(parents=True, exist_ok=True)
t5 = {}
for l in open(args.t5_index):
    r = json.loads(l)
    t5[r['video_id']] = r
u_map, sl_map, box_map = {}, {}, {}
import glob as _g
for sp in sorted(_g.glob(args.shard_glob)):
    d = torch.load(sp, map_location='cpu', weights_only=False)
    u_map.update(d['u']); sl_map.update(d['shortlist']); box_map.update(d['crop_boxes'])

rows = [r for r in csv.DictReader(open(args.freeze_csv)) if r['pool'] == 'train240']
random.Random(20261051).shuffle(rows)
rows = rows[:args.n_train]
samples = []
for r in rows:
    key = f"{r['vid']}|{r['frame']}|{r['ratio']}"
    if key not in u_map:
        continue
    fp = FD / f"{r['vid']}_f{r['frame']}.jpg"
    if not fp.exists():
        t5r = t5[r['vid']]
        cap = cv2.VideoCapture(t5r['video_path'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(r['frame']))
        ok, img = cap.read()
        cap.release()
        if not ok:
            continue
        cv2.imwrite(str(fp), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    sl = [int(c) for c in sl_map[key]]
    boxes = box_map[key]   # 36 boxes, already ordered per the shortlist
    if len(boxes) != len(sl):
        print(f'SKIP {key}: {len(boxes)} boxes vs {len(sl)} shortlist', flush=True)
        continue
    u = np.asarray(u_map[key], np.float32)
    tgt = int(np.argmax(u[sl]))   # position within the shortlist
    samples.append({'img': str(fp), 'boxes': [boxes[j] for j in range(len(sl))],
                    'target': tgt})
print(f'samples: {len(samples)}', flush=True)


def build_prompt(boxes):
    parts = [f'{i}:({b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f})'
             for i, b in enumerate(boxes)]
    return ('This frame lists candidate crop windows: ' + ', '.join(parts) +
            '. Which window best frames the subject? Reply with one number.')


# ---------- model: LoRA on language q/v only ----------
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
model.train()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

# trainable-parameter sanity: LoRA params must live in language_model only
train_names = [n for n, p_ in model.named_parameters() if p_.requires_grad]
n_vis = sum(1 for n in train_names if 'visual' in n)
print(f'trainable tensors: {len(train_names)}, inside visual: {n_vis}', flush=True)
assert n_vis == 0, 'LoRA leaked into the vision tower - abort'


def one_step(sample):
    msgs = [{'role': 'user', 'content': [
        {'type': 'image', 'image': sample['img']},
        {'type': 'text', 'text': build_prompt(sample['boxes'])}]}]
    x = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                 return_dict=True, return_tensors='pt')
    x = {k: v.to('npu') if isinstance(v, torch.Tensor) else v for k, v in x.items()}
    ans = proc.tokenizer(str(sample['target']), return_tensors='pt').input_ids.to('npu')
    input_ids = torch.cat([x['input_ids'], ans], 1)
    labels = torch.cat([torch.full_like(x['input_ids'], -100), ans], 1)
    x['input_ids'] = input_ids
    if 'attention_mask' in x:
        x['attention_mask'] = torch.ones_like(input_ids)
    # processor-derived length-coupled fields must not outlive the concat:
    # get_rope_index indexes mm_token_type_ids with attention_mask and
    # shape-mismatches (token_type_ids is the legacy alias)
    for stale in ('token_type_ids', 'mm_token_type_ids'):
        x.pop(stale, None)
    x['labels'] = labels
    out = model(**x)
    return out.loss


rec = {'protocol': 'R8 VLM_SFT stability probe; LoRA r=8 language q/v; vision frozen; '
                   'single-image forward + grad accum 8; no quality claim',
       'model': 'Qwen2.5-VL-7B', 'n_train': len(samples), 'steps': args.steps,
       'accum': args.accum, 'lr': args.lr, 'history': [], 'status': 'RUNNING'}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(rec, indent=1) + '\n')

t0 = time.time()
loss_acc, nan_seen = 0.0, False
order = list(range(len(samples))) * 3
k = 0
for step in range(1, args.steps + 1):
    loss_s = 0.0
    opt.zero_grad(set_to_none=True)
    for _ in range(args.accum):
        s = samples[order[k % len(order)]]
        k += 1
        loss = one_step(s) / args.accum
        loss.backward()
        loss_s += float(loss.detach())
    gn = torch.nn.utils.clip_grad_norm_(
        [p_ for p_ in model.parameters() if p_.requires_grad], 1.0)
    opt.step()
    loss_acc += loss_s
    if not torch.isfinite(gn):
        nan_seen = True
    if step % 10 == 0 or step == 1:
        el = time.time() - t0
        rec['history'].append({'step': step, 'loss': round(loss_s, 4),
                               'grad_norm': round(float(gn), 3),
                               's_per_step': round(el / step, 2),
                               'nan': bool(nan_seen)})
        rec['status'] = 'RUNNING'
        Path(args.out).write_text(json.dumps(rec, indent=1) + '\n')
        print(f'step {step} loss {loss_s:.4f} gn {float(gn):.2f} '
              f'{el/step:.2f}s/step', flush=True)

el = time.time() - t0
n_vids = 240
steps_per_epoch = n_vids * 8 // args.accum
rec['status'] = 'DONE' if not nan_seen else 'UNSTABLE_NAN'
rec['summary'] = {
    's_per_step': round(el / args.steps, 2),
    'projected_epoch_hours_1card': round(steps_per_epoch * el / args.steps / 3600, 2),
    'loss_first_vs_last': [rec['history'][0]['loss'], rec['history'][-1]['loss']],
    'grad_norm_max': max(h['grad_norm'] for h in rec['history']),
    'finite_grads': not nan_seen}
Path(args.out).write_text(json.dumps(rec, indent=1) + '\n')
print('SUMMARY', json.dumps(rec['summary']), flush=True)
