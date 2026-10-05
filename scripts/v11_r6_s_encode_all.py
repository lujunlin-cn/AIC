"""R6 S-REREAD step 5: full crop+full-image encoding for the frozen pools.

For every frozen row (vid, frame, ratio):
  * B3 score over the 129 candidates -> top-3 (train rows: B3 cache feat;
    dev rows: grid from live_feats_dev80)
  * shortlist = top3 U g33 (<=36); decode the 640px frame ONCE
  * encode every shortlist crop + the full image with the deployment
    Siglip2 contract (valid-token mean, [768] fp32)
Shard rows across 6 cards.  Output per shard:
  r6_s_encode_shard<k>.pt  {crop_emb: {key: [768]}, full_emb: {key: [768]},
                            b3_scores: {rowkey: [129]}, u: {rowkey: [129]},
                            shortlist: {rowkey: [...]},
                            crop_boxes: {rowkey: [[x1,y1,x2,y2] x S]} (640px)}
Run: ASCEND_RT_VISIBLE_DEVICES=$((k+2)) ... --shard k --nshards 6
"""
import argparse, csv, json, os, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '16')
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
from pathlib import Path
import sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--freeze', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/r6_s_pool_freeze.json'))
ap.add_argument('--t5-train-index', type=Path,
                default=Path('/data/aic/experiments/T5_CROPHEAD_V3/live_train_index.jsonl'))
ap.add_argument('--samples-dir', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V8/samples'))
ap.add_argument('--dev80-grid', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/live_feats_dev80'))
ap.add_argument('--release', default='/data/aic/external_datasets/_releases/spatial_crop_v1')
ap.add_argument('--head', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V8/arms/B3_s0/head_s.pt'))
ap.add_argument('--model-dir', default='/data/aic/pretrained/lfm2_5_vl_450m')
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshards', type=int, default=6)
ap.add_argument('--out', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/r6_s_encode_shard.pt'))
args = ap.parse_args()

import torch
import torch_npu                                            # noqa: F401
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False
import av                                                   # noqa: E402
from PIL import Image
sys.path.insert(0, '/root/AIC')
sys.path.insert(0, '/root/AIC/ext_data')
from v11_npu_siglip2_patch import patch_siglip2_npu         # noqa: E402
from aic.max_window_path import geometry                    # noqa: E402
from aicext.release import Release                          # noqa: E402
# Head/RATIOS/GRID/win_boxes are copied verbatim below (NOT imported from
# v11_r6_s_shortlist_oracle - that module parses args at import time and
# would swallow --shard/--nshards).

RATIOS = {'1-3': [1, 3], '3-1': [3, 1], '9-16': [9, 16], '16-9': [16, 9]}
GRID = list(range(0, 129, 4))                          # g33 (amended shortlist)


class Head(torch.nn.Module):
    """Copied verbatim from v8_s_train_multidata.py."""

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


def win_boxes(W, H, ratio, nc):
    w, h, axis = geometry(W, H, ratio)
    span = (W - w) if axis == 0 else ((H - h) if axis == 1 else 0.0)
    offs = np.linspace(0, span, nc) if span > 0 else np.zeros(1)
    boxes = []
    for o in offs:
        if axis == 0:
            boxes.append([o, 0, o + w, h])
        elif axis == 1:
            boxes.append([0, o, w, o + h])
        else:
            boxes.append([0, 0, W, H])
    return np.array(boxes, np.float32), axis
from transformers import AutoProcessor, Lfm2VlForConditionalGeneration

# ---- frozen rows (this shard) ----
rows = list(csv.DictReader(args.freeze.with_name(args.freeze.stem + '_rows.csv').open()))
mine = [r for i, r in enumerate(rows) if i % args.nshards == args.shard]
print(f'shard {args.shard}/{args.nshards}: {len(mine)} rows', flush=True)

# ---- B3 head + inputs ----
ck = torch.load(args.head, map_location='cpu', weights_only=False)
b3 = Head(2305, 129, ch=ck.get('ch', 128)).float()
b3.load_state_dict(ck['state_dict']); b3.eval()
torch.set_grad_enabled(False)


def assemble_feat(grid, W, H, ratio_key, nc=129):
    fh, fw, D = grid.shape
    flat = grid.astype(np.float32).reshape(-1, D)
    boxes, axis = win_boxes(W, H, RATIOS[ratio_key], nc)
    px_per = np.array([W / fw, H / fh])
    gy, gx = np.mgrid[0:fh, 0:fw]
    m = np.zeros((len(boxes), fh * fw), np.float32)
    for j, (x1, y1, x2, y2) in enumerate(boxes):
        cx1, cx2 = int(np.floor(x1 / px_per[0])), int(np.ceil(x2 / px_per[0]))
        cy1, cy2 = int(np.floor(y1 / px_per[1])), int(np.ceil(y2 / px_per[1]))
        cx2, cy2 = max(cx2, cx1 + 1), max(cy2, cy1 + 1)
        m[j] = ((gx >= cx1) & (gx < cx2) & (gy >= cy1) & (gy < cy2)).reshape(-1).astype(np.float32)
    ms = m.sum(1, keepdims=True)
    win = (m @ flat) / ms
    out = ((1 - m) @ flat) / np.maximum((1 - m).sum(1, keepdims=True), 1)
    span = (W - boxes[0][2]) if axis == 0 else ((H - boxes[0][3]) if axis == 1 else 0.0)
    offs = np.linspace(0, span, nc) if span > 0 else np.zeros(1)
    pos = (offs / span if span > 0 else offs).astype(np.float32)
    feat = np.concatenate([win, out, win - out, pos[:, None]], 1)
    return feat.astype(np.float16), boxes


# ---- vision encoder ----
proc = AutoProcessor.from_pretrained(args.model_dir, min_tiles=1, max_tiles=1,
                                     max_image_tokens=256)
model = Lfm2VlForConditionalGeneration.from_pretrained(
    args.model_dir, dtype=torch.float16, low_cpu_mem_usage=True,
    attn_implementation='eager').eval().to('npu')
assert patch_siglip2_npu(model)


def embed(im):
    msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': im},
                                         {'type': 'text', 'text': 'describe'}]}]
    x = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                 return_dict=True, return_tensors='pt')
    x = {k: v.to('npu') for k, v in x.items() if isinstance(v, torch.Tensor)}
    with torch.no_grad():
        out = model.model.vision_tower(pixel_values=x['pixel_values'],
                                       spatial_shapes=x['spatial_shapes'],
                                       pixel_attention_mask=x['pixel_attention_mask'],
                                       return_dict=True)
    valid = int(x['pixel_attention_mask'][0].sum())
    fh, fw = [int(v) for v in x['spatial_shapes'][0]]
    return out.last_hidden_state[0, :valid].float().reshape(fh, fw, -1).mean((0, 1)).cpu()


# ---- media + W/H ----
rel = Release('spatial_crop_v1', path=Path(args.release))
t5meta = {}
for l in args.t5_train_index.read_text().splitlines():
    r = json.loads(l)
    t5meta[r['video_id']] = (r['video_path'], float(r['width']), float(r['height']))
tr_cache = np.load(args.samples_dir / 'live_train.npz', allow_pickle=True)
tr_vid = tr_cache['vid'].astype(str)
per_vid_rows = {}
for i, v in enumerate(tr_vid.tolist()):
    per_vid_rows.setdefault(v, []).append(i)

out = {'crop_emb': {}, 'full_emb': {}, 'b3_scores': {}, 'u': {},
       'shortlist': {}, 'crop_boxes': {}, 'shard': args.shard}
out_path = args.out.with_name(f'{args.out.stem}{args.shard}.pt')
if out_path.exists():                      # checkpoint resume
    try:
        prev = torch.load(out_path, map_location='cpu', weights_only=False)
        for k in ('crop_emb', 'full_emb', 'b3_scores', 'u', 'shortlist',
                  'crop_boxes'):
            out[k].update(prev.get(k, {}))
        print(f'shard {args.shard}: resumed {len(out["b3_scores"])} finished rows',
              flush=True)
    except Exception as e:                 # truncated file from a crash
        print(f'shard {args.shard}: checkpoint unreadable ({e}), starting fresh',
              flush=True)
t0 = time.time()
done = len(out['b3_scores'])
from collections import defaultdict
groups = defaultdict(list)
finished = set(out['b3_scores'])
for r in mine:
    key = f"{r['vid']}|{r['frame']}|{r['ratio']}"
    if key not in finished:
        groups[r['vid']].append(r)
print(f'shard {args.shard}: {len(mine)} rows total, {len(groups)} vids to do',
      flush=True)
print(f'shard {args.shard}: {len(groups)} vids', flush=True)
for vid, rs in groups.items():
    mp, W, H = t5meta[vid]
    s = 640.0 / max(W, H)
    want = {int(r['frame']) for r in rs}
    # ONE sequential decode per video, shared by all its rows
    imgs = {}
    with av.open(mp) as cont:
        stream = cont.streams.video[0]
        stream.thread_type = 'AUTO'                  # multithreaded decode
        fi = 0
        for frame in cont.decode(stream):
            if fi in want:
                im = frame.to_image().convert('RGB')
                imgs[fi] = im.resize((max(1, round(im.width * s)),
                                      max(1, round(im.height * s))),
                                     Image.BILINEAR)
            fi += 1
            if fi > max(want):
                break
    missing = want - set(imgs)
    assert not missing, f'{vid}: counters not reached {sorted(missing)}'
    for r in rs:
        fr, ratio = int(r['frame']), r['ratio']
        if r['pool'] == 'train240':
            cand_rows = [j for j in per_vid_rows[vid] if int(tr_cache['frame'][j]) == fr
                         and str(tr_cache['ratio'][j]) == ratio]
            assert cand_rows, f'{vid}/{fr}/{ratio} missing from cache'
            feat = tr_cache['feat'][cand_rows[0]]
            u129 = tr_cache['u'][cand_rows[0]].astype(np.float32)
        else:
            gp = args.dev80_grid / vid / f'{fr}.npz'
            grid = np.load(gp)['grid']
            feat, _ = assemble_feat(grid, W, H, ratio)
            u129 = np.array(json.loads(r['u']), np.float32)
        sc = b3(torch.from_numpy(feat[None]).float())[0].numpy()
        top3 = np.argsort(-sc)[:3].tolist()
        sl = sorted(set(top3) | set(GRID))
        got = imgs[fr]
        iw, ih = got.size
        boxes, _ = win_boxes(W, H, RATIOS[ratio], 129)
        rowkey = f'{vid}|{fr}|{ratio}'
        out['b3_scores'][rowkey] = sc.astype(np.float16)
        out['u'][rowkey] = u129
        out['shortlist'][rowkey] = sl
        out['crop_boxes'][rowkey] = [[
            max(0, int(round(boxes[c][0] * s))), max(0, int(round(boxes[c][1] * s))),
            min(iw, max(3, int(round(boxes[c][2] * s)))),
            min(ih, max(3, int(round(boxes[c][3] * s))))] for c in sl]
        with torch.no_grad():
            for bi, c in enumerate(sl):
                out['crop_emb'][f'{rowkey}|{c}'] = embed(got.crop(tuple(out['crop_boxes'][rowkey][bi])))
            out['full_emb'][rowkey] = embed(got)
        done += 1
        if done % 10 == 0:
            torch.save(out, out_path)      # incremental checkpoint
        if done % 20 == 0:
            el = time.time() - t0
            print(f'shard {args.shard}: {done}/{len(mine)} rows, {el:.0f}s, '
                  f'{done / el * 3600:.0f} rows/h', flush=True)
torch.save(out, out_path)
print(f'SHARD {args.shard} DONE: {done} rows, {len(out["crop_emb"])} crops, '
      f'{time.time() - t0:.0f}s', flush=True)
