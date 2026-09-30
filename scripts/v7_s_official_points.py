"""V7 S-line official inference: candidate-utility head -> per-keyframe crop point.

Same head and candidate construction as training (129 equal legal windows along
the free axis, centre included); single-tile NaFlex vision-tower features are
re-extracted here from the official keyframe PNGs (long side 640) and the
argmax window centre is emitted as a normalised point, matching the qwen points
schema consumed unchanged by scripts/max_window_release.py (policy QWEN_POINT).

Head class is copied from v7_s_train_head.py (keep in sync).  No teacher output,
no B0 cache, no official-test GT enters this path.
"""
import argparse, json, math, time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='/data/aic/pretrained/lfm2_5_vl_450m')
ap.add_argument('--head', type=Path, required=True)
ap.add_argument('--index', type=Path, default=Path('/data/aic/official_test_20260926/intake/index.enriched.jsonl'))
ap.add_argument('--keyframe-src', type=Path, default=Path('/data/aic/experiments/QWEN_SUBJECT_POINT_OFFICIAL_V1'))
ap.add_argument('--cards', default='2')
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshards', type=int, default=1)
ap.add_argument('--output-dir', type=Path, required=True)
args = ap.parse_args()

import os  # noqa: E402
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = args.cards
import torch  # noqa: E402
import torch_npu  # noqa: E402,F401
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False
from PIL import Image  # noqa: E402
from transformers import AutoProcessor, Lfm2VlForConditionalGeneration  # noqa: E402

import sys  # noqa: E402
sys.path.insert(0, '/root/AIC')
from aic.max_window_path import geometry  # noqa: E402

NC = 129


class Head(torch.nn.Module):
    """MLP + 2 self-attention layers, eager attention (SDPA backend unsupported on ascend910)."""

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

    def forward(self, x):  # x (B,NC,d)
        h = self.proj(x)
        for i in range(2):
            h = self._ff(self._attn(h, i), i)
        return self.out(h).squeeze(-1)  # (B,NC)


proc = AutoProcessor.from_pretrained(args.model, min_tiles=1, max_tiles=1, max_image_tokens=256)
model = Lfm2VlForConditionalGeneration.from_pretrained(
    args.model, dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation='eager').eval().to('npu')
ck = torch.load(args.head, map_location='npu', weights_only=True)
D = int(ck['config']['d'])
assert int(ck['config']['nc']) == NC
head = Head(D, NC).to('npu').float()
head.load_state_dict(ck['state_dict'])
head.eval()

recs = [json.loads(l) for l in args.index.read_text().splitlines() if l.strip()][args.shard::args.nshards]
pdir = args.output_dir / 'points'
pdir.mkdir(parents=True, exist_ok=True)
n_f = 0
t0 = time.time()
for r in recs:
    vid = r['video_id']
    if (pdir / f'{vid}.json').exists():
        continue
    src = json.loads((args.keyframe_src / 'points' / f'{vid}.json').read_text())
    kfs = src['keyframes']
    pts, st = [], []
    t_vid = time.time()
    for kf in kfs:
        img = Image.open(args.keyframe_src / 'keyframes' / vid / f'{kf}.png').convert('RGB')
        W, H = img.size
        msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': img},
                                             {'type': 'text', 'text': 'describe'}]}]
        x = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                     return_dict=True, return_tensors='pt')
        x = {k: v.to('npu') for k, v in x.items() if isinstance(v, torch.Tensor)}
        if 'pixel_values' in x:
            x['pixel_values'] = x['pixel_values'].to(torch.float16)
        with torch.no_grad():
            out = model.model.vision_tower(pixel_values=x['pixel_values'], spatial_shapes=x['spatial_shapes'],
                                           pixel_attention_mask=x['pixel_attention_mask'], return_dict=True)
            valid = int(x['pixel_attention_mask'][0].sum())
            fh, fw = [int(v) for v in x['spatial_shapes'][0]]
            grid = out.last_hidden_state[0, :valid].reshape(fh, fw, -1)[:, : (fw // 2) * 2, :]
            grid = grid[: (fh // 2) * 2].float()
        flat = grid.reshape(-1, grid.shape[-1])
        rw, rh = r['targetRatioWH']
        w, h, axis = geometry(float(W), float(H), [rw, rh])
        span = (W - w) if axis == 0 else ((H - h) if axis == 1 else 0.0)
        offs = np.linspace(0, span, NC) if span > 0 else np.zeros(1)
        n_tok = fh * fw
        m = np.zeros((len(offs), n_tok), dtype=np.float32)
        px_per = np.array([W / fw, H / fh])
        gy, gx = np.mgrid[0:fh, 0:fw]
        for j, o in enumerate(offs):
            if axis == 0:
                x1, y1, x2, y2 = o, 0, o + w, h
            elif axis == 1:
                x1, y1, x2, y2 = 0, o, w, o + h
            else:
                x1, y1, x2, y2 = 0, 0, W, H
            cx1, cx2 = int(math.floor(x1 / px_per[0])), max(int(math.ceil(x2 / px_per[0])), int(math.floor(x1 / px_per[0])) + 1)
            cy1, cy2 = int(math.floor(y1 / px_per[1])), max(int(math.ceil(y2 / px_per[1])), int(math.floor(y1 / px_per[1])) + 1)
            m[j] = ((gx >= cx1) & (gx < cx2) & (gy >= cy1) & (gy < cy2)).reshape(-1).astype(np.float32)
        win = torch.from_numpy(m @ flat.cpu().numpy() / np.maximum(m.sum(1, keepdims=True), 1)).to('npu')
        outm = torch.from_numpy((1 - m) @ flat.cpu().numpy() / np.maximum((1 - m).sum(1, keepdims=True), 1)).to('npu')
        pos = torch.from_numpy((offs / span if span > 0 else offs)[:, None].astype(np.float32)).to('npu')
        feat = torch.cat([win, outm, win - outm, pos], 1).unsqueeze(0)
        with torch.no_grad():
            u = head(feat)[0]
        j = int(u.argmax())
        if axis == 0:
            cx, cy = (offs[j] + w / 2) / W, 0.5
        elif axis == 1:
            cx, cy = 0.5, (offs[j] + h / 2) / H
        else:
            cx, cy = 0.5, 0.5
        pts.append([round(float(cx), 6), round(float(cy), 6), False])
        st.append('ok_head')
        n_f += 1
    rec = {'video_id': vid, 'W': src['W'], 'H': src['H'], 'fps': src['fps'], 'step': src['step'],
           'keyframes': kfs, 'model': f'LFM2.5-VL-450M@fc6221ca597f3315e4f82fc2df606783267b34ba+v7_s_head(seed{ck["config"]["seed"]})',
           'prompt': 'NONE_HEAD', 'prompt_text': 'candidate-utility head over LFM vision tokens (no text prompt at inference)',
           'ratios': {'t': {'ratio': r['targetRatioWH'], 'points': pts, 'status': st, 'raw': [], 'boxes': []}},
           'seconds': round(time.time() - t_vid, 2)}
    (pdir / f'{vid}.json').write_text(json.dumps(rec) + '\n')
    print(f'{vid} kf={len(kfs)}', flush=True)
summ = {'shard': args.shard, 'frames': n_f, 'wall_s': round(time.time() - t0, 1),
        's_per_frame': round((time.time() - t0) / max(n_f, 1), 3), 't_done': time.strftime('%m-%d %H:%M:%S')}
(args.output_dir / f'summary_s{args.shard}.json').write_text(json.dumps(summ, indent=1) + '\n')
print('SUMMARY', json.dumps(summ), flush=True)
