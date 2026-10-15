"""E0.1: 评估 VPB head 在 uniform pool 上的 holdout spearman。
对比 centered pool (0.570) vs uniform pool，判断 VPB 是在学位置先验还是真实内容。
"""
import json
import numpy as np
import torch
import torch.nn as nn
import sys

UNIFORM_POOL = "/root/autodl-tmp/vpb/pool_uniform.jsonl"
VPB_HEAD_PT = "/root/autodl-tmp/vpb/vp_score_head.pt"
FRAMES_NPY = "/root/autodl-tmp/vpb/frames_288.npy"
VP_MODEL = "/root/autodl-tmp/vpb/videoprism_base_f16r288.fp16.pt"

def rankdata(a):
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks

def spearman_rows(pred, target):
    out = []
    for p, t in zip(pred, target):
        if np.std(p) < 1e-6 or np.std(t) < 1e-6:
            continue
        rp, rt = rankdata(p), rankdata(t)
        out.append(np.corrcoef(rp, rt)[0, 1])
    return np.array(out)

class ScoreHead(nn.Module):
    def __init__(self, d=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 512), nn.ReLU(), nn.Linear(512, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

print("Loading uniform pool...")
rows = [json.loads(l) for l in open(UNIFORM_POOL)]
print(f"Total segments: {len(rows)}")

# 检查 git_hard 分布
all_zero = sum(1 for r in rows if all(x == 0 for x in r["gif_hard"]))
print(f"All-zero segments: {all_zero} ({all_zero/len(rows)*100:.1f}%)")

# Holdout split (same seed as training)
N = len(rows)
np.random.seed(13)
hold_idx = np.random.permutation(N)[:125]
train_mask = np.ones(N, bool)
train_mask[hold_idx] = False

gif_hard = np.array([r["gif_hard"] for r in rows], dtype=np.float32)
gif_hold = gif_hard[hold_idx]

print(f"Holdout: {len(hold_idx)} segments")
non_zero_hold = sum(1 for i in hold_idx if not all(gif_hard[i] == 0))
print(f"Holdout non-zero: {non_zero_hold}/{len(hold_idx)}")

# 加载 VideoPrism-B 模型和 head
print("Loading VideoPrism-B model...")
sys.path.insert(0, "/root/autodl-tmp/vpb")
from videoprism_torch import build_videoprism_b
vp_model = build_videoprism_b(VP_MODEL, device="cuda", dtype=torch.float16)
vp_model.eval()

print("Loading VPB score head...")
head = ScoreHead(768).cuda()
head.load_state_dict(torch.load(VPB_HEAD_PT, map_location="cuda"))
head.eval()

print("Loading frame cache...")
frames = np.load(FRAMES_NPY, mmap_mode='r')  # shape: (N_seg, 12, 288, 288, 3)
print(f"Frame cache shape: {frames.shape}")

# 提取 holdout 特征并打分
print("Extracting features and scoring holdout...")
holdout_preds = []
with torch.no_grad():
    for seg_idx in hold_idx:
        # 读取 12 帧 (288, 288, 3) uint8
        seg_frames = frames[seg_idx]  # (12, 288, 288, 3)
        # 转 torch, CHW, normalize [-1, 1], fp16
        inp = torch.from_numpy(seg_frames).permute(0, 3, 1, 2).cuda().half() / 127.5 - 1.0  # (12, 3, 288, 288)
        # VideoPrism-B forward
        feat_t = vp_model(inp.unsqueeze(0))["temporal"][0]  # (12, 768)
        # Head forward
        scores = head(feat_t).cpu().numpy()  # (12,)
        holdout_preds.append(scores)

holdout_preds = np.array(holdout_preds)  # (125, 12)

# 计算 spearman
sp = spearman_rows(holdout_preds, gif_hold)
print(f"\n=== E0.1 Result ===")
print(f"Holdout spearman (uniform pool): {sp.mean():.4f} ± {sp.std():.4f}")
print(f"N valid pairs: {len(sp)}")
print(f"Reference (centered pool, from HEAD_REPORT.json): 0.5696")
print(f"\nInterpretation:")
if sp.mean() < 0.3:
    print("  ✗ VPB head is learning POSITION PRIOR, not content.")
    print("  → Supervision signal problem confirmed.")
elif sp.mean() > 0.5:
    print("  ✓ VPB head learned TRUE CONTENT signal.")
    print("  → Continue current path.")
else:
    print("  ? Inconclusive (0.3 < spearman < 0.5).")
