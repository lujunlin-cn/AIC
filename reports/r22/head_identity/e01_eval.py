"""E0.1: VPB head 在 uniform pool 上的 holdout spearman (v3, 修复 load)。"""
import json, numpy as np, torch, torch.nn as nn, sys, os

UNIFORM_POOL = "/root/autodl-tmp/vpb/pool_uniform.jsonl"
VPB_HEAD_PT = "/root/autodl-tmp/vpb/vp_score_head.pt"
FRAMES_NPY = "/root/autodl-tmp/vpb/frames_288.npy"
VP_MODEL_NPZ = "/root/autodl-tmp/vpb/videoprism_base_f16r288.fp16.pt"

sys.path.insert(0, "/root/autodl-tmp/vpb")
from videoprism_torch import build_base

def rankdata(a):
    a = np.asarray(a, dtype=np.float64); order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64); sa = a[order]; i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]: j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0; i = j + 1
    return ranks

def spearman_rows(pred, target):
    out = []
    for p, t in zip(pred, target):
        if np.std(p) < 1e-6 or np.std(t) < 1e-6: continue
        rp, rt = rankdata(p), rankdata(t)
        out.append(np.corrcoef(rp, rt)[0, 1])
    return np.array(out)

class ScoreHead(nn.Module):
    def __init__(self, d=768):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 512), nn.ReLU(), nn.Linear(512, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

print("Loading uniform pool...")
rows = [json.loads(l) for l in open(UNIFORM_POOL)]
print(f"Total segments: {len(rows)}")
all_zero = sum(1 for r in rows if all(x == 0 for x in r["gif_hard"]))
print(f"All-zero: {all_zero} ({all_zero/len(rows)*100:.1f}%)")

N = len(rows); np.random.seed(13); hold_idx = np.random.permutation(N)[:125]
gif_hard = np.array([r["gif_hard"] for r in rows], dtype=np.float32); gif_hold = gif_hard[hold_idx]
print(f"Holdout: {len(hold_idx)}, non-zero: {sum(1 for i in hold_idx if not all(gif_hard[i] == 0))}")

print("Loading VideoPrism-B + head...")
vp_model = build_base(ckpt_path=VP_MODEL_NPZ).cuda().eval().half()
head = ScoreHead(768).cuda()
ckpt = torch.load(VPB_HEAD_PT, map_location="cuda", weights_only=False)
head.load_state_dict(ckpt["state"])
head.half().eval()

print("Loading frames...")
frames = np.load(FRAMES_NPY, mmap_mode='r'); print(f"Shape: {frames.shape}")

print("Scoring holdout...")
holdout_preds = []
with torch.no_grad():
    for k, seg_idx in enumerate(hold_idx):
        seg_frames = frames[seg_idx]  # (12, 288, 288, 3) uint8
        inp = torch.from_numpy(seg_frames).cuda().half() / 255.0
        inp = inp.unsqueeze(0)  # (1, 12, 288, 288, 3)
        feat = vp_model(inp).mean(2)[0]  # (12, 768)
        scores = head(feat).cpu().numpy()
        holdout_preds.append(scores)
        if (k+1) % 25 == 0: print(f"  {k+1}/125", flush=True)

holdout_preds = np.array(holdout_preds)
sp = spearman_rows(holdout_preds, gif_hold)
print(f"\n=== E0.1 Result ===")
print(f"Uniform pool spearman: {sp.mean():.4f} ± {sp.std():.4f} (n={len(sp)})")
print(f"Centered pool ref: 0.5696")
print("\nInterpretation:")
if sp.mean() < 0.3: print("  ✗ POSITION PRIOR learned → supervision problem")
elif sp.mean() > 0.5: print("  ✓ TRUE CONTENT learned → continue path")
else: print("  ? Inconclusive")
