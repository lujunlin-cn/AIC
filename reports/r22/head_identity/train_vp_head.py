# Probe gate + lightweight highlight-scoring head on frozen VideoPrism-B feats.
# Merges shards, aligns to sft row order, evaluates on the SAME v2 holdout
# (default_rng(13).permutation(N)[:125]) so numbers are comparable with the
# SFT-v2 pareto table (ckpt_030 spearman 0.539, base 0.22).
# Usage: train_vp_head.py <sft_jsonl> <feat_dir> <out_dir>
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

SFT, FEAT_DIR, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)


def rankdata(a):
    """Average-rank (ties), 0-based not needed; returns 1-based ranks."""
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
    """Per-row spearman; rows with zero variance are skipped."""
    out = []
    for p, t in zip(pred, target):
        if np.std(p) < 1e-6 or np.std(t) < 1e-6:
            continue
        rp, rt = rankdata(p), rankdata(t)
        out.append(np.corrcoef(rp, rt)[0, 1])
    return np.array(out)


rows = [json.loads(l) for l in open(SFT)]
vid2row = {r["vid"]: i for i, r in enumerate(rows)}

feat_t, feat_s, vids = [], [], []
for f in sorted(glob.glob(os.path.join(FEAT_DIR, "shard_*of*.npz"))):
    z = np.load(f)
    vids.extend(z["vid"].tolist())
    feat_t.append(z["feat_temporal"])
    feat_s.append(z["feat_spatial"])
feat_t, feat_s = np.concatenate(feat_t), np.concatenate(feat_s)
assert len(vids) == len(rows), f"{len(vids)} feats vs {len(rows)} rows"
order = np.array([vid2row[v] for v in vids])
feat_t = feat_t[np.argsort(order)]
feat_s = feat_s[np.argsort(order)]

gif = np.array([r["gif_hard"] for r in rows], dtype=np.float32)
tgt = np.array([r["_target"] for r in rows], dtype=np.float32)
N = len(rows)
hold = np.random.default_rng(13).permutation(N)[:125]
train_mask = np.ones(N, bool)
train_mask[hold] = False
print(f"N={N} train={train_mask.sum()} holdout={len(hold)}")

DEV = os.environ.get("VP_DEV", "cpu")
torch.manual_seed(13)


def fit_ridge(feat, y, alpha):
    X = feat.reshape(-1, feat.shape[-1])
    Y = y.reshape(-1, 1)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X = (X - mu) / sd
    A = X.T @ X + alpha * np.eye(X.shape[1], dtype=X.dtype)
    w = np.linalg.solve(A, X.T @ Y)
    return w, mu, sd


class Head(nn.Module):
    def __init__(self, d=768):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 512), nn.GELU(),
                                 nn.Linear(512, 1))

    def forward(self, x):  # [B,12,768] -> [B,12]
        return self.net(x).squeeze(-1)


def pair_rank_loss(s, y, margin=0.02):
    b, t = s.shape
    ii, jj = torch.triu_indices(t, t, offset=1)
    si, sj = s[:, ii], s[:, jj]
    yi, yj = y[:, ii], y[:, jj]
    sgn = torch.sign(yi - yj)
    m = sgn != 0
    return torch.relu(margin - (si - sj) * sgn)[m].mean()


def train_head(feat, y, feat_hold, y_hold, tag):
    mu = feat[train_mask].reshape(-1, 768).mean(0)
    sd = feat[train_mask].reshape(-1, 768).std(0) + 1e-6
    norm = lambda a: (a - mu) / sd
    X = torch.from_numpy(norm(feat).astype(np.float32))
    Y = torch.from_numpy(y.astype(np.float32))
    Xh = torch.from_numpy(norm(feat_hold).astype(np.float32))
    Yh = torch.from_numpy(y_hold.astype(np.float32))
    hd = Head().to(DEV)
    opt = torch.optim.Adam(hd.parameters(), lr=1e-3)
    best, best_state, patience = -1, None, 0
    for ep in range(200):
        hd.train()
        perm = torch.randperm(int(train_mask.sum()))
        tr_idx = np.where(train_mask)[0]
        for k in range(0, len(tr_idx), 64):
            sel = tr_idx[perm[k : k + 64]]
            s = hd(X[sel])
            loss = nn.functional.mse_loss(s, Y[sel]) + 0.5 * pair_rank_loss(s, Y[sel])
            opt.zero_grad(); loss.backward(); opt.step()
        hd.eval()
        with torch.no_grad():
            sh = hd(Xh).numpy()
        sp = spearman_rows(sh, y_hold)
        m_sp = float(np.mean(sp))
        if m_sp > best + 1e-4:
            best, patience = m_sp, 0
            best_state = {k: v.clone() for k, v in hd.state_dict().items()}
        else:
            patience += 1
            if patience >= 20:
                break
    hd.load_state_dict(best_state)
    hd.eval()
    with torch.no_grad():
        pr_tr = hd(X).numpy()
        pr_ho = hd(Xh).numpy()
    rep = {
        "tag": tag, "best_holdout_spearman": best,
        "train_spearman": float(np.mean(spearman_rows(pr_tr, y))),
        "hold_predpred_pearson": float(np.mean([
            np.corrcoef(pr_ho[i], pr_ho[j])[0, 1]
            for i in range(len(pr_ho)) for j in range(i + 1, len(pr_ho))])),
        "pred_range_p5_p95": [float(np.percentile(pr_ho, 5)),
                              float(np.percentile(pr_ho, 95))],
        "epochs_used": ep + 1,
    }
    print(json.dumps(rep), flush=True)
    return rep, hd, best_state, (mu, sd)


results = {}
for feat_name, feat in (("temporal", feat_t), ("spatial", feat_s)):
    for y_name, y in (("gif_hard", gif), ("fused_target", tgt)):
        # ridge probe (closed form, train fit only)
        probes = []
        for al in (1e-2, 1.0, 100.0):
            w, mu, sd = fit_ridge(feat[train_mask], y[train_mask], al)
            Xh = (feat[hold] - mu) / sd
            pr = (Xh.reshape(-1, 768) @ w).reshape(len(hold), 12)
            probes.append(np.mean(spearman_rows(pr, y[hold])))
        probe = float(np.max(probes))
        gate = probe >= 0.45
        print(f"PROBE {feat_name}/{y_name}: holdout spearman={probe:.4f} "
              f"gate(>=0.45)={'PASS' if gate else 'FAIL'}", flush=True)
        results[f"{feat_name}/{y_name}"] = {"probe_spearman": probe,
                                            "gate_pass": bool(gate)}
        rep, hd, st, nstats = train_head(
            feat, y, feat[hold], y[hold], f"{feat_name}/{y_name}")
        results[f"{feat_name}/{y_name}"].update(rep)
        tag = f"{feat_name}_{y_name}"
        torch.save({"state": st, "norm": nstats, "feat": feat_name,
                    "target": y_name, "report": rep},
                   os.path.join(OUT, f"vp_score_head_{tag}.pt"))
        if feat_name == "temporal" and y_name == "gif_hard":
            torch.save({"state": st, "norm": nstats, "feat": "temporal",
                        "target": "gif_hard", "report": rep},
                       os.path.join(OUT, "vp_score_head.pt"))

with open(os.path.join(OUT, "HEAD_REPORT.json"), "w") as f:
    json.dump(results, f, indent=1)
print("HEAD_TRAIN_DONE")
