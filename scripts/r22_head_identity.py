#!/usr/bin/env python3
"""R22 阶段1 第 4 项：VPB 评分头身份恢复 + 重现旧读数。

## 为什么需要这个脚本

计划 §2.4：VPB checkpoint 含 `norm=(mu,sd)`、原训练用 GELU/标准化；
**被审计的副本改成 ReLU 且漏掉标准化**。计划 §4 第 4 项要求
「恢复 `norm=(mu,sd)` 与 GELU，在做任何历史预测器对比前，先证明恢复后能重现旧读数」。

`vp_score_head.pt` 自带的 `report.best_holdout_spearman` 就是那条旧读数。
本脚本在**训练时的干净缓存视图**上复算它 —— 一次精确命中同时验证五件事：
头身份（激活）、输入归一化、特征缓存可重绑、行对齐可重建、holdout 划分。

## 四处已定位的身份分歧

  train_vp_head.py:86    nn.Linear(d,512), nn.GELU(), nn.Linear(512,1)   训练真身
  train_vp_head.py:106   norm = lambda a: (a - mu) / sd                   训练真身
  extract_features.py:65 np.stack(imgs)/255.0                             训练真身
  e01_vpb_uniform_eval.py:42  nn.ReLU() 且**:110 完全不做标准化**          ← 副本
  e01_vpb_uniform_eval.py:93  /127.5 - 1.0                                 ← 副本
  e01_eval.py:32        nn.ReLU()，**:60 /255.0 但不做标准化**             ← 副本
  resid_e03.py:46       nn.ReLU() 且**:96 未施加 ckpt 的 norm**            ← 副本

用法：
  python r22_head_identity.py --sft <sft_data_v2.jsonl> --feat-dir <feat> \
      --head <vp_score_head.pt> --out <dir>
"""
import argparse
import glob
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PER = 627
SEED, NHOLD = 13, 125


def rankdata(a):
    a = np.asarray(a, np.float64)
    o = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), np.float64)
    sa = a[o]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def spearman_rows(pred, target, idx):
    out = []
    for i in idx:
        p, t = pred[i], target[i]
        if np.std(p) < 1e-6 or np.std(t) < 1e-6:
            continue
        out.append(np.corrcoef(rankdata(p), rankdata(t))[0, 1])
    return float(np.mean(out))


class Head(nn.Module):
    def __init__(self, d, act):
        super().__init__()
        A = nn.GELU() if act == "gelu" else nn.ReLU()
        self.net = nn.Sequential(nn.Linear(d, 512), A, nn.Linear(512, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main(a):
    rows = [json.loads(l) for l in open(a.sft)]
    sft_vids = [r["vid"] for r in rows]
    N = len(rows)
    gif = np.array([r["gif_hard"] for r in rows], np.float32)

    shard_files = sorted(glob.glob(os.path.join(a.feat_dir, "shard_*of*.npz")))
    vt, vs, vv = [], [], []
    for f in shard_files:
        z = np.load(f)
        v = z["vid"].tolist()
        vt.append(z["feat_temporal"][:PER])
        vs.append(z["feat_spatial"][:PER])
        vv.extend(v[:PER])
    ft = np.concatenate(vt)
    fs = np.concatenate(vs)
    assert len(vv) == N and len(set(vv)) == N, "干净视图重建失败"
    assert set(vv) == set(sft_vids), "vid 集合 != SFT"
    idx = {v: k for k, v in enumerate(sft_vids)}
    order = np.argsort([idx[v] for v in vv])
    ft, fs = ft[order], fs[order]
    assert [vv[i] for i in order] == sft_vids, "逐行对齐失败"

    ck = torch.load(a.head, map_location="cpu", weights_only=False)
    mu = torch.from_numpy(np.asarray(ck["norm"][0], np.float32))
    sd = torch.from_numpy(np.asarray(ck["norm"][1], np.float32))
    expect = ck["report"]["best_holdout_spearman"]
    hold = sorted(np.random.default_rng(SEED).permutation(N)[:NHOLD].tolist())

    variants = [
        ("A_restored_identity_norm_gelu", ft, "gelu", True, "恢复的真身：训练用路径"),
        ("A2_norm_but_relu", ft, "relu", True, "只换激活"),
        ("B_audited_copy_no_norm_relu", ft, "relu", False, "被审计副本的做法"),
        ("C_no_norm_gelu", ft, "gelu", False, "只去归一化"),
        ("A_spatial_restored", fs, "gelu", True, "用 spatial 特征（对照）"),
        ("B_spatial_audited", fs, "relu", False, "spatial + 审计副本（对照）"),
    ]
    out = {"sft_sha256": hashlib.sha256(Path(a.sft).read_bytes()).hexdigest(),
           "head": str(a.head), "head_report": ck["report"],
           "head_target": ck["target"], "head_feat": ck["feat"],
           "checkpoint_recorded_value": expect,
           "norm_sd_min": float(sd.min()), "norm_sd_zeros": int((sd == 0).sum()),
           "holdout_seed": SEED, "holdout_n": len(hold), "variants": {}}
    for name, feat, act, use_norm, note in variants:
        hd = Head(feat.shape[-1], act)
        hd.load_state_dict(ck["state"])
        hd.eval()
        X = torch.from_numpy(feat.astype(np.float32))
        if use_norm:
            X = (X - mu) / sd
        with torch.no_grad():
            pr = hd(X).numpy()
        sp = spearman_rows(pr, gif, hold)
        out["variants"][name] = {"spearman": sp, "delta_vs_ckpt": sp - expect,
                                 "note": note}
        print(f"{name:34s} {sp:.10f}  Δ={sp - expect:+.6f}")

    d = out["variants"]["A_restored_identity_norm_gelu"]["delta_vs_ckpt"]
    out["acceptance"] = {
        "criterion": "恢复身份复算 − ckpt 记录的 best_holdout_spearman，|Δ| < 1e-6",
        "delta": d,
        "verdict": "PASS" if abs(d) < 1e-6 else "FAIL",
        "what_a_pass_simultaneously_verifies": [
            "头激活身份（GELU）", "特征标准化 (x-mu)/sd 由 ckpt 携带",
            "输入预处理 /255.0", "特征缓存可重绑到训练时干净视图",
            "行对齐可重建", "holdout 划分 default_rng(13)[:125]",
        ],
        "audited_copy_bias": out["variants"]["B_audited_copy_no_norm_relu"]["delta_vs_ckpt"],
    }
    if not a.allow_deviation:
        assert abs(d) < 1e-6, f"未精确重现：Δ={d:+.3e}；如需调查用 --allow-deviation"

    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "r22_head_identity.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print("HEAD_IDENTITY_DONE", out["acceptance"]["verdict"], f"Δ={d:+.3e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sft", type=Path, required=True)
    p.add_argument("--feat-dir", type=Path, required=True)
    p.add_argument("--head", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--allow-deviation", action="store_true")
    main(p.parse_args())
