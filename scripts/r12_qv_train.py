#!/usr/bin/env python3
"""R12-QV P1: GATT ranker trained on QVHighlights soft saliency labels.

Single variable vs the frozen R12-GATT chain = supervision source.  Same
TRINITY-style ranker (768->512, 3-layer, 8 heads, 1D RoPE) and pairwise sign
loss, imported from r12_gatt_vmaef_train.  Sequence = 75 per-2 s clip tokens
per 150 s segment; label = union-window soft saliency in [0,1] (0 outside).

Eval: val-split AP, positives = clips inside any relevant window (same
top-n_pos precision convention as slot_ap in the PHD2 chain).

Usage: r12_qv_train.py --seed 0 --tag qv_pair_npu_s0
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

LAB = "/data/aic/experiments_910a/LFM_V11/r12_qv/qv_labels.json"
FEAT = "/data/aic/experiments_910a/LFM_V11/r12_qv/feats"
OUT = "/data/aic/experiments_910a/LFM_V11/r12_qv"
NC = 75

sys.path.insert(0, "/root/AIC/scripts")
from r12_gatt_vmaef_train import Ranker, pairwise_loss  # noqa: E402


def load_seg(vid):
    z = np.load(os.path.join(FEAT, vid + ".npz"))
    x = z["pooled"].astype(np.float32)
    T = x.shape[0]
    if T < NC:  # some segments decode <75 frames; zero-pad to fixed length
        x = np.concatenate([x, np.zeros((NC - T, 768), np.float32)], 0)
    y = np.asarray(lab["labels"][vid][:NC], dtype=np.float32)
    return x, y


@torch.no_grad()
def eval_ap(model, vids, DEV):
    """AP with positives = window clips; top-n_pos precision, mean over segs."""
    aps = []
    model.eval()
    for v in vids:
        x, y = load_seg(v)
        s = model(torch.from_numpy(x).unsqueeze(0).to(DEV))[0].cpu().numpy()
        pos = int((y > 0).sum())
        if pos == 0 or pos == len(y):
            continue
        top = set(np.argsort(-s)[:pos].tolist())
        gt = set(np.where(y > 0)[0].tolist())
        aps.append(len(top & gt) / pos)
    model.train()
    return 100.0 * float(np.mean(aps)) if aps else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-train", type=int, default=0, help="cap train segs")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", choices=["cpu", "npu"], default="npu")
    args = ap.parse_args()

    global lab
    lab = json.load(open(LAB))
    meta = lab["meta"]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    DEV = args.device
    if args.device == "npu":
        import torch_npu  # noqa: F401
        assert torch.npu.device_count() > 0, "no NPU"
        torch.npu.set_compile_mode(jit_compile=False)
        torch.npu.config.allow_internal_format = False
        DEV = "npu"

    tr = sorted(v for v, m in meta.items()
                if m["split"] == "train" and os.path.exists(os.path.join(FEAT, v + ".npz")))
    dv = sorted(v for v, m in meta.items()
                if m["split"] == "val" and os.path.exists(os.path.join(FEAT, v + ".npz")))
    if args.max_train:
        tr = tr[: args.max_train]
    keep = float(np.mean([np.mean(lab["labels"][v]) > 0 for v in tr]))
    print(f"[qvt:{args.tag}] train={len(tr)} val={len(dv)} "
          f"pos_seg_rate={keep:.3f}", flush=True)

    model = Ranker().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    os.makedirs(OUT, exist_ok=True)
    logf = open(os.path.join(OUT, f"{args.tag}.jsonl"), "a", buffering=1)
    logf.write(json.dumps({"cfg": vars(args), "n_train": len(tr), "n_val": len(dv)}) + "\n")

    t0, best = time.time(), -1.0
    for ep in range(args.epochs):
        order = np.random.permutation(len(tr))
        ep_loss, nb = 0.0, 0
        for bi in range(0, len(order), args.bs):
            batch = [tr[i] for i in order[bi: bi + args.bs]]
            segs = [load_seg(v) for v in batch]
            xs = np.stack([s[0] for s in segs])
            ys = np.stack([s[1] for s in segs])
            x = torch.from_numpy(xs).to(DEV)
            y = torch.from_numpy(ys).to(DEV)
            s = model(x)
            loss = torch.stack([pairwise_loss(s[b], y[b]) for b in range(len(batch))]).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += float(loss)
            nb += 1
        d_ap = eval_ap(model, dv, DEV)
        t_ap = eval_ap(model, [tr[i] for i in order[:200]], DEV)
        rec = {"tag": args.tag, "ep": ep, "loss": round(ep_loss / max(nb, 1), 5),
               "val_ap": round(d_ap, 2), "train_ap": round(t_ap, 2),
               "min": round((time.time() - t0) / 60, 1)}
        print("QVT " + json.dumps(rec), flush=True)
        logf.write(json.dumps(rec) + "\n")
        if d_ap == d_ap and d_ap > best:
            best = d_ap
            torch.save(model.state_dict(), os.path.join(OUT, f"{args.tag}_best.pt"))

    rec = {"tag": args.tag, "final": True, "best_val_ap": round(best, 2)}
    print("QV_FINAL " + json.dumps(rec), flush=True)
    logf.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
