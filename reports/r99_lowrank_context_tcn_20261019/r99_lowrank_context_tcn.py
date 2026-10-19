#!/usr/bin/env python3
"""R99: low-rank local--global interaction for target-domain time ranking.

This is the first implementation of the context mechanism proposed in
``video_competitions_research_and_agent_handoff.md``.  A per-frame RGB
embedding is conditioned on the mean representation of the same source video
through a low-rank Hadamard product.  A matched control has the same projection
parameters but receives a zero interaction vector.  A source-permuted global
vector is a negative control: it should remove any source-specific benefit if
the interaction is real rather than a regularizer or a position shortcut.

The labels are public PHD2/GIF observed selections.  This is a mechanism
diagnostic, not an AIC ground-truth evaluator and it never reads official
test labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def fixed_f(scores, y, keep=.85):
    k = max(1, int(round(keep * len(y))))
    idx = np.argsort(-scores, kind="mergesort")[:k]
    return float(2 * y[idx].sum() / max(k + y.sum(), 1e-12))


def evaluate(scores, y, src, mask):
    per = {}
    for i, name in enumerate(src[mask]):
        per.setdefault(str(name), []).append(fixed_f(scores[mask][i], y[mask][i]))
    per = {k: float(np.mean(v)) for k, v in per.items()}
    return {"rows": int(mask.sum()), "sources": len(per),
            "macro_f": float(np.mean(list(per.values()))), "per_source": per}


def paired_bootstrap(a, b, seed=20261019, n=10000):
    ids = sorted(set(a) & set(b))
    d = np.asarray([b[k] - a[k] for k in ids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = np.asarray([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return {"sources": len(d), "mean": float(d.mean()),
            "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
            "better": int((d > 0).sum()), "worse": int((d < 0).sum())}


class ContextNet:
    @staticmethod
    def build(mode: str, rank: int = 32):
        import torch
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.mode = mode
                self.rank = rank
                self.proj = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 64), nn.GELU())
                self.temporal = nn.Conv1d(64, 128, 1)
                self.blocks = nn.ModuleList([nn.Conv1d(128, 128, 3, padding=d, dilation=d)
                                             for d in (1, 2, 4)])
                self.norm = nn.GroupNorm(8, 128)
                # All modes instantiate the same context projection capacity.
                self.local_u = nn.Linear(64, rank)
                self.global_v = nn.Linear(64, rank)
                self.global_linear = nn.Linear(64, rank)
                self.head = nn.Sequential(nn.Linear(128 + 64 + rank, 64), nn.GELU(), nn.Linear(64, 1))

            def forward(self, x, permute_global=None):
                z = self.proj(x)
                h = self.temporal(z.transpose(1, 2))
                for c in self.blocks:
                    h = torch.nn.functional.gelu(self.norm(c(h) + h))
                h = h.transpose(1, 2)
                g = z.mean(1)
                if permute_global is not None:
                    g = g[permute_global]
                if self.mode == "matched_control":
                    extra = torch.zeros((len(x), x.shape[1], self.rank), device=x.device, dtype=x.dtype)
                elif self.mode == "linear_global":
                    extra = self.global_linear(g).unsqueeze(1).expand(-1, x.shape[1], -1)
                elif self.mode == "interaction":
                    extra = self.local_u(z) * self.global_v(g).unsqueeze(1)
                else:
                    raise ValueError(self.mode)
                q = torch.cat([h, z, extra], -1)
                return self.head(q).squeeze(-1)

        return Net()


def train_one(X, Y, src, split, mode, seed, steps, device, rank):
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = ContextNet.build(mode, rank).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=.01)
    tr = split == "train"
    ids = np.flatnonzero(tr)
    Xt = torch.from_numpy(X); Yt = torch.from_numpy(Y)
    best = (-1., None); hist = []
    for step in range(1, steps + 1):
        sel = ids[rng.integers(0, len(ids), 64)]
        xb = Xt[sel].to(device); yb = Yt[sel].to(device)
        logits = net(xb)
        roww = torch.where(yb.sum(1) > 0, torch.ones(len(yb), device=device),
                           torch.full((len(yb),), .15, device=device))
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(
            logits, yb, reduction="none").mean(1) * roww).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.); opt.step()
        if step % 100 == 0 or step == steps:
            net.eval()
            with torch.inference_mode():
                pred = net(Xt.to(device)).sigmoid().cpu().numpy()
            m = evaluate(pred, Y, src, ~tr)
            hist.append({"step": step, "loss": float(loss.detach().cpu()),
                         "dev_macro_f": m["macro_f"]})
            if m["macro_f"] > best[0]:
                best = (m["macro_f"], {k: v.detach().cpu().clone()
                                       for k, v in net.state_dict().items()})
            net.train()
    net.load_state_dict(best[1]); net.eval()
    with torch.inference_mode():
        pred = net(Xt.to(device)).sigmoid().cpu().numpy()
    return net, pred, {"mode": mode, "seed": seed,
                       "params": int(sum(p.numel() for p in net.parameters())),
                       "train": evaluate(pred, Y, src, tr),
                       "dev": evaluate(pred, Y, src, ~tr), "history": hist}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    z = np.load(args.cache, allow_pickle=False)
    X = z["features"][:, :, 0, :].astype(np.float32)
    Y = z["labels"].astype(np.float32)
    src = z["src"].astype(str); split = z["split"].astype(str)
    models = {}
    for mode in ("matched_control", "linear_global", "interaction"):
        net, pred, met = train_one(X, Y, src, split, mode, args.seed,
                                   args.steps, args.device, args.rank)
        models[mode] = met
        torch.save({"state_dict": net.state_dict(), "mode": mode,
                    "rank": args.rank, "params": met["params"]},
                   args.output / f"{mode}.pt")
    base = models["matched_control"]["dev"]["per_source"]
    comp = {m: paired_bootstrap(base, models[m]["dev"]["per_source"])
            for m in ("linear_global", "interaction")}
    # The permutation control uses the fitted interaction head and is evaluated
    # only on held-out rows; no extra training or selection is allowed.
    net = ContextNet.build("interaction", args.rank).to(args.device)
    net.load_state_dict(torch.load(args.output / "interaction.pt", map_location="cpu",
                                   weights_only=True)["state_dict"]); net.eval()
    Xt = torch.from_numpy(X).to(args.device)
    rng = np.random.default_rng(args.seed)
    perm_src = np.flatnonzero(split == "dev")
    # Permute complete source rows, preserving the 8-frame sequence internally.
    perm = perm_src.copy(); rng.shuffle(perm)
    lookup = {int(i): int(j) for i, j in zip(perm_src, perm)}
    idx = np.arange(len(X)); perm_global = np.asarray([lookup.get(int(i), int(i)) for i in idx])
    with torch.inference_mode():
        perm_pred = net(Xt, torch.from_numpy(perm_global).to(args.device)).sigmoid().cpu().numpy()
    perm_eval = evaluate(perm_pred, Y, src, split == "dev")
    comp["interaction_permuted_global"] = paired_bootstrap(
        models["matched_control"]["dev"]["per_source"], perm_eval["per_source"])
    out = {"experiment_id": "R99_LOWRANK_CONTEXT_TCN",
           "protocol": "R92 RGB view + public PHD2/GIF observed-selection labels; source split frozen",
           "paper_transfer": ["YouTube-8M CCRL", "YouTube-8M MOD"],
           "official_test_labels_used": False,
           "audio": "excluded_due_to_project_negative_transfer_evidence",
           "cache_sha256": hashlib.sha256(args.cache.read_bytes()).hexdigest(),
           "rows": int(len(Y)), "sources": int(len(set(src))),
           "train_sources": int(len(set(src[split == "train"]))),
           "dev_sources": int(len(set(src[split == "dev"]))),
           "models": models, "paired_dev": comp,
           "permuted_dev": perm_eval,
           "decision": {"official_package": False,
                        "note": "diagnostic only; observed GIF selections are not complete AIC GT"}}
    (args.output / "metrics.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"models": {k: {"dev": v["dev"]} for k, v in models.items()},
                      "paired_dev": comp}, indent=2))


if __name__ == "__main__":
    main()
