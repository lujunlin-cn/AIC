#!/usr/bin/env python3
"""R101: real low-resolution temporal RGB context for spatial utility.

R70 provides the high-resolution/current-frame candidate tensor and native crop
IoU targets.  This probe adds a genuinely separate frozen ImageNet ResNet18
embedding from neighbouring keyframes (2 on each side), then conditions the
candidate readout through a rank-32 interaction.  The control has identical
parameters but receives zero context; a source-permuted context is evaluated as
a negative control.

It is still a public native-crop diagnostic, not a submission package: the
frozen ResNet is an additional inference component and no official labels are
used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


def load_shards(root):
    parts = []
    for p in sorted(Path(root).glob("shard_*.npz")):
        with np.load(p, allow_pickle=True) as z:
            parts.append({k: z[k] for k in z.files})
    if not parts:
        raise FileNotFoundError(root)
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}


def valid_keyframes(table_root: Path, split: str):
    out = {}
    tab = pickle.loads((table_root / "tables" / f"{split}.pkl").read_bytes())
    for u in tab:
        ratio = "-".join(str(int(x)) for x in u["ratio"])
        key = (split, f"{u['ds']}:{u['vid']}", ratio)
        gf = set(int(x) for x in np.asarray(u["gf"]).tolist())
        out[key] = [int(u["keys"][int(j)]) for j in u["J"] if int(u["keys"][int(j)]) in gf]
    return out


def make_paths(z, table_root: Path, rv_root: Path, live_root: Path):
    source = z["source"].astype(str); ratio = z["ratio"].astype(str); name = z["name"].astype(str)
    valid = {}
    for split in sorted(set(name)):
        valid.update(valid_keyframes(table_root, split))
    paths = [None] * len(source); groups = {}
    for i, (sp, src, rat) in enumerate(zip(name, source, ratio)):
        groups.setdefault((sp, src, rat), []).append(i)
    for key, ids in groups.items():
        kfs = valid.get(key)
        if kfs is None or len(kfs) != len(ids):
            raise ValueError(f"row/keyframe mismatch {key}: rows={len(ids)} kfs={0 if kfs is None else len(kfs)}")
        ds, vid = key[1].split(":", 1)
        root = live_root if ds == "live" else rv_root
        for i, kf in zip(ids, kfs):
            if ds == "live":
                p = root / vid / f"{kf}.png"
            else:
                p = root / "d1" / vid / f"{kf}.png"
            if not p.is_file():
                raise FileNotFoundError(str(p))
            paths[i] = p
    return paths, groups


def extract_embeddings(paths, weights: Path, device: str, batch: int):
    import torch
    from PIL import Image
    from torchvision.models import resnet18
    from torchvision.transforms import v2 as T
    model = resnet18(weights=None); model.fc = torch.nn.Identity()
    state = torch.load(weights, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    state = {k.removeprefix("module.").removeprefix("backbone."): v for k, v in state.items()}
    model.load_state_dict(state, strict=False); model.to(device).eval()
    tfm = T.Compose([T.ToImage(), T.Resize((224, 224)), T.ToDtype(torch.float32, scale=True),
                     T.Normalize([.485, .456, .406], [.229, .224, .225])])
    out = np.zeros((len(paths), 512), np.float32)
    for i in range(0, len(paths), batch):
        ims = torch.stack([tfm(Image.open(p).convert("RGB")) for p in paths[i:i + batch]])
        with torch.inference_mode():
            out[i:i + len(ims)] = model(ims.to(device)).float().cpu().numpy()
    return out


def neighbour_context(emb, groups):
    c = np.zeros_like(emb, dtype=np.float32)
    for ids in groups.values():
        if len(ids) < 2:
            continue
        for pos, i in enumerate(ids):
            nb = ids[max(0, pos - 2):pos] + ids[pos + 1:min(len(ids), pos + 3)]
            if nb:
                c[i] = emb[np.asarray(nb)].mean(0)
    return c


def units(z):
    return np.asarray([f"{n}|{s}|{r}" for n, s, r in zip(z["name"].astype(str), z["source"].astype(str), z["ratio"].astype(str))])


def evaluate(pred, y, unit, mask):
    sel = pred.argmax(1); val = y[np.arange(len(y)), sel]; per = {}
    for i in np.flatnonzero(mask): per.setdefault(str(unit[i]), []).append(float(val[i]))
    per = {k: float(np.mean(v)) for k, v in per.items()}
    return {"rows": int(mask.sum()), "units": len(per), "macro_keyframe_iou": float(np.mean(list(per.values()))), "per_unit": per}


def paired(a, b, seed=20261019, n=10000):
    ids = sorted(set(a) & set(b)); d = np.asarray([b[k] - a[k] for k in ids], float); rng = np.random.default_rng(seed)
    bs = np.asarray([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return {"units": len(d), "mean": float(d.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "better": int((d > 0).sum()), "worse": int((d < 0).sum())}


class Net:
    @staticmethod
    def build(mode, rank=32):
        import torch.nn as nn
        class M(nn.Module):
            def __init__(self):
                super().__init__(); self.mode = mode; self.rank = rank
                self.cur = nn.Sequential(nn.LayerNorm(2305), nn.Linear(2305, 128), nn.GELU())
                self.ctx = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 128), nn.GELU())
                self.u = nn.Linear(128, rank); self.v = nn.Linear(128, rank)
                self.head = nn.Sequential(nn.Linear(128 + rank, 96), nn.GELU(), nn.Linear(96, 1))
            def forward(self, x, c):
                x = x.float(); c = c.float(); h = self.cur(x); g = self.ctx(c)
                if self.mode == "control": q = h.new_zeros((*h.shape[:-1], self.rank))
                else: q = self.u(h) * self.v(g).unsqueeze(1)
                return self.head(__import__("torch").cat([h, q], -1)).squeeze(-1)
        return M()


def predict(net, x, c, device, batch):
    import torch
    out = []
    with torch.inference_mode():
        for i in range(0, len(x), batch):
            out.append(net(torch.from_numpy(x[i:i + batch]).to(device), torch.from_numpy(c[i:i + batch]).to(device)).float().cpu().numpy())
    return np.concatenate(out)


def train_one(x, c, y, tr, unit, mode, seed, steps, device, batch):
    import torch
    torch.manual_seed(seed); rng = np.random.default_rng(seed); m = Net.build(mode).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-3); ids = np.flatnonzero(tr); best = (-1, None); hist = []
    xt = torch.from_numpy(x); ct = torch.from_numpy(c); yt = torch.from_numpy(y)
    for step in range(1, steps + 1):
        take = ids[rng.integers(0, len(ids), batch)]; pred = m(xt[take].to(device), ct[take].to(device)); loss = ((pred - yt[take].to(device)) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.); opt.step()
        if step % 100 == 0 or step == steps:
            m.eval(); allp = predict(m, x, c, device, batch); ev = evaluate(allp, y, unit, ~tr); hist.append({"step": step, "loss": float(loss.detach().cpu()), "dev": ev["macro_keyframe_iou"]})
            if ev["macro_keyframe_iou"] > best[0]: best = (ev["macro_keyframe_iou"], {k: v.detach().cpu().clone() for k, v in m.state_dict().items()})
            m.train()
    m.load_state_dict(best[1]); m.eval(); allp = predict(m, x, c, device, batch)
    return m, allp, {"mode": mode, "params": int(sum(p.numel() for p in m.parameters())), "train": evaluate(allp, y, unit, tr), "dev": evaluate(allp, y, unit, ~tr), "history": hist}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--shards", type=Path, required=True); ap.add_argument("--table-root", type=Path, required=True); ap.add_argument("--rv-root", type=Path, required=True); ap.add_argument("--live-root", type=Path, required=True); ap.add_argument("--resnet", type=Path, required=True); ap.add_argument("--output", type=Path, required=True); ap.add_argument("--steps", type=int, default=1200); ap.add_argument("--batch", type=int, default=8); ap.add_argument("--embed-batch", type=int, default=64); ap.add_argument("--device", default="cuda"); a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    z = load_shards(a.shards); x = z["x"].astype(np.float16); y = z["y"].astype(np.float32); tr = np.isin(z["name"].astype(str), ["rv_train", "live_train"]); unit = units(z); paths, groups = make_paths(z, a.table_root, a.rv_root, a.live_root)
    emb = extract_embeddings(paths, a.resnet, a.device, a.embed_batch); c = neighbour_context(emb, groups).astype(np.float16)
    models = {}
    for mode in ("control", "interaction"):
        m, p, met = train_one(x, c, y, tr, unit, mode, 20261019, a.steps, a.device, a.batch); models[mode] = met
        import torch; torch.save({"state_dict": m.state_dict(), "mode": mode, "params": met["params"]}, a.output / f"{mode}.pt")
    base = models["control"]["dev"]["per_unit"]; main = {"interaction": paired(base, models["interaction"]["dev"]["per_unit"])}
    m = Net.build("interaction").to(a.device); import torch; m.load_state_dict(torch.load(a.output / "interaction.pt", map_location="cpu", weights_only=True)["state_dict"]); m.eval(); idx = np.flatnonzero(~tr); perm = idx.copy(); np.random.default_rng(20261019).shuffle(perm); cp = c.copy(); cp[idx] = c[perm]; pp = predict(m, x, cp, a.device, a.batch); pem = evaluate(pp, y, unit, ~tr); main["interaction_permuted_context"] = paired(base, pem["per_unit"])
    out = {"experiment_id": "R101_SPATIAL_RESNET_CONTEXT", "protocol": "R70 native candidate utility + frozen ResNet18 neighbouring keyframe context; source-disjoint RV/LIVE split", "official_test_labels_used": False, "audio": "excluded_due_to_project_negative_transfer_evidence", "resnet_sha256": hashlib.sha256(a.resnet.read_bytes()).hexdigest(), "rows": int(len(y)), "train_rows": int(tr.sum()), "dev_rows": int((~tr).sum()), "units": int(len(set(unit))), "models": models, "paired_dev": main, "permuted_dev": pem, "decision": {"official_package": False, "note": "diagnostic only; current-frame V9 candidate features plus separate low-resolution RGB context"}}
    (a.output / "metrics.json").write_text(json.dumps(out, indent=2) + "\n"); print(json.dumps({"models": {k: {"dev": v["dev"]} for k, v in models.items()}, "paired_dev": main}, indent=2))


if __name__ == "__main__": main()
