#!/usr/bin/env python3
"""R104: visual-only AViMoS saliency pretraining smoke.

The AViMoS audio stream is intentionally excluded.  This trains a small
MobileNetV2 visual saliency decoder on source-disjoint videos and reports
saliency metrics against a center-prior baseline.  It is a transfer probe,
not an AIC score claim; the saved decoder can be used to produce saliency
statistics for the native crop utility head.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def metrics(pred, gt):
    p = pred.reshape(len(pred), -1).astype(np.float64); g = gt.reshape(len(gt), -1).astype(np.float64)
    p = np.maximum(p, 1e-9); p /= p.sum(1, keepdims=True); g = np.maximum(g, 1e-9); g /= g.sum(1, keepdims=True)
    cc, sim, kl, nss = [], [], [], []
    yy, xx = np.mgrid[0:gt.shape[1], 0:gt.shape[2]]
    for a, b in zip(p, g):
        aa = (a - a.mean()) / max(a.std(), 1e-9); bb = (b - b.mean()) / max(b.std(), 1e-9)
        cc.append(float(np.mean(aa * bb))); sim.append(float(np.minimum(a, b).sum())); kl.append(float(np.sum(b * np.log((b + 1e-9) / (a + 1e-9)))))
        # NSS with continuous fixation density as weights.
        nss.append(float((aa * b).sum()))
    return {"CC": float(np.mean(cc)), "SIM": float(np.mean(sim)), "KL": float(np.mean(kl)), "NSS": float(np.mean(nss))}


def load_data(root):
    xs, ys, splits, names = [], [], [], []
    for p in sorted((Path(root) / "samples").glob("*.npz")):
        z = np.load(p, allow_pickle=True); n = str(z["name"]); sp = str(z["split"])
        xs.append(z["images"]); ys.append(z["maps"]); splits.extend([sp] * len(z["images"])); names.extend([n] * len(z["images"]))
    return np.concatenate(xs), np.concatenate(ys), np.asarray(splits), np.asarray(names)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True); ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--batch", type=int, default=16); ap.add_argument("--device", default="cuda"); ap.add_argument("--seed", type=int, default=20261019); a = ap.parse_args()
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
    torch.manual_seed(a.seed); np.random.seed(a.seed); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    images, maps, splits, names = load_data(a.data); tr, dv = splits == "train", splits == "dev"
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features.to(device).eval()
    for p in backbone.parameters(): p.requires_grad_(False)
    decoder = nn.Sequential(nn.Conv2d(1280, 256, 1), nn.GELU(), nn.Conv2d(256, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, 1, 1)).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    opt = torch.optim.AdamW(decoder.parameters(), lr=2e-3, weight_decay=1e-4); rng = np.random.default_rng(a.seed); best = (1e9, None); hist = []
    def forward(ix):
        x = torch.from_numpy(images[ix]).permute(0, 3, 1, 2).float().to(device) / 255.; x = (x - mean) / std
        with torch.no_grad(): h = backbone(x)
        return F.interpolate(decoder(h), size=maps.shape[1:], mode="bilinear", align_corners=False).squeeze(1)
    ids = np.flatnonzero(tr)
    for ep in range(a.epochs):
        rng.shuffle(ids); decoder.train(); losses=[]
        for st in range(0, len(ids), a.batch):
            ix=ids[st:st+a.batch]; logits=forward(ix); target=torch.from_numpy(maps[ix]).float().to(device); logp=logits.flatten(1).log_softmax(1); loss=-(target.flatten(1)*logp).sum(1).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        decoder.eval(); with_no = torch.no_grad()
        with with_no:
            logits=forward(np.flatnonzero(dv)); pred=logits.flatten(1).softmax(1).cpu().numpy().reshape((-1,)+maps.shape[1:])
        met=metrics(pred,maps[dv]); hist.append({"epoch":ep+1,"loss":float(np.mean(losses)),**met})
        if met["KL"] < best[0]: best=(met["KL"],{k:v.detach().cpu().clone() for k,v in decoder.state_dict().items()})
        print(json.dumps(hist[-1]),flush=True)
    decoder.load_state_dict(best[1]); torch.save(decoder.state_dict(),out/"decoder.pt")
    # Center Gaussian-like prior is represented by a uniform center disk only
    # through the empirical mean map, which is a strong and reproducible prior.
    center = np.broadcast_to(maps[tr].mean(0), maps[dv].shape); base=metrics(center,maps[dv])
    report={"experiment_id":"R104_AVIMOS_VISUAL_SALIENCY","videos":int(len(set(names))),"train_videos":int(len(set(names[tr]))),"dev_videos":int(len(set(names[dv]))),"train_samples":int(tr.sum()),"dev_samples":int(dv.sum()),"visual_only":True,"audio":"excluded_due_to_project_negative_transfer_evidence","decoder_params":sum(p.numel() for p in decoder.parameters()),"best_dev":hist[np.argmin([h["KL"] for h in hist])],"dev_center_mean_map":base,"history":hist}
    (out/"metrics.json").write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2))


if __name__ == "__main__": main()
