"""R11 S2 readouts a/b/c: detector boxes as new candidate source on the LIVE pool.

Inputs
  s1_score_table.npz   frozen B3 score field (pred/u per row, 129 candidates)
  v8 samples npz       (pool, vid, frame) row keys, identical row order to the table
  v8_manifest.jsonl    spatial GT boxes per row
  r11_s2a_live/        RT-DETR top-3 person boxes per (vid, frame), same
                       detection contract as the day-1 PHD2 run

Readouts (plan: reports/20261007_round11_plan.md S2, step 3)
  a. new-pool oracle (det window U 129-grid) - old-pool oracle (129-grid):
     does the candidate ceiling rise?            gate: >= +0.02
  b. agreement: B3 argmax window vs det window IoU >= thr rate (is the
     detector just information B3 already has?)
  c. conditional re-rank: score' = pred + beta * iou(det_win, grid_win_j),
     beta tuned on dev pool, applied ONCE to each eval pool, decoded with
     the S1 Viterbi (lam from s1_audit).           gate: >= +0.005, CI>0

Pure offline CPU.  IoU/win construction copied from v8_build_samples /
scripts.benchmark_spatial so u_det is on exactly the same scale as u.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "16")

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--table", type=Path,
                default=Path("/data/aic/experiments_910a/LFM_V11/r11/s1_score_table.npz"))
ap.add_argument("--samples-dir", type=Path,
                default=Path("/data/aic/experiments_910a/LFM_V8/samples"))
ap.add_argument("--manifest", type=Path,
                default=Path("/data/aic/experiments_910a/LFM_V8/v8_manifest.jsonl"))
ap.add_argument("--det-root", type=Path,
                default=Path("/data/aic/experiments_910a/LFM_V11/r11_s2a_live"))
ap.add_argument("--s1-audit", type=Path,
                default=Path("/data/aic/experiments_910a/LFM_V11/r11/s1_audit.json"))
ap.add_argument("--out", type=Path,
                default=Path("/data/aic/experiments_910a/LFM_V11/r11/s2a_readouts.json"))
ap.add_argument("--dev-pool", default="live_dev")
ap.add_argument("--eval-pools", nargs="+", default=["live_confirmation"])
ap.add_argument("--agree-thr", type=float, default=0.5)
ap.add_argument("--betas", nargs="+", type=float, default=[0, 0.25, 0.5, 1, 2])
args = ap.parse_args()

sys.path.insert(0, "/root/AIC")
from scripts.benchmark_spatial import iou  # noqa: E402  (same fn that built u)
from aic.max_window_path import geometry  # noqa: E402
from aic.r11_trajectory import (RATIOS, cluster_ci, decode_pool, load_table,
                                paired_per_source)  # noqa: E402

NC = 129


def geometry_of(row):
    W, H = float(row["W"]), float(row["H"])
    w, h, axis = geometry(W, H, RATIOS[row["ratio"]])
    span = (W - w) if axis == 0 else ((H - h) if axis == 1 else 0.0)
    return W, H, w, h, axis, span


def grid_windows(row, n_cand=NC):
    """(n,4) win_px for one manifest row - verbatim v8_build_samples geometry."""
    W, H, w, h, axis, span = geometry_of(row)
    offs = np.linspace(0, span, n_cand) if span > 0 else np.zeros(1)
    win = np.zeros((len(offs), 4))
    for j, o in enumerate(offs):
        if axis == 0:
            win[j] = [o, 0, o + w, h]
        elif axis == 1:
            win[j] = [0, o, w, o + h]
        else:
            win[j] = [0, 0, W, H]
    return win


def det_window(row, box_cxcywh):
    """Crop window of the row's ratio centred on the detector box (px, clipped)."""
    W, H, w, h, axis, _ = geometry_of(row)
    cx, cy = float(box_cxcywh[0]) * W, float(box_cxcywh[1]) * H
    if axis == 0:
        x0 = min(max(cx - w / 2.0, 0.0), W - w)
        return np.array([x0, 0.0, x0 + w, h], dtype=np.float64)
    if axis == 1:
        y0 = min(max(cy - h / 2.0, 0.0), H - h)
        return np.array([0.0, y0, w, y0 + h], dtype=np.float64)
    return np.array([0.0, 0.0, W, H], dtype=np.float64)


def mean_iou(win, gts):
    """One window vs all GT boxes, mean - same reduction as v8 u."""
    return float(iou(win[None, :].astype(np.float64),
                     np.asarray(gts, dtype=np.float64)[None, ...]).mean())


def main():
    tab = load_table(args.table)
    pred_all, u_all = tab["pred"], tab["u"]
    vid_all, ratio_all, pool_all, ord_all = tab["vid"], tab["ratio"], tab["pool"], tab["ord"]
    pos_all = tab.get("pos")

    gt_of = {}
    for line in args.manifest.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        gt_of[(r["split"], r["vid"], int(r["frame"]))] = r

    res = {"protocol": "R11 S2a readouts a/b/c on LIVE pool; det=RT-DETR r50vd top-1",
           "dev_pool": args.dev_pool, "eval_pools": args.eval_pools,
           "agree_thr": args.agree_thr}

    # ---- per-row detector terms; rows[k]=None when detection missing/absent
    cov = {}
    det_rows = {}
    for pool in [args.dev_pool] + args.eval_pools:
        with np.load(args.samples_dir / f"{pool}.npz", allow_pickle=True) as z:
            vids = [str(v) for v in z["vid"]]
            frames = [int(f) for f in z["frame"]]
        m = pool_all == pool
        assert m.sum() == len(vids), f"{pool}: table/samples row mismatch"
        rows, n_ok, n_tot = [], 0, 0
        for vid, fr in zip(vids, frames):
            man = gt_of.get((pool, vid, fr))
            dpz = args.det_root / f"{vid}_{fr:06d}.npz"
            if man is None or not dpz.exists():
                rows.append(None)
                continue
            with np.load(dpz) as z:
                ok = int(z["ok"])
                box1 = z["det_box"][0].astype(np.float64).copy()
                conf1 = float(z["det_conf"][0])
                box3 = z["det_box"].astype(np.float64)
                conf3 = z["det_conf"].astype(np.float64)
            n_tot += 1
            if not ok or conf1 < 0.3:
                rows.append(None)
                continue
            gwin = grid_windows(man)
            dwin = det_window(man, box1)
            u_det = mean_iou(dwin, man["gt"])
            prior = iou(np.broadcast_to(dwin, gwin.shape), gwin.astype(np.float64))
            argmax = int(np.argmax(pred_all[m][len(rows)]))
            agree = float(iou(gwin[argmax][None, :], dwin[None, :])[0]) >= args.agree_thr
            u_det3 = max((mean_iou(det_window(man, b), man["gt"])
                          for b, cf in zip(box3, conf3) if cf >= 0.3), default=u_det)
            rows.append({"u_det": u_det, "u_det3": u_det3, "prior": prior, "agree": agree})
            n_ok += 1
        det_rows[pool] = rows
        cov[pool] = {"rows_with_det": n_ok, "rows": n_tot,
                     "frac": round(n_ok / max(n_tot, 1), 4)}
    res["det_coverage"] = cov

    # ---------------- readout a: oracle lift (gate >= +0.02) ----------------
    res["a_oracle"] = {}
    for pool in [args.dev_pool] + args.eval_pools:
        m = pool_all == pool
        u, rows, vids = u_all[m], det_rows[pool], vid_all[m]
        d1, d3 = {}, {}
        for i in range(len(u)):
            if rows[i] is None:
                continue
            v = str(vids[i])
            old = float(u[i].max())
            e1, e3, cnt = d1.get(v, (0.0, 0.0, 0))
            d1[v] = (e1 + (max(old, rows[i]["u_det"]) - old),
                     e3 + (max(old, rows[i]["u_det3"]) - old), cnt + 1)
        mean1, ci1 = cluster_ci({v: e1 / c for v, (e1, e3, c) in d1.items()})
        mean3, ci3 = cluster_ci({v: e3 / c for v, (e1, e3, c) in d1.items()})
        res["a_oracle"][pool] = {
            "n_rows": sum(c for _, _, c in d1.values()), "n_src": len(d1),
            "det_top1_mean_lift": mean1, "det_top1_ci95": ci1, "gate_ge_002": bool(mean1 >= 0.02),
            "det_top3_mean_lift": mean3, "det_top3_ci95": ci3}

    # ---------------- readout b: agreement with B3 argmax -------------------
    res["b_agreement"] = {}
    for pool in [args.dev_pool] + args.eval_pools:
        rows = det_rows[pool]
        hits = [r["agree"] for r in rows if r is not None]
        res["b_agreement"][pool] = {"n": len(hits),
                                    "rate": float(np.mean(hits)) if hits else None}

    # ---------------- readout c: conditional re-rank (gate >= +0.005 CI>0) --
    lam = float(json.loads(args.s1_audit.read_text()).get("selected_lam", 0.0))

    def prior_matrix(m, rows):
        P = np.zeros((int(m.sum()), pred_all.shape[1]), dtype=np.float32)
        for k, r in enumerate(rows):
            if r is not None:
                P[k] = r["prior"]
        return P

    def eval_decode(m, rows, beta):
        P = prior_matrix(m, rows)
        pos = pos_all[m] if pos_all is not None else None
        base, dp = decode_pool(pred_all[m] + beta * P, u_all[m], vid_all[m],
                               ratio_all[m], ord_all[m], lam, pos=pos)
        per = paired_per_source(u_all[m], vid_all[m], ratio_all[m], ord_all[m], base, dp)
        return cluster_ci(per)

    dev_m = pool_all == args.dev_pool
    res["c_rerank"] = {"lam_from_s1": lam, "betas": args.betas, "dev_sweep": {}}
    for b in args.betas:
        mean, ci = eval_decode(dev_m, det_rows[args.dev_pool], b)
        res["c_rerank"]["dev_sweep"][str(b)] = {"mean": mean, "ci95": ci}
    best_beta = max(args.betas, key=lambda k: res["c_rerank"]["dev_sweep"][str(k)]["mean"])
    res["c_rerank"]["selected_beta"] = float(best_beta)

    res["c_rerank"]["eval"] = {}
    for pool in args.eval_pools:
        mean, ci = eval_decode(pool_all == pool, det_rows[pool], best_beta)
        res["c_rerank"]["eval"][pool] = {
            "mean_delta": mean, "ci95": ci, "gate_ge_005_ci_pos": bool(mean >= 0.005 and ci[0] > 0)}

    args.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("det_coverage", "a_oracle", "b_agreement",
                                          "c_rerank")}, indent=1))


if __name__ == "__main__":
    main()
