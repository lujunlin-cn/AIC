"""R11 O1-scout / V1 features: E3 event-role JSONs -> per-frag feature table.

Turns the V4-E3 teacher event roles (MAIN/SETUP/RESULT/IDLE/JUNK per 2 s
segment, scripts/qwen_event_segments.py output) into the R10 readout
contract, so `r10_audio_head.py --feature e3_role --feat-root <out>` runs
UNCHANGED:

    <out>/<frag>.npz : e3_role (8, 5) f4   one-hot role block per slot
                       slot_cov (8,) f4    role coverage per slot (0..1)
                       keep_prior (8,) f4  keep-family prior (MAIN/SETUP/RESULT=1)
                       t (8,) f4           slot centre seconds (copied)
    e3_missing.json  : frags without role coverage (constructive fallback list)

Slot alignment: slot centre t -> segment id = floor(t / 2).  Slots with no
segment JSON fall back to nearest segment within +-3 s, else zero row
(+coverage 0).  Scout readout (O1 gate): per-source keep-segment union vs
GT-positive slots overlap, written to scout_readout.json.

Usage (910A):
    python3 scripts/r11_o1_v1_features.py \
      --e3-dir   /data/aic/experiments_910a/<V4_E3_out_dir> \
      --manifest-dir /data/aic/experiments_910a/LFM_V11/r8_npu/r9_audit \
      --out /data/aic/experiments_910a/LFM_V11/r11_e3
CPU only.  Missing E3 dir -> SystemExit with the expected layout, so the
stage-1 chain can branch to re-inference (stage2 NPU arm).
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

ROLES = ["MAIN", "SETUP", "RESULT", "IDLE", "JUNK"]
KEEP = {"MAIN", "SETUP", "RESULT"}


def slot_features(roles_by_seg, t8):
    """(8,5) one-hot role block + coverage + keep prior for 8 slot centres."""
    feat = np.zeros((len(t8), len(ROLES)), dtype=np.float32)
    cov = np.zeros(len(t8), dtype=np.float32)
    keep = np.zeros(len(t8), dtype=np.float32)
    for i, ti in enumerate(t8):
        seg = int(ti // 2)
        r = roles_by_seg.get(seg)
        if r is None:  # nearest segment within +-3 s
            near = sorted(roles_by_seg, key=lambda s: min(abs(s * 2 - ti), abs(s * 2 + 2 - ti)))
            if near and abs(near[0] * 2 - ti) <= 3.0:
                seg, r = near[0], roles_by_seg[near[0]]
        if r in ROLES:
            feat[i, ROLES.index(r)] = 1.0
            cov[i] = 1.0
            keep[i] = 1.0 if r in KEEP else 0.0
    return feat, cov, keep


def scout_readout(man, roles_by_vid):
    """O1 gate numbers: keep-segment union vs GT-positive-slot overlap."""
    rows = []
    for r in man:
        vid = r["source_id"]
        rb = roles_by_vid.get(vid)
        if not rb:
            continue
        keep_segs = {s for s, role in rb.items() if role in KEEP}
        y = np.asarray(r.get("labels", r.get("y", [])), dtype=np.float32)
        if not keep_segs:
            rows.append({"vid": vid, "keep_seg": 0, "gt_pos": int((y > 0).sum()),
                         "overlap": 0.0, "n_slots": len(y)})
            continue
        # slot centre approximation: slot j of K spans j*T/K..(j+1)*T/K; use manifest t if present
        ts = r.get("t")
        ks = sum(1 for j, yj in enumerate(y) if yj > 0 and (ts is None or any(
            s * 2 <= ts[j] < s * 2 + 2 for s in keep_segs)))
        rows.append({"vid": vid, "keep_seg": len(keep_segs), "gt_pos": int((y > 0).sum()),
                     "overlap": round(ks / max((y > 0).sum(), 1), 4),
                     "n_slots": len(y)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--e3-dir", type=Path, required=True)
    ap.add_argument("--manifest-dir", type=Path,
                    default=Path("/data/aic/experiments_910a/LFM_V11/r8_npu/r9_audit"))
    ap.add_argument("--audio-root", type=Path,
                    default=Path("/data/aic/experiments_910a/LFM_V11/r10_audio"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if not args.e3_dir.exists():
        raise SystemExit(f"E3 role dir not found: {args.e3_dir} -- expected V4-E3 "
                         f"output (<vid>.json roles per 2 s segment). Run stage2 NPU "
                         f"re-inference arm first (scripts/r11_stage2_npu.sh V1).")
    man = []
    splits = [args.manifest_dir / f"{s}.jsonl" for s in ("train_public", "eval_public")]
    splits = [p for p in splits if p.exists()] or sorted(args.manifest_dir.glob("*.jsonl"))
    for p in splits:
        man += [json.loads(l) for l in open(p) if l.strip()]
    seen, man_dedup = set(), []
    for r in man:
        if "fragment_id" in r and r["fragment_id"] not in seen:
            seen.add(r["fragment_id"])
            man_dedup.append(r)
    man = [r for r in man_dedup if "source_id" in r]
    print(f"manifest frags={len(man)}")
    args.out.mkdir(parents=True, exist_ok=True)

    # slot time axis lives inside the R10 audio npz (t field); manifest rows
    # carry labels only.  Audio features also tell us which frags exist.
    audio_t = {}
    for f in Path(args.audio_root).glob("p*/*.npz"):
        try:
            audio_t[f.stem] = np.load(f)["t"].astype(np.float32)
        except (FileNotFoundError, KeyError):
            pass

    n_cov, missing = 0, []
    scout_rows, roles_by_vid = [], {}
    for r in man:
        p = args.e3_dir / f"{r['source_id']}.json"
        outp = args.out / f"{r['fragment_id']}.npz"
        if outp.exists():
            n_cov += 1
            continue
        if not p.exists() or r["fragment_id"] not in audio_t:
            missing.append(r["fragment_id"])
            continue
        d = json.loads(p.read_text())
        rb = {}
        for blk in d if isinstance(d, list) else [d]:
            for seg, blk_roles in (blk.get("roles") or {}).items():
                rb[int(seg)] = blk_roles
        roles_by_vid[r["source_id"]] = rb
        t8 = audio_t[r["fragment_id"]]
        if t8.shape != (8,):
            missing.append(r["fragment_id"])
            continue
        feat, cov, keep = slot_features(rb, t8)
        np.savez(outp, e3_role=feat, slot_cov=cov, keep_prior=keep, t=t8)
        n_cov += 1
    (args.out / "e3_missing.json").write_text(json.dumps(missing))
    cov_frac = n_cov / max(len(man), 1)
    scout_rows = scout_readout(
        [r for r in man if (args.out / f"{r['fragment_id']}.npz").exists()], roles_by_vid)
    if scout_rows:
        ov = float(np.mean([x["overlap"] for x in scout_rows]))
        summ = {"n_frags": len(scout_rows), "role_coverage_frac": round(cov_frac, 4),
                "mean_keep_gt_overlap": round(ov, 4),
                "status": "OK" if cov_frac >= 0.1 else "INSUFFICIENT_COVERAGE",
                "gate_o1_ge_06": ov >= 0.6,
                "note": "coverage <0.1 => roles need PHD2 re-inference (stage2 arm B); "
                        "overlap >=0.6 keeps O1-verifier queued"}
        (args.out / "scout_readout.json").write_text(json.dumps({"summary": summ, "rows": scout_rows[:50]}, indent=1))
        print(json.dumps(summ, indent=1))
    print(f"feature frags written={n_cov} missing_e3={len(missing)}")


if __name__ == "__main__":
    main()
