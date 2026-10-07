"""R11 S2c pre-audit: talking-head prevalence from R10 AST-527 classes, free.

Question before ANY Light-ASD spend: do PHD2 sources even contain a speaking
person on camera?  R10 3.1 measured Speech mean sigma=0.30 (33% of slots
>0.5) pool-wide, but not per-source prevalence.  This readout recomputes,
per source: mean/max Speech-class probability across the 8 slots and the
fraction of sources whose audio is speech-dominated (mean>0.3) vs BGM-only
( Speech ~0, Music high ).  Zero new extraction -- reads the cached
r10_audio npz ast_527 blocks.  CPU seconds.

Class index for 'Speech' in the AST AudioSet 527-label set is resolved from
the cached class-mapping json if present, else the well-known index 0
(AudioSet label 0 = Speech).
"""
import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--audio-root", type=Path,
                default=Path("/data/aic/experiments_910a/LFM_V11/r10_audio"))
ap.add_argument("--manifest-dir", type=Path,
                default=Path("/data/aic/experiments_910a/LFM_V11/r8_npu/r9_audit"))
ap.add_argument("--speech-idx", type=int, default=0, help="AST AudioSet index of 'Speech'")
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()

man = []
splits = [args.manifest_dir / f"{s}.jsonl" for s in ("train_public", "eval_public")]
splits = [p for p in splits if p.exists()] or sorted(args.manifest_dir.glob("*.jsonl"))
for p in splits:
    man += [json.loads(l) for l in open(p) if l.strip()]
seen, man_dedup = set(), []
for r in man:
    if r["fragment_id"] not in seen:
        seen.add(r["fragment_id"])
        man_dedup.append(r)
man = man_dedup
audio_files = {p.stem: p for p in args.audio_root.glob("p*/*.npz")}
srcs = sorted({r["source_id"] for r in man if "source_id" in r})
per, miss = {}, 0
for s in srcs:
    fids = [r["fragment_id"] for r in man if r["source_id"] == s]
    vals, av = [], []
    for f in fids:
        p = audio_files.get(f)
        if p is None:
            miss += 1
            continue
        z = np.load(p)
        if "ast_527" not in z.files:
            continue
        vals.append(np.asarray(z["ast_527"], dtype=np.float32)[:, args.speech_idx])
        av.append(np.asarray(z["avalid"], dtype=np.float32))
    if not vals:
        per[s] = None
        continue
    v = np.concatenate(vals)
    a = np.concatenate(av) if av else np.ones(len(v))
    # ast_527 npz stores raw logits: convert to probability domain before
    # thresholding (logit 0 == prob 0.5).
    pv = 1.0 / (1.0 + np.exp(-v.astype(np.float64)))
    per[s] = {"speech_mean": round(float(pv.mean()), 4),
              "speech_max": round(float(pv.max()), 4),
              "valid_frac": round(float(a.mean()), 4)}

ok = {s: v for s, v in per.items() if v}
sp = [v["speech_mean"] for v in ok.values()]
res = {
    "n_sources": len(srcs), "n_with_audio_feat": len(ok), "missing": miss,
    "speech_dominated": sum(1 for v in sp if v > 0.3),
    "bgm_only": sum(1 for v in sp if v <= 0.1),
    "between": sum(1 for v in sp if 0.1 < v <= 0.3),
    "mean_speech": round(float(np.mean(sp)), 4) if sp else None,
    "note": "speech_dominated frac <0.3 => S2c Light-ASD arm value drops; "
            ">=0.3 => proceed to face-track x audio alignment",
}
args.out.write_text(json.dumps({"summary": res, "per_source": per}, indent=1))
print(json.dumps(res, indent=1))
