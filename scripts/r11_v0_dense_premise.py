"""R11 V0: dense-view premise probe (R8 2.5 KD prerequisite).

Question (never measured directly): does a head fed DENSER temporal
observation beat the same head on the sparse 8-anchor view, on true
labelled IoU?  If NO, the dense-view KD arm (6 NPU-h) never starts.

Probe design (kept small on purpose):
  arm S (sparse): existing V8/V11 slot-pool samples (8 anchors, cached)
  arm D (dense) : the same GT/candidates, observation built from the
                   frozen native contract (R5 sec 3, 16 native frames per
                   1 s action unit) instead of 8 anchors
  same head family (B3 MLP+attn), same train/dev split, 3 seeds;
  judged on live_dev/rv_dev head-argmax IoU, paired per source, cluster CI.

fail-fast: candidate cache roots are probed; if no dense-observation cache
exists, the script exits with NOT_RUN and the exact paths it looked at --
the stage-1 chain records that as V0=NOT_RUN (premise UNTESTED, KD arm
stays gated until the cache is built).
"""
import argparse
import json
import sys
from pathlib import Path

CAND_DENSE_ROOTS = [
    "/data/aic/experiments_910a/LFM_V11/native_frames",
    "/data/aic/experiments_910a/LFM_V11/r5_native",
    "/data/aic/experiments_910a/LFM_V8/native16_feats",
]
CAND_SPARSE_ROOTS = [
    "/data/aic/experiments_910a/LFM_V8/samples",
    "/data/aic/experiments_910a/LFM_V11/samples",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sparse-root", type=Path, default=None)
    ap.add_argument("--dense-root", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sparse = args.sparse_root or next((Path(p) for p in CAND_SPARSE_ROOTS if Path(p).exists()), None)
    dense = args.dense_root or next((Path(p) for p in CAND_DENSE_ROOTS if Path(p).exists()), None)
    if sparse is None or dense is None:
        res = {"status": "NOT_RUN",
               "reason": "dense-observation cache not found; V0 premise stays UNTESTED",
               "probed_dense": CAND_DENSE_ROOTS,
               "probed_sparse": [str(sparse)]}
        args.out.write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
        return
    # Dense cache located: the full two-head comparison needs the native
    # contract builder (R5 sec 3); that chain runs on the 910A host where
    # the cache lives -- left to the stage-2 NPU arm with this JSON as its
    # prereg record.  Here we only verify the sparse side loads.
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import numpy as np
        z = np.load(Path(sparse) / "live_dev_u.npy")
        res = {"status": "CACHE_FOUND_RUN_ON_910A",
               "sparse_root": str(sparse), "dense_root": str(dense),
               "sparse_rows": int(z.shape[0]),
               "protocol": "B3 head x2 (8-anchor vs 16-native-frame obs), 3 seeds, "
                           "paired per-source head-argmax IoU, cluster CI; gate: dense-sparse >0"}
    except Exception as e:  # noqa: BLE001
        res = {"status": "NOT_RUN", "error": str(e)}
    args.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
