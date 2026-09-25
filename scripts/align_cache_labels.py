#!/usr/bin/env python3
"""Copy labels/masks from a reference cache onto an alternate encoder cache.

Used to keep the canonical internal-TSM probe on exactly the same proxy label
version as its A0 control while retaining the alternate feature arrays.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from aic.features import load_feature_cache, save_feature_cache

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument("--target-manifest",required=True); ap.add_argument("--reference-dir",required=True); a=ap.parse_args(argv)
    rows=[json.loads(x) for x in Path(a.target_manifest).read_text().splitlines() if x.strip()]
    changed=0
    for row in rows:
        tp=Path(row["path"]); rp=Path(a.reference_dir)/tp.name
        target=load_feature_cache(tp); ref=load_feature_cache(rp)
        if not np.array_equal(target["frame_indices"],ref["frame_indices"]):
            raise ValueError(f"frame alignment mismatch: {tp} vs {rp}")
        metadata=dict(target["metadata"]); metadata["label_source_copied_from"]=str(rp)
        save_feature_cache(tp,target["features"],target["frame_indices"],target["timestamps"],ref["labels"],ref["mask"],metadata,target["aux"])
        changed += 1
    print(json.dumps({"updated":changed,"target_manifest":a.target_manifest,"reference_dir":a.reference_dir}))
if __name__=="__main__": raise SystemExit(main())
