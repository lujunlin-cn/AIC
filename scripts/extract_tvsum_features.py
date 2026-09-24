#!/usr/bin/env python3
"""Extract pinned headless ResNet18 features for the TVSum proxy split."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from aic.features import extract_manifest, save_backbone_state
from aic.models import load_imagenet_backbone


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--backbone-state", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    os.environ.setdefault("TORCH_HOME", str(Path(args.backbone_state).parent))
    import torch

    backbone, provenance = load_imagenet_backbone()
    audit = save_backbone_state(backbone, args.backbone_state, provenance)
    all_results = []
    manifest_dir = Path(args.manifest_dir)
    for split in ("train", "val"):
        manifest = manifest_dir / f"{split}.jsonl"
        if not manifest.exists():
            continue
        # A temporary bounded manifest makes --limit deterministic and leaves
        # the source provenance manifest untouched.
        records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        if args.limit:
            records = records[:args.limit]
        temporary = manifest_dir / f".{split}.extract.jsonl"
        temporary.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in records) + "\n", encoding="utf-8")
        try:
            results = extract_manifest(temporary, args.cache_dir, backbone, args.device,
                                       args.sample_fps, args.batch_size, skip_existing=True)
        finally:
            temporary.unlink(missing_ok=True)
        by_id = {x["video_id"]: x for x in records}
        output_manifest = []
        for result in results:
            item = dict(by_id[result["video_id"]])
            item.update({"path": result["path"], "feature_dim": result.get("feature_dim"),
                         "cache_frames": result.get("frames")})
            output_manifest.append(item)
        out = manifest_dir / f"{split}_features.jsonl"
        out.write_text("\n".join(json.dumps(x, ensure_ascii=False, sort_keys=True)
                                  for x in output_manifest) + "\n", encoding="utf-8")
        all_results.extend(results)
    print(json.dumps({"backbone": audit, "cache_count": len(all_results)},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
