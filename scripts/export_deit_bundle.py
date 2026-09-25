#!/usr/bin/env python3
"""Export a complete FP16 DeiT-S/16 + Temporal U-Net inference bundle."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vit", required=True)
    ap.add_argument("--temporal", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    vit = torch.load(args.vit, map_location="cpu", weights_only=True)
    if "state_dict" in vit:
        vit = vit["state_dict"]
    temporal = torch.load(args.temporal, map_location="cpu", weights_only=True)
    temporal = temporal["model"] if "model" in temporal else temporal
    cast = lambda state: {k: v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu()
                          for k, v in state.items()}
    bundle = {"format_version": 1, "architecture": "Bs0_deit_s_temporal",
              "persistent_dtype": "fp16", "backbone_name": "deit_small_patch16_224",
              "backbone_state": cast(vit), "temporal_state": cast(temporal),
              "parameter_count": sum(v.numel() for v in vit.values()) + sum(v.numel() for v in temporal.values()),
              "source_temporal_checkpoint": str(args.temporal),
              "submission_candidate": True,
              "size_note": "FP16 DeiT-S/16 + TemporalUNet; measure bytes before competition use"}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out)
    print(json.dumps({"path": str(out), "bytes": out.stat().st_size,
                      "MB_decimal": out.stat().st_size / 1e6,
                      "parameter_count": bundle["parameter_count"]}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
