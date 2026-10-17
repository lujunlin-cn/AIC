#!/usr/bin/env python3
"""Run the already audited InternVideo2 R43 scorer on the 12 labelled clips.

This produces only public-source diagnostic scores.  It never reads the AIC
test set or a test label.  The output is consumed by the R44 residual probe.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch


def load(path: Path):
    spec = importlib.util.spec_from_file_location("r43_iv2", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r43-script", type=Path, required=True)
    ap.add_argument("--model-root", type=Path, required=True)
    ap.add_argument("--video-root", type=Path, required=True)
    ap.add_argument("--video-ids", type=Path, required=True,
                    help="newline separated public video IDs")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    mod = load(args.r43_script)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model, _module, params = mod._load_model(args.model_root, device, dtype)
    prompts = mod._text_features(model, mod.ALL_PROMPTS, device)
    out = {"model_params": params, "videos": {}}
    ids = [x.strip() for x in args.video_ids.read_text().splitlines() if x.strip()]
    for pos, vid in enumerate(ids):
        path = args.video_root / f"{vid}.mp4"
        score, diag = mod._score_video(model, prompts, path, device, dtype, anchor_batch_size=4)
        out["videos"][vid] = {"frame_score": score.tolist(), **diag}
        print(json.dumps({"completed": pos + 1, "total": len(ids), "video_id": vid}, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "videos": len(ids), "model_params": params}))


if __name__ == "__main__":
    main()
