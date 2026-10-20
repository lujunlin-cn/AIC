#!/usr/bin/env python3
"""Real text, public-video and multi-crop tests of pinned Valen weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aic.valen_adapter import ValenDecisionAdapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from PIL import Image
    import torch
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter(); adapter = ValenDecisionAdapter(args.model)
    load_s = time.perf_counter() - t
    image = Image.open(args.image).convert("RGB"); W, H = image.size
    content = [{"type": "text", "text": "Four crop options from the same source image follow. Compare composition."}]
    for k in range(4):
        p = out / f"crop_{k}.png"
        w = round(W * .6); x = round((W-w)*k/3)
        image.crop((x, 0, x+w, H)).save(p)
        content += [{"type": "text", "text": f"Candidate {chr(65+k)}:"},
                    {"type": "image_url", "image_url": {"url": str(p.resolve())}}]
    requests = [
        {"state": "A cat is on the sofa.", "questions": {"animal": {"type": "choice", "instructions": "Which animal is present?", "criteria": {"cat": "A cat", "dog": "A dog"}}}},
        {"state": {"messages": [{"role": "user", "content": content}]},
         "questions": {"crop": {"type": "choice", "instructions": "Choose the crop that best preserves the main subject and action with balanced composition.", "criteria": {chr(65+k): f"Candidate {chr(65+k)} crop image" for k in range(4)}}}},
        {"state": {"messages": [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": str(Path(args.video).resolve())}}]}]},
         "questions": {"motion": {"type": "choice", "instructions": "What is visible in this video?", "criteria": {"motion": "An action or movement is visible", "still": "The scene remains still"}}}},
    ]
    results = []
    for i, r in enumerate(requests):
        z, timing = adapter.decide_batch([r])
        results.append({"kind": ["text", "multi_image", "video"][i], "raw": z[0]["raw"], "media": z[0]["media"], "timing": timing})
        print(json.dumps(results[-1], default=str), flush=True)
    # The official minibatch path must agree with individual model.predict.
    z, timing = adapter.decide_batch(requests[:2])
    for i in range(2):
        for qid, a in z[i]["raw"]["answers"].items():
            for key, value in a["probabilities"].items():
                assert abs(value-results[i]["raw"]["answers"][qid]["probabilities"][key]) < 2e-3
    report = {"status": "PASS", "model": args.model, "parameters": adapter.parameters,
              "load_s": load_s, "results": results, "batch_timing": timing,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "torch": torch.__version__}
    (out / "smoke.json").write_text(json.dumps(report, indent=2, default=str)+"\n")


if __name__ == "__main__":
    main()
