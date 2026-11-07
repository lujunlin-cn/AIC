#!/usr/bin/env python3
"""Read 32 bytes from published weight endpoints; no full download or model load.

Run only on a network-enabled research executor. Output is immutable by default.
This is an access probe, not an inference, checksum or training reproduction.
"""
import argparse
import hashlib
import json
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ENDPOINTS = {
    "mobilenet_v3_small": "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
    "mobilenet_v3_large_v2": "https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth",
    "deit_small": "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
    "dinov2_vits14_author": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
    "dinov2_small_hf": "https://huggingface.co/facebook/dinov2-small/resolve/ed25f3a31f01632728cabb09d1542f84ab7b0056/model.safetensors",
    "videomae_finetuned_hf": "https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics/resolve/488eb9a0565f257b32866000305c8178965eb9f6/model.safetensors",
    "clip_base_vitb32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
}


def probe(name, url):
    record = {"name": name, "url": url, "weight_tier": "W1", "full_hash_verified": False, "model_loaded": False}
    request = urllib.request.Request(url, headers={"User-Agent": "AIC-research-readonly/1.0", "Range": "bytes=0-31"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            data = response.read(32)  # bounded even if server ignores Range
            record.update(status=response.status, content_type=response.headers.get("Content-Type"),
                          content_range=response.headers.get("Content-Range"), bytes_read=len(data),
                          prefix_hex=data.hex(), prefix_sha256=hashlib.sha256(data).hexdigest())
            zip_header = data.startswith(b"PK\x03\x04")
            safe_header = len(data) >= 9 and 1 < int.from_bytes(data[:8], "little") < 1024 * 1024 and data[8:9] == b"{"
            if response.status in (200, 206) and (zip_header or safe_header):
                record["weight_tier"] = "W2"
    except Exception as exc:
        record["error"] = str(exc)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = {"date_shanghai": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "scope": "weight endpoints only; base CLIP does not establish HL-CLIP finetuned weight availability",
              "records": [probe(name, url) for name, url in ENDPOINTS.items()]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"records": len(result["records"]), "tiers": {r["name"]: r["weight_tier"] for r in result["records"]}}, indent=2))


if __name__ == "__main__":
    main()
