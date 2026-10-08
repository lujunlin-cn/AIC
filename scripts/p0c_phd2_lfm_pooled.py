#!/usr/bin/env python3
"""P0-c: PHD2-domain LFM2VL vision-tower pooled features for R12-GATT.

Faithful port of v8_live_extract.py's vision() contract (the B3 representation):
frames resized long-side 640, min_tiles=max_tiles=1, fp16 eager NPU forward,
pooled = grid mean over valid tokens -> (768,) fp16 per frame.

Input : /data/aic/experiments_910a/PHD2_FRAG_V1/frames/<frag>/<t>.jpg  (8 frames/frag)
Output: /data/aic/experiments_910a/LFM_V11/r12_p0c/phd2_lfm_pooled/<frag>.npz
        {ts [8] f8, pooled [8,768] f16}
Resume: frags with existing npz are skipped. 32 shards, incremental flush.
"""
import argparse
import json
import os
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--frames-root", type=Path,
                default=Path("/data/aic/experiments_910a/PHD2_FRAG_V1/frames"))
ap.add_argument("--model", default="/data/aic/pretrained/lfm2_5_vl_450m")
ap.add_argument("--out", type=Path,
                default=Path("/data/aic/experiments_910a/LFM_V11/r12_p0c/phd2_lfm_pooled"))
ap.add_argument("--cards", default="0")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--limit", type=int, default=0, help="debug: cap frag count")
args = ap.parse_args()

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = args.cards

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False
assert torch.npu.device_count() > 0, "no NPU"
from PIL import Image  # noqa: E402
from transformers import AutoProcessor, Lfm2VlForConditionalGeneration  # noqa: E402

sys_p = Path("/root/AIC/scripts")
import sys  # noqa: E402
sys.path.insert(0, str(sys_p))
from v11_npu_siglip2_patch import patch_siglip2_npu  # noqa: E402

proc = AutoProcessor.from_pretrained(args.model, min_tiles=1, max_tiles=1, max_image_tokens=256)
model = Lfm2VlForConditionalGeneration.from_pretrained(
    args.model, dtype=torch.float16, low_cpu_mem_usage=True,
    attn_implementation="eager").eval().to("npu")
assert patch_siglip2_npu(model), "siglip2 npu patch failed"
print(f"[p0c] model ready shard {args.shard}/{args.nshards}", flush=True)


def vision(im: Image.Image):
    """Verbatim port of v8_live_extract.vision() (lines 57-73): pooled (768,) f16."""
    msgs = [{"role": "user", "content": [{"type": "image", "image": im},
                                         {"type": "text", "text": "describe"}]}]
    x = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                 return_dict=True, return_tensors="pt")
    x = {k: v.to("npu") for k, v in x.items() if isinstance(v, torch.Tensor)}
    x["pixel_values"] = x["pixel_values"].to(torch.float16)
    with torch.no_grad():
        out = model.model.vision_tower(pixel_values=x["pixel_values"], spatial_shapes=x["spatial_shapes"],
                                       pixel_attention_mask=x["pixel_attention_mask"], return_dict=True)
        valid = int(x["pixel_attention_mask"][0].sum())
        fh, fw = [int(v) for v in x["spatial_shapes"][0]]
        grid = out.last_hidden_state[0, :valid].reshape(fh, fw, -1)[:, : (fw // 2) * 2, :]
        grid = grid[: (fh // 2) * 2]
        pooled = grid.float().mean((0, 1))
    return pooled.cpu().numpy().astype(np.float16)


def main():
    args.out.mkdir(parents=True, exist_ok=True)
    frags = sorted(d.name for d in args.frames_root.iterdir() if d.is_dir())
    mine = [f for i, f in enumerate(frags) if i % args.nshards == args.shard]
    if args.limit:
        mine = mine[: args.limit]
    todo = [f for f in mine if not (args.out / f"{f}.npz").exists()]
    print(f"[p0c shard {args.shard}] mine={len(mine)} todo={len(todo)}", flush=True)

    n_err = 0
    for k, frag in enumerate(todo):
        fdir = args.frames_root / frag
        try:
            ts, pools = [], []
            for fp in sorted(fdir.glob("*.jpg"), key=lambda p: float(p.stem)):
                im = Image.open(fp).convert("RGB")
                if max(im.size) > 640:
                    im = im.resize((round(im.width * 640 / max(im.size)),
                                    round(im.height * 640 / max(im.size))))
                ts.append(float(fp.stem))
                pools.append(vision(im))
            po = np.stack(pools) if pools else np.zeros((0, 768), np.float16)
            np.savez(args.out / f"{frag}.npz", ts=np.array(ts, np.float64), pooled=po)
        except Exception as e:  # noqa: BLE001
            n_err += 1
            (args.out / f"{frag}.err").write_text(str(e)[:300])
        if (k + 1) % 50 == 0:
            print(f"[p0c shard {args.shard}] pairs_total={k + 1}/{len(todo)} err={n_err}", flush=True)
    print(f"[p0c shard {args.shard}] DONE total={len(todo)} err={n_err}", flush=True)


if __name__ == "__main__":
    main()
