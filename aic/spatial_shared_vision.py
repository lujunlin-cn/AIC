"""The verified P2 Qwen3.5 visual tower, shared at deployment (zero new encoder).

This is intentionally not the historical R70 LFM feature contract. The new
contract is frozen Qwen pre-merger spatial tokens, fractional crop pooling,
and 12 deterministic geometry features. No random projection/backbone.
"""
from __future__ import annotations

import io
import json
import math
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

from aic.scale_policy import canonical_sha, sha

GEOMETRY_DIM = 12
FEATURE_VERSION = "P2_QWEN35_PREMERGER_ROI_V1"


def unmerge_grid(tokens, height, width, merge=2):
    """Undo Qwen's merge-group-major 2x2 token ordering, not plain reshape."""
    tokens = np.asarray(tokens)
    if height % merge or width % merge or len(tokens) != height * width:
        raise ValueError("processor/token spatial grid mismatch")
    return tokens.reshape(height//merge, width//merge, merge, merge, -1).transpose(0, 2, 1, 3, 4).reshape(height, width, -1)


def pool_candidates(grid, boxes_xywh, width, height, subject_mask=None):
    """Fractional-area ROI pooling in original-image coordinates, no GT input."""
    grid = np.asarray(grid, np.float32); boxes = np.asarray(boxes_xywh, np.float32)
    if grid.ndim != 3 or boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(grid).all():
        raise ValueError("invalid real token grid/candidate boxes")
    if (boxes[:, :2] < 0).any() or (boxes[:, 2:] <= 0).any() or ((boxes[:, :2]+boxes[:, 2:]) > [width+1e-6, height+1e-6]).any():
        raise ValueError("candidate is out of bounds; no silent pooling clamp")
    gh, gw, dim = grid.shape
    x0 = np.arange(gw) * width/gw; y0 = np.arange(gh) * height/gh
    x1 = x0 + width/gw; y1 = y0 + height/gh
    mx = np.maximum(0, np.minimum(boxes[:, 0:1]+boxes[:, 2:3], x1)-np.maximum(boxes[:, 0:1], x0))/(width/gw)
    my = np.maximum(0, np.minimum(boxes[:, 1:2]+boxes[:, 3:4], y1)-np.maximum(boxes[:, 1:2], y0))/(height/gh)
    weights = (my[:, :, None] * mx[:, None, :]).reshape(len(boxes), -1)
    flat = grid.reshape(-1, dim); inside = weights @ flat / np.maximum(weights.sum(1, keepdims=True), 1e-8)
    outside = (1-weights) @ flat / np.maximum((1-weights).sum(1, keepdims=True), 1e-8)
    norm = boxes / [width, height, width, height]
    cx, cy = norm[:, 0]+norm[:, 2]/2, norm[:, 1]+norm[:, 3]/2
    subject = np.zeros((len(boxes), 3), np.float32)
    if subject_mask is not None:
        mask = np.asarray(subject_mask, float)
        if mask.shape != (gh, gw) or not np.isfinite(mask).all() or (mask < 0).any() or (mask > 1).any():
            raise ValueError("subject mask must match visual grid and lie in [0,1]")
        overlap = weights @ mask.reshape(-1)
        subject[:, 0] = overlap / max(float(mask.sum()), 1e-8)
        subject[:, 1] = overlap / np.maximum(weights.sum(1), 1e-8)
        subject[:, 2] = 1  # presence indicator; no mask is distinct from an empty subject
    geom = np.column_stack((norm, cx, cy, norm[:, 2]*norm[:, 3], np.log(boxes[:, 2]/boxes[:, 3]), np.ones(len(boxes)), subject))
    assert geom.shape[1] == GEOMETRY_DIM
    return np.concatenate((inside, outside, inside-outside, geom), axis=1).astype(np.float32)


def load_image(row, path_aliases=None):
    def resolve(name):
        for old, new in sorted((path_aliases or {}).items(), key=lambda kv: -len(kv[0])):
            if name == old or name.startswith(old.rstrip("/") + "/"):
                return Path(new + name[len(old):])
        return Path(name)
    if row.get("input_archive"):
        with ZipFile(resolve(row["input_archive"])) as archive:
            payload = archive.read(row["input_image"])
    else:
        payload = resolve(row["input_image"]).read_bytes()
    import hashlib
    if hashlib.sha256(payload).hexdigest() != row["image_sha256"]:
        raise ValueError("public image bytes changed")
    with Image.open(io.BytesIO(payload)) as source:
        image = source.convert("RGB")
    if row.get("transposed_geometry_augmentation"):
        image = image.transpose(Image.Transpose.TRANSPOSE)
    if list(image.size) != row.get("media_size", [row["image_width"], row["image_height"]]):
        raise ValueError("public media/view size mismatch")
    # Same resize as P2 spatial observations. Original pixel geometry is retained
    # separately, so low-resolution cached images do not become 640px GT.
    w, h = image.size
    return image.resize((max(32, round(w * 640/max(w, h)/32)*32), max(32, round(h * 640/max(w, h)/32)*32)), Image.Resampling.BICUBIC)


def load_subject_mask(row, grid_shape, path_aliases=None):
    name=row["subject_mask"]
    for old,new in sorted((path_aliases or {}).items(),key=lambda kv:-len(kv[0])):
        if name.startswith(old.rstrip("/")+"/"):
            name=new+name[len(old):];break
    if sha(name)!=row["subject_mask_sha256"]:
        raise ValueError("SACD subject mask identity changed")
    with Image.open(name) as source:
        mask=np.asarray(source.convert("L"),float)
    if mask.shape!=(row["image_height"],row["image_width"]):
        raise ValueError("subject mask is not in original image coordinates")
    # Released masks may encode foreground as 1 or 255. Binarization is a mask
    # contract, not a learned subject detector or synthetic human annotation.
    mask_image=Image.fromarray(((mask>0)*255).astype(np.uint8))
    return np.asarray(mask_image.resize((grid_shape[1],grid_shape[0]),Image.Resampling.BOX),float)/255


class SharedP2Vision:
    def __init__(self, model_dir, identity_path, device="cuda:0"):
        import torch
        from safetensors import safe_open
        from transformers import AutoConfig, AutoProcessor
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

        self.device = device; model_dir = Path(model_dir)
        identity = json.loads(Path(identity_path).read_text())["model_evidence"]
        for name in ("config.json", "processor_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
            if sha(model_dir/name) != identity["base_files"][name]["sha256"]:
                raise ValueError("P2 processor/base identity changed: " + name)
        cfg = AutoConfig.from_pretrained(model_dir, local_files_only=True).vision_config
        cfg._attn_implementation = "eager"
        # No full language-model allocation. These exact tensors already count
        # in P2's 8,999,771,458 elements; a separate encoder may not be deployed.
        with torch.device(device):
            self.vision = Qwen3_5VisionModel(cfg).to(dtype=torch.bfloat16)
        index = json.loads((model_dir/"model.safetensors.index.json").read_text())["weight_map"]
        selected = {k: v for k, v in index.items() if k.startswith("model.visual.")}
        if not selected:
            raise ValueError("verified compact base has no shared visual tower")
        for shard in set(selected.values()):
            if sha(model_dir/shard) != identity["base_files"][shard]["sha256"]:
                raise ValueError("P2 visual shard hash mismatch")
        state = {}
        for shard in sorted(set(selected.values())):
            with safe_open(model_dir/shard, framework="pt", device="cpu") as reader:
                for name in selected:
                    if selected[name] == shard:
                        state[name.removeprefix("model.visual.")] = reader.get_tensor(name).to(device=device, dtype=torch.bfloat16)
        self.vision.load_state_dict(state, strict=True)
        del state
        self.vision.requires_grad_(False); self.vision.eval()
        self.processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True).image_processor
        self.merge = cfg.spatial_merge_size
        self.identity = {"feature_version": FEATURE_VERSION, "base_identity_sha256": canonical_sha(identity),
                         "visual_shards": {s: sha(model_dir/s) for s in sorted(set(selected.values()))},
                         "processor_sha256": identity["base_files"]["processor_config.json"]["sha256"],
                         "visual_dim": cfg.hidden_size, "geometry_dim": GEOMETRY_DIM,
                         "added_encoder_parameters": 0, "frozen": True,
                         "preprocessing": "RGB,PIL_BICUBIC_side640_alignment32,processor_do_resize_false",
                         "source_sha256": sha(__file__)}
        self.identity_sha256 = canonical_sha(self.identity)

    def encode(self, images):
        import torch
        data = self.processor(images=images, do_resize=False, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.vision(data["pixel_values"].to(torch.bfloat16), grid_thw=data["image_grid_thw"])
        if self.vision.training or any(p.requires_grad for p in self.vision.parameters()):
            raise ValueError("shared P2 encoder must remain frozen/eval")
        tokens = out.last_hidden_state.float().cpu().numpy(); result = []; start = 0
        for t, h, w in data["image_grid_thw"].tolist():
            if t != 1:
                raise ValueError("still-image token contract required")
            result.append(unmerge_grid(tokens[start:start+h*w], h, w, self.merge)); start += h*w
        if start != len(tokens):
            raise ValueError("vision batch token coverage mismatch")
        return result
