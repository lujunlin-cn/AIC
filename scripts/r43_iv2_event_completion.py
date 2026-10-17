#!/usr/bin/env python3
"""R43 InternVideo2 Stage2-6B event-completion inference.

The script deliberately keeps the spatial contract from the R26 B3 package.  It
uses the public InternVideo2 video/text embedding interface, scores one four
frame window per second, interpolates that score to original decoded frames,
and selects exactly round(.85*N) frames per video.  It is an inference-only
diagnostic/candidate generator; no test labels or pseudo labels are read.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import torch


EVENT_POSITIVE = [
    "a decisive highlight moment in a video, showing the key action or result",
    "the crucial moment of an event, where the important action is happening",
    "an exciting memorable moment worth keeping in a short video",
    "a successful action, reveal, demonstration, or strong reaction",
]
ROUTINE_NEGATIVE = [
    "a routine transition or ordinary continuation with no important event",
    "a static setup, waiting moment, or uninteresting routine activity",
    "a blurry or incidental frame that is not worth keeping",
]
PHASE_PREPARATION = ["an event is being prepared and the key result has not happened yet"]
PHASE_DECISIVE = ["the key action or decisive change is happening now"]
PHASE_RESULT = ["the result, reveal, or immediate reaction has just occurred"]
ALL_PROMPTS = EVENT_POSITIVE + ROUTINE_NEGATIVE + PHASE_PREPARATION + PHASE_DECISIVE + PHASE_RESULT


def _install_optional_stubs() -> None:
    """Provide only the optional imports needed by the upstream source.

    R43 disables flash attention/fused kernels.  The source nevertheless imports
    their symbols eagerly, and timm/einops are not part of the competition
    environment.  These tiny compatible definitions keep the audited standard
    PyTorch path explicit instead of silently installing a different kernel.
    """
    def module(name: str):
        m = types.ModuleType(name)
        m.__spec__ = importlib.util.spec_from_loader(name, loader=None)
        return m

    timm = module("timm")
    timm_models = module("timm.models")
    timm_layers = module("timm.models.layers")

    class DropPath(torch.nn.Module):
        def __init__(self, drop_prob: float = 0.0):
            super().__init__()
            self.drop_prob = float(drop_prob)

        def forward(self, x):
            if not self.training or self.drop_prob == 0.0:
                return x
            keep = 1.0 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            mask = x.new_empty(shape).bernoulli_(keep).div_(keep)
            return x * mask

    def to_2tuple(x):
        return (x, x) if not isinstance(x, (tuple, list)) else tuple(x)

    timm_layers.DropPath = DropPath
    timm_layers.to_2tuple = to_2tuple
    timm_layers.trunc_normal_ = torch.nn.init.trunc_normal_
    timm_models.layers = timm_layers
    timm.models = timm_models
    sys.modules.update({"timm": timm, "timm.models": timm_models, "timm.models.layers": timm_layers})

    einops = module("einops")

    def rearrange(x, pattern, **kwargs):
        # The standard (non-flash) attention path uses these two expressions.
        if pattern == "b s (three h d) -> b s three h d":
            three, heads = int(kwargs["three"]), int(kwargs["h"])
            return x.reshape(x.shape[0], x.shape[1], three, heads, x.shape[2] // (three * heads))
        if pattern == "b s h d -> b s (h d)":
            return x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        raise RuntimeError(f"unexpected einops expression on R43 path: {pattern}")

    einops.rearrange = rearrange
    sys.modules["einops"] = einops

    flash = module("flash_attn")
    flash_if = module("flash_attn.flash_attn_interface")
    flash_if.flash_attn_varlen_qkvpacked_func = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("R43 flash attention path is disabled")
    )
    flash_pad = module("flash_attn.bert_padding")
    flash_pad.unpad_input = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("flash path disabled"))
    flash_pad.pad_input = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("flash path disabled"))
    flash_mlp = module("flash_attn.modules.mlp")
    flash_mlp.FusedMLP = None
    flash_rms = module("flash_attn.ops.rms_norm")
    flash_rms.DropoutAddRMSNorm = None
    sys.modules.update({
        "flash_attn": flash,
        "flash_attn.flash_attn_interface": flash_if,
        "flash_attn.bert_padding": flash_pad,
        "flash_attn.modules": module("flash_attn.modules"),
        "flash_attn.modules.mlp": flash_mlp,
        "flash_attn.ops": module("flash_attn.ops"),
        "flash_attn.ops.rms_norm": flash_rms,
    })


def _load_upstream_module(root: Path):
    _install_optional_stubs()
    # InternVideo2's vendored BERT source targets transformers 4.x; the cloned
    # environment is newer and moved these two helpers to pytorch_utils.
    import transformers.modeling_utils as _mu
    import transformers.pytorch_utils as _pu
    import transformers.tokenization_utils as _tu
    import unicodedata
    for _name in ("apply_chunking_to_forward", "find_pruneable_heads_and_indices", "prune_linear_layer"):
        if not hasattr(_mu, _name) and hasattr(_pu, _name):
            setattr(_mu, _name, getattr(_pu, _name))
    if not hasattr(_mu, "find_pruneable_heads_and_indices"):
        def _find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            heads = set(heads) - set(already_pruned_heads)
            mask = torch.ones(n_heads, head_size, dtype=torch.bool)
            for head in heads:
                head -= sum(1 for old in already_pruned_heads if old < head)
                mask[head] = False
            index = torch.arange(mask.numel(), dtype=torch.long)[mask.view(-1)]
            return heads, index
        _mu.find_pruneable_heads_and_indices = _find_pruneable_heads_and_indices
    if not hasattr(_tu, "_is_control"):
        _tu._is_control = lambda ch: unicodedata.category(ch) in ("Cc", "Cf") and ch not in ("\t", "\n", "\r")
    if not hasattr(_tu, "_is_whitespace"):
        _tu._is_whitespace = lambda ch: ch in (" ", "\t", "\n", "\r") or unicodedata.category(ch) == "Zs"
    if not hasattr(_tu, "_is_punctuation"):
        _tu._is_punctuation = lambda ch: (33 <= ord(ch) <= 47) or (58 <= ord(ch) <= 64) or (91 <= ord(ch) <= 96) or (123 <= ord(ch) <= 126) or unicodedata.category(ch).startswith("P")
    name = "aic_r43_modeling_internvideo2"
    spec = importlib.util.spec_from_file_location(name, root / "modeling_internvideo2.py")
    if spec is None or spec.loader is None:
        raise FileNotFoundError(root / "modeling_internvideo2.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    # The source's internal AutoModel mapping names this module explicitly.
    sys.modules["modeling_internvideo2"] = module
    spec.loader.exec_module(module)

    vocab = root / "vocab.txt"
    if not vocab.exists():
        raise FileNotFoundError(f"BERT vocabulary missing: {vocab}")

    def tokenizer_local(cls, *_args, **_kwargs):
        return cls(vocab_file=str(vocab), do_lower_case=True)

    module.BertTokenizer.from_pretrained = classmethod(tokenizer_local)

    # Removed from transformers 5's base class but still called by the vendored
    # BERT forward.  R43 never supplies a head mask, so the faithful no-mask
    # branch is sufficient and avoids introducing a new attention policy.
    if not hasattr(module.BertModel, "get_head_mask"):
        module.BertModel.get_head_mask = lambda self, head_mask, num_hidden_layers, is_attention_chunked=False: [None] * num_hidden_layers if head_mask is None else head_mask

    def build_bert_local(model_config, _pretrain, checkpoint, encoder_width=None):
        cfg_path = root / "configs" / "config_bert_large.json"
        cfg = module.BertConfig.from_json_file(str(cfg_path))
        cfg.encoder_width = model_config.vision_encoder.d_model if encoder_width is None else encoder_width
        cfg.gradient_checkpointing = checkpoint
        cfg.fusion_layer = model_config.text_encoder.fusion_layer
        if not model_config.multimodal.enable:
            cfg.fusion_layer = cfg.num_hidden_layers
        # The Stage2 checkpoint contains the complete text encoder.  Constructing
        # from the public base checkpoint would waste network time and memory.
        # The vendored BERT classes target transformers 4.x; transformers 5.x
        # invokes a new tied-weight bookkeeping hook during initialization.
        # Initialization is unnecessary because every parameter is assigned from
        # the verified Stage2 shards immediately afterward.
        module.BertModel.init_weights = lambda self: None
        module.BertForMaskedLM.init_weights = lambda self: None
        return module.BertForMaskedLM(cfg)

    module.build_bert = build_bert_local
    return module


def _load_model(root: Path, device: torch.device, dtype: torch.dtype):
    from safetensors import safe_open

    module = _load_upstream_module(root)
    config = module.InternVideo2_Stage2_Config.from_pretrained(str(root))
    config.device = str(device)
    # Keep all construction on meta; every checkpoint tensor is assigned once
    # directly on the target device, avoiding a second 6B CPU copy.
    # The upstream constructor materializes a drop-path schedule and immediately
    # calls .item() on it.  Keep that tiny schedule on CPU while all learned
    # tensors remain meta; this is the same audited adapter used by the native
    # Stage2 smoke path.
    _orig_linspace = torch.linspace
    def _cpu_linspace(*args, **kwargs):
        kwargs["device"] = "cpu"
        return _orig_linspace(*args, **kwargs)
    torch.linspace = _cpu_linspace
    try:
        with torch.device("meta"):
            model = module.InternVideo2_Stage2(config=config, is_pretrain=True)
    finally:
        torch.linspace = _orig_linspace
    expected = set(model.state_dict())
    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    loaded = set()
    by_shard: dict[str, list[str]] = {}
    for key, shard in index["weight_map"].items():
        by_shard.setdefault(shard, []).append(key)
    for shard, names in sorted(by_shard.items()):
        state = {}
        with safe_open(str(root / shard), framework="pt", device="cpu") as stream:
            for key in names:
                tensor = stream.get_tensor(key)
                if tensor.is_floating_point():
                    tensor = tensor.to(dtype=dtype)
                state[key] = tensor.to(device=device, non_blocking=True)
                loaded.add(key)
        model.load_state_dict(state, strict=False, assign=True)
        del state
    # The HF shard intentionally omits tied MLM decoder tensors.  Recreate the
    # tie exactly as the upstream BERT class specifies before auditing missing
    # weights; an uninitialized decoder would otherwise be a silent 30k-vocab
    # parameter hole.
    decoder = model.text_encoder.cls.predictions.decoder
    decoder.weight = model.text_encoder.bert.embeddings.word_embeddings.weight
    missing = sorted(expected - loaded - {
        "text_encoder.cls.predictions.decoder.weight",
        "text_encoder.cls.predictions.decoder.bias",
    })
    if missing:
        raise RuntimeError(f"checkpoint assignment missing {len(missing)} tensors, e.g. {missing[:5]}")
    model.eval()
    count = sum(p.numel() for p in model.parameters())
    return model, module, count


def _text_features(model, prompts: list[str], device: torch.device):
    with torch.inference_mode():
        batch = model.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=model._config.max_txt_l,
            return_tensors="pt",
        ).to(device)
        _, pooled = model.encode_text(batch)
        out = model.text_proj(pooled)
        return torch.nn.functional.normalize(out, dim=-1)


def _frame_tensor(cap: cv2.VideoCapture, frame_index: int, frame_count: int, device: torch.device, dtype: torch.dtype):
    frame_index = max(0, min(int(frame_index), max(frame_count - 1, 0)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        # Some H.264 files in the official pool expose an inaccurate/random
        # seek index near the tail: OpenCV can report failure even though the
        # same ordinal frame decodes correctly in a sequential pass.  Retry
        # from the beginning and stop only after reading the requested ordinal
        # frame.  This preserves the original frame ID contract; it never
        # substitutes a neighboring frame.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame = None
        for ordinal in range(frame_index + 1):
            ok, candidate = cap.read()
            if not ok:
                raise RuntimeError(
                    f"decode failed at frame {frame_index} after sequential retry at {ordinal}"
                )
            frame = candidate
        ok = frame is not None
    if not ok or frame is None:
        raise RuntimeError(f"decode failed at frame {frame_index}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_CUBIC)
    x = torch.from_numpy(frame).permute(2, 0, 1).float().div_(255.0)
    mean = torch.tensor([.485, .456, .406]).view(3, 1, 1)
    std = torch.tensor([.229, .224, .225]).view(3, 1, 1)
    return ((x - mean) / std).to(device=device, dtype=dtype)


def _score_video(
    model,
    prompts: torch.Tensor,
    path: Path,
    device: torch.device,
    dtype: torch.dtype,
    anchor_batch_size: int = 1,
):
    if anchor_batch_size < 1:
        raise ValueError("anchor_batch_size must be >= 1")
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(fps) or fps <= 0 or n <= 0:
        cap.release()
        raise RuntimeError(f"invalid video metadata: {path}")
    duration = max((n - 1) / fps, 0.0)
    anchors = np.arange(0.0, duration + 1e-6, 1.0, dtype=np.float64)
    # Include the final frame's second if a short tail would otherwise be omitted.
    if len(anchors) == 0 or anchors[-1] < duration:
        anchors = np.r_[anchors, duration]
    offsets = (-1.5, -0.5, 0.5, 1.5)
    scores = []
    with torch.inference_mode():
        pending = []
        pending_indices = []

        def flush_pending():
            if not pending:
                return
            batch = torch.stack(pending, dim=0)
            v = model.get_vid_feat(batch).reshape(len(pending), -1)
            batch_logits = 100.0 * (v @ prompts.T)
            scores.extend(batch_logits.float().cpu().numpy())
            pending.clear()
            pending_indices.clear()

        for i, sec in enumerate(anchors):
            center = int(round(sec * fps))
            ids = [int(round((sec + off) * fps)) for off in offsets]
            pending.append(torch.stack([_frame_tensor(cap, j, n, device, dtype) for j in ids], dim=0))
            pending_indices.append(i)
            if len(pending) >= anchor_batch_size or i == len(anchors) - 1:
                flush_pending()
            if (i + 1) % 25 == 0:
                print(json.dumps({"video": path.name, "anchors": i + 1, "total_anchors": len(anchors)}), flush=True)
        flush_pending()
    cap.release()
    logits = np.asarray(scores, dtype=np.float64)
    p = {
        "event": logits[:, : len(EVENT_POSITIVE)].mean(1),
        "routine": logits[:, len(EVENT_POSITIVE):len(EVENT_POSITIVE) + len(ROUTINE_NEGATIVE)].mean(1),
    }
    start = len(EVENT_POSITIVE) + len(ROUTINE_NEGATIVE)
    p["prep"] = logits[:, start:start + len(PHASE_PREPARATION)].mean(1)
    start += len(PHASE_PREPARATION)
    p["decisive"] = logits[:, start:start + len(PHASE_DECISIVE)].mean(1)
    start += len(PHASE_DECISIVE)
    p["result"] = logits[:, start:start + len(PHASE_RESULT)].mean(1)
    event_score = p["event"] - p["routine"]
    completion = event_score + 0.50 * (p["decisive"] + p["result"] - p["prep"])
    if not np.isfinite(completion).all():
        raise RuntimeError("non-finite InternVideo2 prompt logits")
    # Map every original frame to its nearest one-second anchor.  The tiny tie
    # term prefers the frame closest to the scored center and is deterministic.
    frame_times = np.arange(n, dtype=np.float64) / fps
    frame_score = np.interp(frame_times, anchors, completion)
    return frame_score, {"fps": fps, "frame_count": n, "duration": duration, "anchors": len(anchors),
                         "anchor_batch_size": anchor_batch_size,
                         "event_score": completion.tolist(), "finite": True}


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    out = np.empty(len(values), dtype=np.float64)
    if len(values) <= 1:
        return np.ones(len(values), dtype=np.float64)
    out[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
    return out


def _load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _select(frame_score: np.ndarray, keep: int, fps: float) -> list[int]:
    n = len(frame_score)
    t = np.arange(n, dtype=np.float64) / max(fps, 1e-6)
    center = _rank(-np.abs(t / max(t[-1], 1e-6) - 0.5))
    fused = 0.75 * _rank(frame_score) + 0.25 * center
    # Deterministic original-frame tie break, exact per-video budget.
    ids = np.argsort(-fused, kind="mergesort")[:keep]
    return sorted(int(i) for i in ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-root", type=Path, required=True)
    ap.add_argument("--video-root", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--b3-predictions", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--anchor-batch-size", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    started = time.monotonic()
    model, _module, model_params = _load_model(args.model_root, device, dtype)
    prompts = _text_features(model, ALL_PROMPTS, device)
    rows = _load_jsonl(args.index)
    b3_rows = {str(r["video_id"]): r for r in _load_jsonl(args.b3_predictions)}
    if args.limit:
        rows = rows[: args.limit]
    results = []
    diagnostics = []
    result_dir = args.run_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    for pos, record in enumerate(rows):
        video_id = str(record["video_id"])
        path = args.video_root / f"{video_id}.mp4"
        if not path.exists():
            raise FileNotFoundError(path)
        saved_path = result_dir / f"{video_id}.json"
        if args.resume and saved_path.exists():
            saved = json.loads(saved_path.read_text(encoding="utf-8"))
            results.append(saved["row"])
            diagnostics.append(saved["diagnostics"])
            print(json.dumps({"resumed": pos + 1, "total": len(rows), "video_id": video_id,
                              "selected_frames": len(saved["row"]["predictions"])}), flush=True)
            continue
        frame_score, diag = _score_video(model, prompts, path, device, dtype, args.anchor_batch_size)
        keep = round(0.85 * len(frame_score))
        selected = _select(frame_score, keep, diag["fps"])
        parent = b3_rows[video_id]
        boxes = {int(x["frame"]): x["bboxes"] for x in parent["predictions"]}
        if any(i not in boxes for i in selected):
            raise RuntimeError(f"B3 parent lacks selected frame for video {video_id}")
        row = {"video_id": parent["video_id"], "targetRatioWH": parent["targetRatioWH"],
               "predictions": [{"frame": i, "bboxes": boxes[i]} for i in selected]}
        video_diag = {"video_id": video_id, "position": pos, "selected_frames": len(selected),
                      "total_frames": len(frame_score), **diag}
        results.append(row)
        diagnostics.append(video_diag)
        temporary = saved_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"row": row, "diagnostics": video_diag}, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, saved_path)
        print(json.dumps({"completed": pos + 1, "total": len(rows), "video_id": video_id,
                          "selected_frames": len(selected), "total_frames": len(frame_score)}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    report = {"experiment_id": "R43_IV2_EVENT_COMPLETION_20261017", "smoke": args.smoke,
              "videos": len(results), "model_params": model_params, "model_params_b": model_params / 1e9,
              "spatial_parent_params": 450283377, "total_declared_params": model_params + 450283377,
              "runtime_s": time.monotonic() - started, "diagnostics": diagnostics,
              "output": str(args.output), "official_score": None}
    (args.run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("videos", "model_params", "total_declared_params", "runtime_s", "output")}))


if __name__ == "__main__":
    main()
