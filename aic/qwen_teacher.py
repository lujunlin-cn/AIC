"""Qwen-VL teacher for temporal highlight selection.

Design contract
----------------
The teacher never emits frame numbers or pixel coordinates. It observes a fixed
numbered grid of uniformly-sampled frames ("anchors"), each annotated with its
anchor id and timestamp. It returns highlight *segments* referencing anchor ids
or times. Program code then:

* maps each returned segment back to the enclosing anchor time range,
* expands that range to the set of *original decoded frame indices*,
* runs the existing spatial crop path on those frames,
* emits a validated competition JSONL row.

This keeps all frame/geometry bookkeeping deterministic and auditable; the
model only makes the semantic temporal-selection judgement.

Model loading is isolated behind ``QwenTeacher`` so the module can be imported
and unit-tested without weights (``backend="mock"`` uses a scripted segment
list). The real backend uses ``transformers`` ``Qwen3VLForConditionalGeneration``
with ``device_map`` and FP16 on V100 (no BF16/FlashAttention).
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from .video import _decoded, frame_timeline, probe_video


# ---------------------------------------------------------------------------
# Prompt / schema versioning
# ---------------------------------------------------------------------------

PROMPT_VERSION = "qwen_temporal_v1"

_SYSTEM = (
    "You are an expert video highlight editor. You are shown a numbered grid of "
    "frames sampled uniformly from one video. Each frame is labelled "
    "[id=T seconds]. Identify the segments that belong in a highlight reel: "
    "important actions, key events, emotional peaks, skilled moments, or "
    "visually striking content. Skip static, repetitive, transitional, blurry, "
    "or low-information stretches. It is acceptable for a video to have no "
    "highlight at all."
)

_USER_TMPL = (
    "These {n} frames are sampled every {step:.2f}s from a {dur:.1f}s video "
    "({ratio_desc}).\n"
    "Return ONLY a JSON object (no prose) of the form:\n"
    "{{\"segments\":[{{\"start_id\":<anchor id>,\"end_id\":<anchor id>,"
    "\"importance\":<0-1>,\"reason\":<short phrase>}}],\"summary\":<short>}}\n"
    "Rules: use only anchor ids that appear in the grid; start_id <= end_id; "
    "choose as many or as few segments as the content warrants (possibly none); "
    "importance is your confidence that the whole span is a highlight."
)


@dataclass(frozen=True)
class Anchor:
    """One uniformly-sampled observation point shown to the teacher."""
    anchor_id: int          # 0..n-1 position in the grid the model sees
    frame_index: int        # original decoded frame index
    time_seconds: float     # display time relative to first PTS


@dataclass(frozen=True)
class TeacherSegment:
    start_id: int
    end_id: int
    importance: float
    reason: str = ""


def build_anchors(path: str | Path, sample_fps: float) -> list[Anchor]:
    """Uniformly sample display times, mapping each to an original frame index.

    Anchors are the only temporal references the model may use. Sampling at a
    fixed fps over real display PTS keeps coverage uniform across long videos.
    """
    timeline = frame_timeline(path)
    if not timeline:
        raise ValueError(f"empty timeline: {path}")
    anchors: list[Anchor] = []
    cursor = 0
    n = len(timeline)
    step = 1.0 / sample_fps
    k = 0
    while True:
        t = k * step
        if t > timeline[-1].time_seconds + 1e-9:
            break
        while cursor + 1 < n and timeline[cursor + 1].time_seconds <= t + 1e-9:
            cursor += 1
        anchors.append(Anchor(len(anchors), timeline[cursor].index,
                              timeline[cursor].time_seconds))
        k += 1
    return anchors


def extract_anchor_frames(path: str | Path, anchors: Sequence[Anchor],
                          size: int = 448) -> list[np.ndarray]:
    """Decode once, returning RGB thumbnails aligned 1:1 with ``anchors``."""
    wanted = {a.frame_index: a.anchor_id for a in anchors}
    out: dict[int, np.ndarray] = {}
    for stamp, frame in _decoded(path):
        if stamp.index in wanted:
            out[wanted[stamp.index]] = _thumb(frame, size)
        if len(out) == len(wanted):
            break
    return [out[i] for i in range(len(anchors))]


def _thumb(frame, size: int) -> np.ndarray:
    rgb = frame.to_ndarray(format="rgb24")
    if size:
        h, w = rgb.shape[:2]
        scale = min(size / w, size / h, 1.0)
        if scale < 1.0:
            import cv2
            rgb = cv2.resize(rgb, (max(1, round(w * scale)),
                                   max(1, round(h * scale))),
                             interpolation=cv2.INTER_AREA)
    return rgb


def anchors_and_frames(path: str | Path, sample_fps: float,
                       size: int = 448):
    """Single decode: build the uniform anchor grid AND grab thumbnails.

    One pass over the decoder yields (anchors, frames, timeline) so callers do
    not pay a second full decode just to get PTS after sampling.
    """
    frames: list = []          # raw frames kept minimal: index->thumb
    thumbs: dict[int, np.ndarray] = {}
    timeline: list = []
    step = 1.0 / sample_fps
    next_t = 0.0
    pending: list[Anchor] = []
    for stamp, frame in _decoded(path):
        timeline.append(stamp)
        # an anchor at time t maps to the LAST decoded frame with time<=t.
        while next_t <= stamp.time_seconds + 1e-9:
            pending.append(Anchor(len(pending), stamp.index,
                                  stamp.time_seconds))
            thumbs[len(pending) - 1] = _thumb(frame, size)
            next_t += step
    anchors = pending
    return anchors, [thumbs[i] for i in range(len(anchors))], timeline


# ---------------------------------------------------------------------------
# Structured output parsing
# ---------------------------------------------------------------------------

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_segments(text: str, n_anchors: int) -> tuple[list[TeacherSegment], str]:
    """Parse the model reply; clamp ids to the grid; drop malformed entries.

    Returns (segments, status). status in {ok, empty, repaired, failed}.
    """
    if not isinstance(text, str) or not text.strip():
        return [], "failed"
    match = _JSON_BLOCK.search(text)
    if not match:
        return [], "failed"
    try:
        data = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return [], "failed"
    raw = data.get("segments", [])
    if not isinstance(raw, list):
        return [], "failed"
    segs: list[TeacherSegment] = []
    repaired = False
    for item in raw:
        if not isinstance(item, Mapping):
            repaired = True
            continue
        try:
            s = int(item["start_id"]); e = int(item["end_id"])
        except (KeyError, TypeError, ValueError):
            repaired = True
            continue
        imp = item.get("importance", 0.5)
        try:
            imp = float(imp)
        except (TypeError, ValueError):
            imp = 0.5
            repaired = True
        if not (0 <= s <= e < n_anchors):
            repaired = True
            continue
        segs.append(TeacherSegment(s, e, float(np.clip(imp, 0, 1)),
                                   str(item.get("reason", ""))[:200]))
    if not segs:
        return [], ("repaired" if repaired else "empty")
    return segs, ("repaired" if repaired else "ok")


def segments_to_frames(segments: Sequence[TeacherSegment],
                       anchors: Sequence[Anchor],
                       timeline) -> list[int]:
    """Expand anchor-id segments to original frame indices (sorted, deduped).

    A segment [start_id,end_id] covers every original frame whose display time
    lies between the start anchor's time and the next anchor after end_id (or
    the video end). This treats each anchor as the start of a sampling cell and
    is deterministic.
    """
    if not anchors:
        return []
    n = len(anchors)
    times = [a.time_seconds for a in anchors]
    video_end = timeline[-1].time_seconds if timeline else times[-1]
    selected: set[int] = set()
    for seg in segments:
        t0 = times[seg.start_id]
        t1 = (times[seg.end_id + 1] if seg.end_id + 1 < n else video_end + 1e-6)
        for stamp in timeline:
            if t0 - 1e-9 <= stamp.time_seconds <= t1 + 1e-9:
                selected.add(stamp.index)
    return sorted(selected)


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

@dataclass
class QwenTeacher:
    """Frozen teacher. ``backend='mock'`` yields scripted segments for tests."""
    model_path: str | None = None
    backend: str = "hf"               # 'hf' | 'vllm' | 'mock'
    device_map: str | Mapping | None = "auto"
    attn_implementation: str = "sdpa"  # 'eager' materializes O(N^2) attn and OOMs on many images
    max_new_tokens: int = 512
    image_size: int = 448
    block_size: int = 32  # anchors per forward pass; bounds vision-token count
    mock_segments: list = field(default_factory=list)  # for backend='mock'
    # vLLM backend: KV-cache size, CUDA-graph capture sizes and torch.compile
    # kernels are profiled/tuned by vLLM itself; only the envelope is fixed.
    tensor_parallel_size: int = 4
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 16384
    max_num_seqs: int = 16   # bounds auto CUDA-graph capture sizes (V100 32GB)

    _model: Any = field(default=None, init=False, repr=False)
    _processor: Any = field(default=None, init=False, repr=False)

    def load(self) -> None:
        if self.backend == "mock" or self._model is not None:
            return
        if self.backend == "vllm":
            from vllm import LLM
            self._model = LLM(
                model=self.model_path, dtype="float16",   # V100: no bf16
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                max_num_seqs=self.max_num_seqs,
                limit_mm_per_prompt={"image": self.block_size, "video": 0},
                seed=0)
            return
        import torch
        from transformers import (AutoProcessor,
                                  Qwen3VLForConditionalGeneration)
        dtype = torch.float16               # V100: no bf16
        self._processor = AutoProcessor.from_pretrained(self.model_path)
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path, dtype=dtype, device_map=self.device_map,
            attn_implementation=self.attn_implementation)
        self._model.eval()

    def select_segments(self, frames: Sequence[np.ndarray],
                        anchors: Sequence[Anchor],
                        target_ratio: Sequence[float],
                        sample_fps: float, duration: float) -> tuple[list[TeacherSegment], str, str]:
        """Return (segments, status, raw_text).

        Long videos are observed in anchor *blocks* (route doc section 5.1):
        each block feeds ``block_size`` consecutive anchor thumbnails to the
        model with their GLOBAL anchor ids, then returned ids map straight back
        onto the global grid. Segment ids are therefore already global and need
        no re-indexing; blocks never overlap so coverage stays uniform.
        """
        if self.backend == "mock":
            segs = [TeacherSegment(int(s["start_id"]), int(s["end_id"]),
                                   float(s.get("importance", 0.5)),
                                   str(s.get("reason", "")))
                    for s in self.mock_segments]
            raw = json.dumps({"segments": self.mock_segments})
            return segs, ("ok" if segs else "empty"), raw
        self.load()
        ratio_desc = ("landscape" if float(target_ratio[0]) > float(target_ratio[1])
                      else "portrait")
        n = len(anchors)
        bs = max(1, int(self.block_size))
        if self.backend == "vllm":
            replies = self._vllm_replies(frames, anchors, sample_fps, duration,
                                         ratio_desc, bs)
            return _merge_block_replies(replies, n)
        import torch
        replies: list[str] = []
        for lo in range(0, n, bs):
            block_frames = list(frames[lo:lo + bs])
            messages = _block_messages(block_frames, anchors[lo:lo + bs], lo, bs,
                                       n, sample_fps, duration, ratio_desc)
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = self._processor(text=[text], images=block_frames,
                                     return_tensors="pt").to(self._model.device)
            with torch.inference_mode():
                out = self._model.generate(**inputs, do_sample=False,
                                           max_new_tokens=self.max_new_tokens)
            replies.append(self._processor.batch_decode(
                out[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True)[0])
        return _merge_block_replies(replies, n)

    def _vllm_replies(self, frames, anchors, sample_fps, duration,
                      ratio_desc, bs) -> list[str]:
        """All blocks of one video in a single ``LLM.chat`` batch (greedy)."""
        from PIL import Image
        from vllm import SamplingParams
        conversations = []
        for lo in range(0, len(anchors), bs):
            pil = [Image.fromarray(f) for f in frames[lo:lo + bs]]
            conversations.append(_block_messages(
                pil, anchors[lo:lo + bs], lo, bs, len(anchors), sample_fps,
                duration, ratio_desc, image_key="image_pil"))
        params = SamplingParams(temperature=0.0, max_tokens=self.max_new_tokens,
                                seed=0)
        outs = self._model.chat(conversations, params, use_tqdm=False)
        return [o.outputs[0].text for o in outs]


def _block_messages(block_frames, block_anchors, lo, bs, n, sample_fps,
                    duration, ratio_desc, image_key="image"):
    """Chat messages for one anchor block; identical prompt for every backend."""
    ids = [a.anchor_id for a in block_anchors]
    user = _USER_TMPL.format(
        n=len(block_anchors), step=1.0 / sample_fps,
        dur=duration, ratio_desc=ratio_desc)
    # Tell the model the grid is global but it only sees this window.
    user += ("\nYou are shown anchors with GLOBAL ids "
             f"{ids[0]}..{ids[-1]} (window {lo // bs + 1}); "
             "the full video grid has ids 0.."
             f"{n - 1}. Use the global ids you actually see.")
    content = [{"type": image_key, image_key: f} for f in block_frames]
    content.append({"type": "text", "text": user})
    return [{"role": "system", "content": _SYSTEM},
            {"role": "user", "content": content}]


def _merge_block_replies(replies: Sequence[str], n: int):
    """Parse per-block replies; status is the worst block unless any segment."""
    _order = {"failed": 0, "empty": 1, "repaired": 2, "ok": 3}
    all_segs: list[TeacherSegment] = []
    worst_status = "empty"
    for reply in replies:
        segs, status = parse_segments(reply, n)  # ids are global already
        all_segs.extend(segs)
        if _order[status] < _order[worst_status]:
            worst_status = status
    if all_segs and worst_status == "empty":
        worst_status = "ok"
    all_segs.sort(key=lambda s: (s.start_id, s.end_id))
    return all_segs, worst_status, "\n".join(replies)
