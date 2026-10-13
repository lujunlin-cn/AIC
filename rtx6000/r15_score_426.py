#!/usr/bin/env python3
"""R15 CUDA port of r13_q35_zeroshot_score.py: score the 426-video official
eval set with Qwen3.5-4B + a LoRA adapter, protocol-aligned with the
V2ARB_CKPT030 pack (official 35.44):
  - n_pts = max(4, min(max_frames, round(duration))); uniform indices
    idxs = round(i*(total-1)/(n-1)) via cv2 seek (identical formula)
  - identical prompt; default KEEPS the thinking phase (4B degrades to
    all-zero scores without it); --no-think replaces it once, matching the
    RL-training distribution lane
  - same parse order: strict json_array_scores -> lenient parse_scores ->
    None. The NPU stage-2 forced-JSON follow-up is NOT ported; unparsed
    videos are recorded with scores=null and the packager applies the same
    uniform time-ramp fallback as for parse failures.
Output per video: {video_id, frame_indices, scores, n_frames, n_points, raw}.
Run under ROLL_PY (vLLM, owns the GPU). Batched: frames are held per batch
only (426 x 12 frames at full res does not fit host RAM in one shot).
"""
import argparse
import json
import os
import re
import sys

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def json_array_scores(raw, n):
    """Exact JSON array of n numbers anywhere in raw (scale-aware, clamped,
    non-constant) or None. The ONLY path counted as a true JSON answer."""
    best = None
    for mch in re.finditer(r"\[[^\[\]]*\]", raw):
        try:
            arr = json.loads(mch.group(0))
            if (isinstance(arr, list) and len(arr) == n
                    and all(isinstance(x, (int, float)) for x in arr)):
                best = arr
                break
        except Exception:
            pass
    if best is None:
        return None
    vals = [float(x) for x in best]
    mx = max(vals) if vals else 1.0
    if mx > 1.5:
        vals = [v / 100.0 for v in vals]
    vals = [min(max(v, 0.0), 1.0) for v in vals]
    if max(vals) - min(vals) < 1e-6:
        return None
    return vals


def parse_scores(raw, n):
    """JSON array anywhere in raw (scale-aware, clamped); else last-n
    plausible-scale floats; else None. Constant arrays -> None."""
    best = None
    for mch in re.finditer(r"\[[^\[\]]*\]", raw):
        try:
            arr = json.loads(mch.group(0))
            if (isinstance(arr, list) and len(arr) == n
                    and all(isinstance(x, (int, float)) for x in arr)):
                best = arr
                break
        except Exception:
            pass
    if best is not None:
        vals = [float(x) for x in best]
        mx = max(vals) if vals else 1.0
        if mx > 1.5:
            vals = [v / 100.0 for v in vals]
        vals = [min(max(v, 0.0), 1.0) for v in vals]
        if max(vals) - min(vals) < 1e-6:
            return None
        return vals
    fl = re.findall(r"-?\d+\.\d+|-?\d+", raw)
    vals = [float(x) for x in fl]
    plaus = ([v for v in vals if 0.0 <= v <= 1.0]
             + [v / 100.0 for v in vals if 1.0 < v <= 100.0])
    if len(plaus) >= n:
        vals = plaus[-n:]
        if max(vals) - min(vals) < 1e-6:
            return None
        return vals
    return None


PROMPT = ("These are {n} frames sampled in order from one video. "
          "Score each frame 0.00-1.00 for how likely it appears in a "
          "user-made highlight GIF of this video. Reply with ONLY a JSON "
          "array of {n} numbers, nothing else.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intake", default="/root/autodl-tmp/vpb_artifacts/intake/intake_426.jsonl")
    ap.add_argument("--video-root", default="/root/autodl-tmp/vpb_artifacts/video_426")
    ap.add_argument("--base", default="/data/aic/pretrained/Qwen3.5-4B")
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter dir (LoRARequest hot-load); omit for base")
    ap.add_argument("--label", default="s030",
                    help="tag inside the out-dir name only, informational")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-frames", type=int, default=12)
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=64,
                    help="videos per vLLM submit (host-RAM frame budget)")
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(args.base, use_fast=False)
    eos_id = proc.tokenizer.eos_token_id
    llm = LLM(model=args.base, max_model_len=12288,
              gpu_memory_utilization=args.gpu_mem_util,
              enable_lora=args.adapter is not None, max_lora_rank=16,
              max_num_seqs=args.max_num_seqs,
              limit_mm_per_prompt={"video": 1})
    lreq = (LoRARequest("adapter", 1, args.adapter)
            if args.adapter else None)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens,
                        stop_token_ids=[eos_id])

    def _mm_meta(n):
        return {"fps": 1.0, "frames_indices": list(range(n)),
                "total_num_frames": n, "do_sample_frames": False}

    rows = []
    for line in open(args.intake):
        r = json.loads(line)
        rows.append((r["video_id"], float(r["duration"])))
    print(f"[426] intake={len(rows)} adapter={args.adapter} "
          f"no_think={args.no_think}", flush=True)

    n_strict = n_loose = n_fail = done = 0
    for b0 in range(0, len(rows), args.batch):
        blk = rows[b0:b0 + args.batch]
        pending = []
        for vid, dur in blk:
            out_path = os.path.join(args.out_dir, f"{vid}.json")
            if os.path.exists(out_path):
                continue
            path = os.path.join(args.video_root, f"{vid}.mp4")
            n_pts = int(max(4, min(args.max_frames, round(dur))))
            try:
                cap = cv2.VideoCapture(path)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total <= 0:
                    cap.release()
                    print(f"[426] {vid} no frames, skip", flush=True)
                    continue
                idxs = [int(round(i * (total - 1) / max(n_pts - 1, 1)))
                        for i in range(n_pts)]
                frames = []
                for i in idxs:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ok, fr = cap.read()
                    if ok:
                        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                        h, w = fr.shape[:2]
                        # vLLM token-budget guard: cap long side at 1024
                        # (≈ the parent chain's cap_pixels_per_frame scale
                        # for large videos); 12 frames otherwise overflow
                        # max_model_len on 1080p/4K sources
                        if max(h, w) > 1024:
                            s = 1024.0 / max(h, w)
                            fr = cv2.resize(fr, (int(round(w * s)),
                                                 int(round(h * s))))
                        frames.append(fr)
                cap.release()
            except Exception as e:  # noqa: BLE001
                print(f"[426] {vid} DECODE_ERR {e}", flush=True)
                continue
            if len(frames) < 4:
                print(f"[426] {vid} only {len(frames)} frames, skip",
                      flush=True)
                continue
            pending.append((vid, idxs, frames))

        if not pending:
            continue
        prompts = []
        for vid, idxs, frames in pending:
            p = PROMPT.format(n=len(frames))
            msgs = [{"role": "user", "content": [
                {"type": "video", "video": None},
                {"type": "text", "text": p}]}]
            t = proc.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True)
            if args.no_think:
                t = t.replace("<think>\n", "<think>\n\n</think>\n\n", 1)
            prompts.append({"prompt": t,
                            "multi_modal_data": {
                                "video": (np.stack(frames), _mm_meta(
                                    len(frames)))}})
        outs = llm.generate(prompts, sp, lora_request=lreq)
        for (vid, idxs, frames), o in zip(pending, outs):
            raw = o.outputs[0].text
            n = len(frames)
            sc = json_array_scores(raw, n)
            if sc is not None:
                n_strict += 1
            else:
                sc = parse_scores(raw, n)
                if sc is not None:
                    n_loose += 1
                else:
                    n_fail += 1
            with open(os.path.join(args.out_dir, f"{vid}.json"), "w") as f:
                json.dump({"video_id": vid, "frame_indices": idxs,
                           "scores": sc, "n_frames": n, "n_points": n,
                           "raw": raw[-2000:]}, f)
            done += 1
        print(f"[426] progress done={done} strict={n_strict} "
              f"loose={n_loose} fail={n_fail}", flush=True)

    print(f"[426] DONE done={done} strict={n_strict} loose={n_loose} "
          f"fail={n_fail} -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
