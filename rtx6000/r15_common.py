#!/usr/bin/env python3
"""R15 RTX PRO 6000 RL stack: NPU-agnostic shared helpers.

Extracted verbatim from r13_grpo_train.py (PROMPT / load_frames /
build_inputs / truncate_at_eos / seq_logprob_loss / orarl_advantage) so the
CUDA migration keeps the exact same prompt, preprocessing and loss math.
reward v4 is imported from r15_grpo_reward.py (byte-identical copy of
r13_grpo_reward.py, see its header for the v4 provenance).

NO torch_npu, NO safe_chunk patch, NO jit_compile flags here: on CUDA the
stock transformers 5.19 code path is the correct path. GDN kernel selection
(fla triton vs safe torch chunk) is decided per-process in the entry
scripts, gated by r15_parity.py.
"""
import json
import os

import numpy as np
import torch

from r15_grpo_reward import parse_scores, reward_v4  # noqa: F401

PROMPT = ("These are {n} frames sampled in order from one video. "
          "Score each frame 0.00-1.00 for how likely it appears in a "
          "user-made highlight GIF of this video. Reply with ONLY a JSON "
          "array of {n} numbers, nothing else.")


def load_frames(row):
    import cv2
    fl = []
    for fn in row["files"]:
        img = cv2.imread(os.path.join(row["frames_dir"], fn))
        if img is None:
            raise IOError(f"{row['vid']}: bad jpg {fn}")
        fl.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return fl


def build_texts(proc, rows):
    """Chat-template prompt text with no-think prefill, identical to
    r13_grpo_train.build_inputs (same replace of the <think> block)."""
    texts = []
    for r in rows:
        n = len(r["gif_hard"])
        msgs = [{"role": "user", "content": [
            {"type": "video", "video": None},
            {"type": "text", "text": PROMPT.format(n=n)},
        ]}]
        t = proc.apply_chat_template(msgs, tokenize=False,
                                     add_generation_prompt=True)
        t = t.replace("<think>\n", "<think>\n\n</think>\n\n", 1)
        texts.append(t)
    return texts


def build_inputs(proc, rows, dev):
    """HF/transformers batch inputs (trainer path + parity baseline).
    Left padding, capped pixels per frame, fps=1.0 video metadata."""
    texts = build_texts(proc, rows)
    proc.tokenizer.padding_side = "left"
    frames = [load_frames(r) for r in rows]
    inputs = proc(text=texts, videos=frames, padding=True,
                  cap_pixels_per_frame=True,
                  video_metadata=[{"fps": 1.0, "total_num_frames": len(fl)}
                                  for fl in frames],
                  return_tensors="pt")
    inputs = {k: v.to(dev) if torch.is_tensor(v) else v
              for k, v in inputs.items()}
    return inputs, texts[0]


def truncate_at_eos(ids_1d, eos_id, pad_id):
    keep = []
    for t in ids_1d.tolist():
        if t == eos_id:
            keep.append(eos_id)
            break
        if t == pad_id:
            break
        keep.append(t)
    return torch.tensor(keep, dtype=ids_1d.dtype)


def seq_logprob_loss(model, prompt_row, tail_ids_list, advs, eos_id, K=280,
                     entropy_beta=0.0, ref_model=None, kl_beta=0.0):
    """Unchanged math from r13_grpo_train.py: policy gradient over the 3+1
    sequences, one forward each (bsz1 accumulation, video tensors shared).
    Loss per sequence = -A * mean token log-prob of the tail."""
    losses = []
    for tail, a in zip(tail_ids_list, advs):
        if a is None or abs(a) < 1e-8 or tail.numel() == 0:
            continue
        tail = tail.to(prompt_row["input_ids"].device).view(1, -1)
        ids = torch.cat([prompt_row["input_ids"], tail], dim=1)
        am = torch.cat([prompt_row["attention_mask"],
                        torch.ones_like(tail)], dim=1)
        labels = ids.clone()
        labels[am == 0] = -100
        labels[0, :-tail.shape[1]] = -100
        # eos masking ONLY in the tail region: the chat prompt itself
        # contains <|im_end|> == eos_id (smoke bug 3, 2026-10-11)
        t0_ = prompt_row["input_ids"].shape[1]
        ep = (ids[0, t0_:] == eos_id).nonzero()
        if ep.numel() > 0:
            labels[0, t0_ + ep[0, 0] + 1:] = -100
        mmtti = prompt_row.get("mm_token_type_ids")
        if mmtti is not None:
            mmtti = torch.cat([mmtti, torch.zeros(1, tail.shape[1],
                                                  dtype=mmtti.dtype,
                                                  device=mmtti.device)], dim=1)
        out = model(input_ids=ids, attention_mask=am,
                    pixel_values=prompt_row.get("pixel_values"),
                    video_grid_thw=prompt_row.get("video_grid_thw"),
                    mm_token_type_ids=mmtti,
                    logits_to_keep=min(K, ids.shape[1]))
        logits = out.logits[:, :-1]
        tgt = labels[:, -logits.shape[1]:]
        lp = torch.log_softmax(logits.float(), dim=-1)
        tok_lp = lp.gather(-1, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        m = tgt.ne(-100)
        if int(m.sum()) == 0:
            continue
        term = -a * tok_lp[m].mean()
        if entropy_beta > 0.0:
            # policy entropy on the same tail positions (full-vocab over the
            # already-computed log_softmax); maximise H -> subtract beta*H
            ent = -(lp.exp() * lp).sum(-1)[m].mean()
            term = term - entropy_beta * ent
        if ref_model is not None and kl_beta > 0.0:
            # KL(current || ref) on the same tail positions, against the
            # frozen init policy (ckpt_030) - the anchoring term that keeps
            # the policy from drifting into the single-template basin
            with torch.no_grad():
                ref_out = ref_model(input_ids=ids, attention_mask=am,
                                    pixel_values=prompt_row.get("pixel_values"),
                                    video_grid_thw=prompt_row.get("video_grid_thw"),
                                    mm_token_type_ids=mmtti,
                                    logits_to_keep=min(K, ids.shape[1]))
                lp_ref = torch.log_softmax(ref_out.logits[:, :-1].float(), dim=-1)
            p_cur = lp.exp()
            kl = (p_cur * (lp - lp_ref)).sum(-1)[m].mean()
            term = term + kl_beta * kl
        losses.append(term)
    return torch.stack(losses).sum() if losses else None


def orarl_advantage(r_on, r_gt):
    """Unchanged from r13_grpo_train.py (anchor = 2*w_q, 2026-10-11)."""
    mu = float(np.mean(r_on))
    adv = [r - mu for r in r_on]
    adv = [a * (1.5 if a > 0 else 1.0) for a in adv]
    mean = float(np.mean(adv))
    adv = [a - mean for a in adv]
    if r_gt <= mu:
        w_q = 0.0
    else:
        w_q = min(1.0, ((r_gt - mu) / (r_gt + 1e-6)) ** 2)
    adv_gt = 2.0 * w_q
    return adv, adv_gt
