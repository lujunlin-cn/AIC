"""Pinned Valen Choice adapter. Candidate IDs, never generated coordinates."""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
from pathlib import Path
import time


@dataclass
class DecisionResult:
    question_id: str
    candidate_probs: dict[str, float]
    selected_id: str
    raw_response: dict


def validate_choice(raw, qid, candidates):
    a = raw["answers"][qid]
    p = a["probabilities"]
    if set(p) != set(candidates) or any(isinstance(v, bool) or not isinstance(v, (int, float))
                                      or not math.isfinite(v) or v < 0 for v in p.values()):
        raise ValueError("Invalid candidate probability mapping")
    if not math.isclose(sum(p.values()), 1., abs_tol=1e-6):
        raise ValueError("Candidate probabilities do not sum to one")
    selected = a["choice"]
    if selected not in p or p[selected] < max(p.values()) - 1e-7:
        raise ValueError("Choice does not match maximum probability")
    return DecisionResult(qid, p, selected, raw)


class ValenDecisionAdapter:
    def __init__(self, model_path, device="cuda", max_pixels=131072, video_frames=16,
                 checkpoint=None):
        import torch
        from transformers import AutoModel
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.model_path = str(Path(model_path).resolve())
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True,
                                               dtype="auto", attn_implementation="sdpa")
        self.checkpoint = str(Path(checkpoint).resolve()) if checkpoint else None
        if self.checkpoint:
            self._load_checkpoint(self.checkpoint)
        self.model = self.model.to(device).eval()
        # Retain published FP32 parameters and the checkpoint's BF16 autocast.
        self.compiler = self.model.make_compiler(execution="shared_state")
        self.compiler.media_kwargs = {**self.compiler.media_kwargs,
                                      "images_kwargs": {"size": {"shortest_edge": 65536,
                                                                 "longest_edge": max_pixels}},
                                      "videos_kwargs": {"fps": None, "num_frames": video_frames,
                                                        "size": {"shortest_edge": 4096,
                                                                 "longest_edge": max_pixels}}}
        module = importlib.import_module(self.model.__class__.__module__)
        self.answer = module.answer
        self.parameters = sum(p.numel() for p in self.model.parameters())
        self.device = device

    def _load_checkpoint(self, checkpoint):
        """Restore the exact SFT LoRA, merger and decision head state.

        The public Valen base directory contains the frozen 2B model.  The
        training run stores only 411 updated tensors, so loading the base
        directory alone silently evaluates the untrained head.  Recreate the
        same PEFT module layout before applying the checkpoint and require
        every checkpoint tensor to have a destination.
        """
        import torch
        from peft import LoraConfig, get_peft_model
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        weights = state.get("weights")
        if not isinstance(weights, dict):
            raise ValueError(f"checkpoint has no weights mapping: {checkpoint}")
        lora_targets = [
            "linear_attn.out_proj", "linear_attn.in_proj_qkv",
            "linear_attn.in_proj_z", "linear_attn.in_proj_b",
            "linear_attn.in_proj_a", "self_attn.q_proj", "self_attn.k_proj",
            "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj",
            "mlp.up_proj", "mlp.down_proj",
        ]
        lora_keys = [k for k in weights if ".lora_A." in k]
        if lora_keys:
            cfg = state.get("config", {})
            lora = LoraConfig(
                r=int(cfg.get("lora_rank", 32)),
                lora_alpha=int(cfg.get("lora_alpha", 64)),
                target_modules=lora_targets,
                lora_dropout=0.0,
                bias="none",
                task_type=None,
            )
            self.model.backbone.language_model = get_peft_model(
                self.model.backbone.language_model, lora
            )
        destination = set(self.model.state_dict())
        missing_dest = sorted(set(weights) - destination)
        if missing_dest:
            raise RuntimeError(f"checkpoint tensors without model destination: {missing_dest[:8]}")
        _, unexpected = self.model.load_state_dict(weights, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected checkpoint tensors: {unexpected[:8]}")
        self.checkpoint_parameters = len(weights)

    def compile(self, request):
        return self.compiler.compile({"request": request})

    def decide_batch(self, requests, max_tokens=8192):
        import torch
        t = time.perf_counter()
        states = [self.compile(r) for r in requests]
        compile_s = time.perf_counter() - t
        # Qwen's visual tensors are concatenated across states.  Only batch
        # states with identical visual tensor shapes; unlike language-only
        # rows, different image grids cannot safely share one visual batch.
        def signature(state):
            parts = []
            for branch in state.questions[0].branches if state.questions else []:
                for key, value in branch.inputs.items():
                    if isinstance(value, torch.Tensor) and key in {"pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"}:
                        parts.append((key, tuple(value.shape), str(value.dtype)))
            return (tuple(parts), tuple((q.qid, len(q.branches)) for q in state.questions))
        groups = {}
        for i, state in enumerate(states):
            groups.setdefault(signature(state), []).append(i)
        logits = [None] * len(states); forward_s = 0.0
        with torch.inference_mode():
            for indices in groups.values():
                subset = [states[i] for i in indices]
                torch.cuda.synchronize(); started = time.perf_counter()
                if len(subset) == 1:
                    z = [self.model.forward_state(subset[0])]
                else:
                    z = self.model.forward_state_batch(subset, max_tokens=max_tokens)
                torch.cuda.synchronize(); forward_s += time.perf_counter() - started
                for i, value in zip(indices, z):
                    logits[i] = value
        outputs = []
        for request, state, zz in zip(requests, states, logits):
            raw = {"model": "Valen", "answers": {q.qid: self.answer(q, z) for q, z in zip(state.questions, zz)},
                   "usage": {"input_tokens": state.logical_tokens, "compute_tokens": state.compute_tokens}}
            parsed = {}
            for qid, question in request["questions"].items():
                parsed[qid] = validate_choice(raw, qid, question["criteria"])
            outputs.append({"results": parsed, "raw": raw,
                            "media": state.media, "tokens": state.compute_tokens})
        return outputs, {"compile_s": compile_s, "forward_s": forward_s,
                         "states": len(states), "tokens": [s.compute_tokens for s in states]}
