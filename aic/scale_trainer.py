"""Model-agnostic causal SFT/DPO steps for a Video-ORA scale-action adapter."""

from __future__ import annotations

import math
from typing import Iterable

import torch

from aic.scale_objectives import causal_completion_logps, causal_completion_sft_loss, dpo_loss


def set_trainable_parameter_allowlist(model, allowed_names: Iterable[str]) -> dict:
    allowed = set(allowed_names)
    if not allowed:
        raise ValueError("an explicit nonempty trainable-parameter allowlist is required")
    parameters = dict(model.named_parameters())
    missing = allowed - set(parameters)
    if missing:
        raise ValueError(f"allowlisted parameter names are missing: {sorted(missing)}")
    for name, parameter in parameters.items():
        parameter.requires_grad_(name in allowed)
    trainable = [name for name, parameter in parameters.items() if parameter.requires_grad]
    if set(trainable) != allowed:
        raise RuntimeError("trainable parameter allowlist was not applied exactly")
    return {
        "trainable_names": trainable,
        "trainable_parameters": sum(parameters[name].numel() for name in trainable),
        "frozen_parameters": sum(parameter.numel() for parameter in parameters.values()
                                  if not parameter.requires_grad),
    }


def unique_parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _logits(output) -> torch.Tensor:
    if hasattr(output, "logits"):
        result = output.logits
    elif isinstance(output, dict) and "logits" in output:
        result = output["logits"]
    else:
        raise ValueError("model output must expose causal logits")
    if not isinstance(result, torch.Tensor):
        raise ValueError("model logits must be a tensor")
    return result


def _forward_logps(model, item: dict) -> torch.Tensor:
    output = model(**item["model_inputs"])
    return causal_completion_logps(
        _logits(output), item["labels"], item["completion_mask"]
    )


def sft_train_step(model, optimizer, batch: dict, max_grad_norm: float = 1.0) -> dict:
    if "model_inputs" not in batch or "labels" not in batch or "completion_mask" not in batch:
        raise ValueError("SFT batch is missing model inputs or completion labels")
    optimizer.zero_grad(set_to_none=True)
    logits = _logits(model(**batch["model_inputs"]))
    loss = causal_completion_sft_loss(logits, batch["labels"], batch["completion_mask"])
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite SFT loss")
    loss.backward()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable or not any(parameter.grad is not None for parameter in trainable):
        raise RuntimeError("SFT produced no trainable gradients")
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm, error_if_nonfinite=True)
    if not math.isfinite(float(grad_norm)):
        raise RuntimeError("nonfinite SFT gradient norm")
    optimizer.step()
    completion_tokens = int((batch["completion_mask"][:, 1:].bool()
                             & batch["labels"][:, 1:].ne(-100)).sum().item())
    return {"loss": float(loss.detach()), "grad_norm": float(grad_norm),
            "completion_tokens": completion_tokens}


def _assert_same_dpo_prompt(batch: dict) -> None:
    if "chosen" not in batch or "rejected" not in batch:
        raise ValueError("DPO batch requires chosen and rejected examples")
    chosen_hash = batch.get("chosen_prompt_sha256")
    rejected_hash = batch.get("rejected_prompt_sha256")
    if not chosen_hash or chosen_hash != rejected_hash:
        raise ValueError("chosen/rejected must use the identical prompt and template")
    for left_key, right_key, description in (
        ("chosen_source_id", "rejected_source_id", "source"),
        ("chosen_image_id", "rejected_image_id", "image"),
        ("chosen_image_sha256", "rejected_image_sha256", "image bytes"),
    ):
        left, right = batch.get(left_key), batch.get(right_key)
        if not left or left != right:
            raise ValueError(f"chosen/rejected must use the same {description}")
    if batch.get("chosen_position_xy") != batch.get("rejected_position_xy"):
        raise ValueError("chosen/rejected must use the same frozen position")
    if batch.get("chosen_target_ratio") != batch.get("rejected_target_ratio"):
        raise ValueError("chosen/rejected must use the same target ratio")
    if batch.get("position_source") != "P1_frozen":
        raise ValueError("DPO batch must use the frozen P1 position contract")


def dpo_train_step(model, reference_model, optimizer, batch: dict, beta: float,
                   max_grad_norm: float = 1.0) -> dict:
    _assert_same_dpo_prompt(batch)
    if any(parameter.requires_grad for parameter in reference_model.parameters()):
        raise ValueError("DPO reference parameters must be frozen")
    optimizer.zero_grad(set_to_none=True)
    policy_chosen = _forward_logps(model, batch["chosen"])
    policy_rejected = _forward_logps(model, batch["rejected"])
    with torch.no_grad():
        reference_chosen = _forward_logps(reference_model, batch["chosen"])
        reference_rejected = _forward_logps(reference_model, batch["rejected"])
    loss = dpo_loss(policy_chosen, policy_rejected, reference_chosen,
                    reference_rejected, beta)
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite DPO loss")
    loss.backward()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable or not any(parameter.grad is not None for parameter in trainable):
        raise RuntimeError("DPO produced no trainable gradients")
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm, error_if_nonfinite=True)
    if not math.isfinite(float(grad_norm)):
        raise RuntimeError("nonfinite DPO gradient norm")
    optimizer.step()
    with torch.no_grad():
        policy_margin = policy_chosen - policy_rejected
        reference_margin = reference_chosen - reference_rejected
    return {
        "loss": float(loss.detach()),
        "policy_chosen_logp": float(policy_chosen.detach().mean()),
        "policy_rejected_logp": float(policy_rejected.detach().mean()),
        "reference_chosen_logp": float(reference_chosen.mean()),
        "reference_rejected_logp": float(reference_rejected.mean()),
        "chosen_rejected_margin": float((policy_margin - reference_margin).mean()),
        "reference_log_ratio": float((policy_chosen.detach() - reference_chosen
                                       - policy_rejected.detach() + reference_rejected).mean()),
        "grad_norm": float(grad_norm),
    }
