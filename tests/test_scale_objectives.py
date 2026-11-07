from types import SimpleNamespace

import pytest
import torch
from torch import nn

from aic.scale_objectives import (
    causal_completion_logps,
    causal_completion_sft_loss,
    dpo_loss,
    paired_source_bootstrap_ci,
    position_drift_metrics,
    scale_collapse_rate,
    source_macro_mean,
)
from aic.scale_trainer import dpo_train_step, set_trainable_parameter_allowlist, sft_train_step


def test_sft_and_dpo_losses_have_finite_nonzero_gradients_on_cpu():
    logits = torch.randn(2, 5, 11, requires_grad=True)
    labels = torch.tensor([[-100, -100, 2, 3, -100], [-100, 4, 5, 6, -100]])
    mask = labels.ne(-100)
    sft = causal_completion_sft_loss(logits, labels, mask)
    sft.backward(retain_graph=True)
    assert torch.isfinite(sft)
    assert logits.grad is not None and torch.count_nonzero(logits.grad) > 0

    values = [torch.tensor([v], requires_grad=need_grad) for v, need_grad in
              ((-1.0, True), (-2.0, True), (-1.5, False), (-2.1, False))]
    loss = dpo_loss(values[0], values[1], values[2], values[3], beta=0.1)
    loss.backward()
    assert torch.isfinite(loss)
    assert values[0].grad is not None and values[0].grad.abs().item() > 0


def test_completion_logps_ignore_prompt_and_noncompletion_tokens():
    logits = torch.zeros(1, 4, 5)
    logits[0, 1, 2] = 2.0
    labels = torch.tensor([[-100, -100, 2, -100]])
    mask = torch.tensor([[False, False, True, False]])
    score = causal_completion_logps(logits, labels, mask)
    assert score.shape == (1,)
    assert score.item() == pytest.approx(torch.log_softmax(logits[0, 1], -1)[2].item())


class TinyCausalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.head = nn.Linear(4, 8)

    def forward(self, input_ids):
        return SimpleNamespace(logits=self.head(self.embedding(input_ids)))


def _example(ids):
    input_ids = torch.tensor([ids])
    labels = input_ids.clone()
    labels[:, :2] = -100
    return {"model_inputs": {"input_ids": input_ids}, "labels": labels,
            "completion_mask": labels.ne(-100)}


def test_cpu_sft_and_dpo_steps_update_only_allowlisted_parameters():
    model = TinyCausalModel()
    result = set_trainable_parameter_allowlist(model, ["head.weight", "head.bias"])
    assert result["trainable_parameters"] == model.head.weight.numel() + model.head.bias.numel()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    before = model.head.weight.detach().clone()
    metrics = sft_train_step(model, optimizer, _example([1, 2, 3, 4]))
    assert metrics["completion_tokens"] == 2
    assert not torch.equal(before, model.head.weight.detach())

    reference = TinyCausalModel()
    reference.load_state_dict(model.state_dict())
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    pair = {
        "chosen": _example([1, 2, 3, 4]),
        "rejected": _example([1, 2, 4, 3]),
        "chosen_prompt_sha256": "same-prompt",
        "rejected_prompt_sha256": "same-prompt",
        "chosen_source_id": "source-a",
        "rejected_source_id": "source-a",
        "chosen_image_id": "image-a",
        "rejected_image_id": "image-a",
        "chosen_image_sha256": "same-image-bytes",
        "rejected_image_sha256": "same-image-bytes",
        "chosen_position_xy": [100, 200],
        "rejected_position_xy": [100, 200],
        "chosen_target_ratio": [9, 16],
        "rejected_target_ratio": [9, 16],
        "position_source": "P1_frozen",
    }
    dpo_metrics = dpo_train_step(model, reference, optimizer, pair, beta=0.1)
    assert dpo_metrics["grad_norm"] > 0
    assert torch.isfinite(torch.tensor(dpo_metrics["loss"]))
    with pytest.raises(ValueError, match="identical prompt"):
        dpo_train_step(model, reference, optimizer,
                       dict(pair, rejected_prompt_sha256="different"), beta=0.1)
    with pytest.raises(ValueError, match="same image bytes"):
        dpo_train_step(model, reference, optimizer,
                       dict(pair, rejected_image_sha256="different-image"), beta=0.1)


def test_source_macro_and_bootstrap_use_sources_as_units():
    rows = [
        {"source_id": "a", "iou": 1.0},
        {"source_id": "a", "iou": 0.0},
        {"source_id": "b", "iou": 0.5},
    ]
    assert source_macro_mean(rows) == pytest.approx(0.5)
    ci = paired_source_bootstrap_ci({"a": 0.1, "b": 0.2, "c": -0.1}, seed=11, replicates=500)
    assert ci["sources"] == 3 and ci["bootstrap_unit"] == "source"
    assert scale_collapse_rate([0, 0, 1, 2]) == 0.5
    drift = position_drift_metrics({"a": [1, 2]}, {"a": [2, 2]})
    assert drift["within_one_pixel_tolerance"]
    assert drift["max_axis_drift_px"] == 1
