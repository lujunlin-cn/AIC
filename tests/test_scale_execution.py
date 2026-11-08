import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from aic.scale_execution import (fingerprint, run_training, source_balanced_indices,
                                 validate_manifests, preference_accuracy_report)
from aic.scale_preferences import geometry_preference_pairs
from aic.scale_videoora import build_prompt, collate_completions
from scripts.run_scale_stage import cuda_watchdog


def sample(source: str, split: str, dataset="LIVE_YT_VC") -> dict:
    return {"dataset_source": dataset, "source_id": source, "image_id": source + ":0",
            "split": split, "image_width": 640, "image_height": 360,
            "center_xy": [320, 180], "position_source": "P1_frozen",
            "target_ratio": [16, 9], "gt_crop_xywh": [160, 90, 320, 180],
            "input_image": source + ".jpg", "image_sha256": hashlib.sha256(source.encode()).hexdigest()}


class CpuBackend:
    def __init__(self, fail_at=None):
        torch.manual_seed(17)
        self.model = nn.Linear(1, 2)
        self.calls = 0
        self.fail_at = fail_at

    def train_mode(self):
        self.model.train()

    def completion_logps(self, rows, actions):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("simulated interruption")
        logits = self.model(torch.ones(len(rows), 1))
        scores = logits.log_softmax(-1).gather(1, torch.tensor(actions).unsqueeze(-1)).squeeze(-1)
        return scores, torch.ones(len(rows), dtype=torch.long)

    def predict(self, rows, **kwargs):
        return self.model(torch.ones(len(rows), 1)).argmax(-1).tolist()

    def save_adapter(self, path):
        path.mkdir()
        torch.save(self.model.state_dict(), path / "adapter.pt")


def config():
    return {
        "training": {"trainable_parameter_allowlist": ["weight", "bias"],
                     "max_steps": 2, "batch_size": 2, "optimizer_learning_rate": .01,
                     "eval_every": 1, "seed": 17, "dpo_beta": .1,
                     "majority_scale_upper_bound": 1.0, "sft_checkpoint_sha256": "a" * 64},
        "data": {"scale_bins": {"values": [.5, 1.0]},
                 "minimum_position_conditioned_iou": .2,
                 "source_balanced_weights": {"LIVE_YT_VC": 1, "GNMC": 1}},
    }


def manifests():
    return ([sample("train1", "train"), sample("train2", "train", "GNMC")],
            [sample("val1", "val"), sample("val2", "val", "GNMC")])


def test_sft_loop_resume_matches_uninterrupted_optimizer_and_source_schedule(tmp_path):
    train, val = manifests()
    normal = CpuBackend()
    result = run_training(normal, train, val, config(), tmp_path / "normal", "scale_sft")
    interrupted = CpuBackend(fail_at=2)
    with pytest.raises(RuntimeError, match="interruption"):
        run_training(interrupted, train, val, config(), tmp_path / "resumed", "scale_sft")
    resumed = CpuBackend()
    run_training(resumed, train, val, config(), tmp_path / "resumed", "scale_sft",
                 tmp_path / "resumed/checkpoint-000001")
    for name, value in normal.model.named_parameters():
        assert torch.equal(value, dict(resumed.model.named_parameters())[name])
    assert result["stage"] == "scale_sft"
    assert len(result["evaluations"]) == 2
    assert (tmp_path / "normal/best.json").is_file()


def test_dpo_loop_uses_frozen_sft_reference_cache_and_resumes_without_recomputing_it(tmp_path):
    train, val = manifests()
    pairs = [dict(row, **geometry_preference_pairs(row, [.5, 1.0], .1, 1)[0]) for row in train]
    backend = CpuBackend()
    before = backend.model.weight.detach().clone()
    run_training(backend, pairs, val, config(), tmp_path / "dpo", "spatial_dpo")
    cache = torch.load(tmp_path / "dpo/frozen_reference_logps.pt", weights_only=True)
    assert cache["sft_checkpoint_sha256"] == "a" * 64
    assert cache["values"].requires_grad is False
    assert not torch.equal(before, backend.model.weight)
    metrics = json.loads((tmp_path / 'dpo/checkpoint-000002/evaluation.json').read_text())
    assert metrics['training_preference_accuracy']['used_for_model_selection'] is False
    observed = metrics['training_preference_accuracy']['metrics_by_dataset_and_supervision']
    assert sum(m['sampled_pair_evaluations'] for m in observed.values()) == config()['training']['batch_size']
    assert all(0 <= m['source_macro_preference_accuracy'] <= 1 for m in observed.values())
    original_cache = (tmp_path / "dpo/frozen_reference_logps.pt").read_bytes()
    run_training(CpuBackend(), pairs, val, config(), tmp_path / "dpo", "spatial_dpo",
                 tmp_path / "dpo/checkpoint-000002")
    assert (tmp_path / "dpo/frozen_reference_logps.pt").read_bytes() == original_cache


def test_preference_accuracy_is_source_macro_and_keeps_supervision_semantics():
    observations = [dict(dataset_source='LIVE',source_id='many',score_name='IoU',correct=True)] * 30
    observations += [dict(dataset_source='LIVE',source_id='one',score_name='IoU',correct=False)]
    observations += [dict(dataset_source='GAICD',source_id='many',score_name='MOS',correct=False)]
    report = preference_accuracy_report(observations)
    assert report['split'] == 'train' and not report['used_for_model_selection']
    assert report['metrics_by_dataset_and_supervision']['LIVE:IoU']['source_macro_preference_accuracy'] == .5
    assert report['metrics_by_dataset_and_supervision']['GAICD:MOS']['source_macro_preference_accuracy'] == 0.


def test_manifest_validation_rejects_source_and_image_leakage_and_illegal_parent_center():
    train, val = manifests()
    with pytest.raises(ValueError, match="source crosses"):
        validate_manifests(train, [dict(train[0], split="val")])
    with pytest.raises(ValueError, match="identical image"):
        validate_manifests(train, [dict(val[0], image_sha256=train[0]["image_sha256"])])
    from aic.scale_preferences import fixed_center_sft_target
    invalid = fixed_center_sft_target(dict(train[0], center_xy=[0, 0]), [.5, 1.0], .2)
    assert invalid["label_status"] == "position_shift_invalid"
    assert invalid["scale_id"] is None


def test_source_balancing_does_not_multiply_weight_by_frame_count():
    rows = [sample("many_frames", "train")] * 100 + [sample("one_frame", "train")]
    indices = source_balanced_indices(rows, {"LIVE_YT_VC": 1}, 2000, 22)
    fraction = sum(rows[index]["source_id"] == "one_frame" for index in indices) / len(indices)
    assert .45 < fraction < .55
    assert indices == source_balanced_indices(rows, {"LIVE_YT_VC": 1}, 2000, 22)


class CharTokenizer:
    padding_side = "left"
    eos_token = "~"
    pad_token_id = 0

    def __call__(self, texts, **kwargs):
        sequences = [[ord(char) for char in text] for text in texts]
        width = max(map(len, sequences))
        return {"input_ids": torch.tensor([ids + [0] * (width - len(ids)) for ids in sequences]),
                "attention_mask": torch.tensor([[1] * len(ids) + [0] * (width - len(ids)) for ids in sequences])}


class FakeProcessor:
    tokenizer = CharTokenizer()

    def __call__(self, text, images, **kwargs):
        values = self.tokenizer(text)
        values["pixel_values"] = torch.ones(len(text), 3)
        values["mm_token_type_ids"] = torch.ones_like(values["input_ids"])
        return values


def test_multimodal_collator_masks_prompt_and_padding_and_keeps_images():
    processor = FakeProcessor()
    samples = [{"prompt": "abc", "image": object()}, {"prompt": "abcdefgh", "image": object()}]
    batch = collate_completions(processor, samples, [0, 1], "cpu", set(range(128)))
    assert processor.tokenizer.padding_side == "left"
    assert batch["values"]["pixel_values"].shape == (2, 3)
    assert batch["positions"][0, 0] == 2
    assert batch["positions"][1, 0] == 7
    assert batch["values"]["mm_token_type_ids"][0, 3:].sum() == 0
    assert batch["labels"][0, 0] == ord("{")
    assert "gt_crop" not in build_prompt(sample("train", "train"), [.5, 1.0])
    with pytest.raises(ValueError, match="compact vocabulary"):
        collate_completions(processor, samples, [0, 1], "cpu", {1, 2})


def test_watchdog_persists_consumed_process_time_in_ledger_without_using_cuda(tmp_path):
    import sys

    args = SimpleNamespace(budget=tmp_path / "budget.json", stage="scale_sft", output=tmp_path / "out")
    result = cuda_watchdog(args, [sys.executable, "-c", "pass"])
    ledger = json.loads(args.budget.read_text())
    assert result == 0
    assert ledger["total_used_hours"] > 0
    assert ledger["stages"]["scale_sft"]["used_hours"] == ledger["total_used_hours"]
    assert "active_run" not in ledger
    assert not ledger["budget_exceeded"]
