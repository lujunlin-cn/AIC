"""Offline bridge to the cloned Video-ORA compact loader used by prior runs.

Imports the exact, hash-verified remote ``model_io.py`` only on explicit startup.
No Hub download or alternate base model is created. Actual frozen P1 banks are
loaded before the explicitly allowlisted, isolated Scale LoRA is injected.
"""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

from PIL import Image
import torch
from torch.nn import functional as F

from aic.scale_runtime import format_scale_action, parse_scale_action, verify_tokenizer_roundtrip


def build_prompt(row: dict, bins: list[float]) -> str:
    width, height = row["image_width"], row["image_height"]
    center = [row["center_xy"][0] / width, row["center_xy"][1] / height]
    return (
        "Choose the composition scale for this image. Keep the given crop center fixed. "
        "Scale is the linear size relative to the maximum legal window at the target ratio. "
        f"Target width:height={row['target_ratio'][0]}:{row['target_ratio'][1]}; "
        f"fixed normalized center={json.dumps(center)}; "
        f"ordered scale_id to scale mapping={json.dumps(dict(enumerate(bins)))}. "
        'Return only JSON with one integer field: {"scale_id":k}.'
    )


def collate_completions(processor, samples: list[dict], actions: list[int],
                        device: str, retained_tokens=None) -> dict:
    if not samples or len(samples) != len(actions):
        raise ValueError("completion rows and actions must align")
    tokenizer = processor.tokenizer
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        values = processor(text=[sample["prompt"] for sample in samples],
                           images=[sample["image"] for sample in samples],
                           add_special_tokens=False, do_resize=False, padding=True,
                           return_tensors="pt")
        encoded = tokenizer([format_scale_action(action) + tokenizer.eos_token for action in actions],
                            add_special_tokens=False, padding=True, return_tensors="pt")
    finally:
        tokenizer.padding_side = old_side
    prompt_lengths = values["attention_mask"].sum(1).long()
    target_lengths = encoded["attention_mask"].sum(1).long()
    if (prompt_lengths <= 0).any() or (target_lengths <= 0).any():
        raise ValueError("every completion needs a nonempty prompt and response")
    target_width = int(target_lengths.max())
    total_width = int((prompt_lengths + target_lengths).max())
    ids = torch.full((len(samples), total_width), tokenizer.pad_token_id, dtype=torch.long)
    attention = torch.zeros_like(ids)
    labels = torch.full((len(samples), target_width), -100, dtype=torch.long)
    prediction_positions = torch.zeros_like(labels)
    original_shape = values["input_ids"].shape
    tails = {key: torch.zeros((len(samples), total_width), dtype=value.dtype)
             for key, value in values.items()
             if key not in {"input_ids", "attention_mask"} and value.ndim == 2
             and value.shape == original_shape}
    for i, (prompt_length, target_length) in enumerate(zip(prompt_lengths.tolist(), target_lengths.tolist())):
        target = encoded["input_ids"][i, :target_length]
        if retained_tokens is not None and not set(target.tolist()) <= retained_tokens:
            raise ValueError("scale action contains tokens absent from the compact vocabulary")
        ids[i, :prompt_length] = values["input_ids"][i, :prompt_length]
        ids[i, prompt_length:prompt_length + target_length] = target
        attention[i, :prompt_length + target_length] = 1
        labels[i, :target_length] = target
        prediction_positions[i] = (prompt_length - 1 + torch.arange(target_width)).clamp_max(total_width - 1)
        for key in tails:
            tails[key][i, :prompt_length] = values[key][i, :prompt_length]
    inputs = {"input_ids": ids, "attention_mask": attention, **tails}
    inputs.update({key: value for key, value in values.items() if key not in inputs})
    return {"values": {key: value.to(device) for key, value in inputs.items()},
            "labels": labels.to(device), "positions": prediction_positions.to(device),
            "token_counts": target_lengths.to(device)}


class VideoOraScaleBackend:
    def __init__(self, config: dict, device: str):
        from transformers import AutoProcessor

        settings = config["runtime"]
        if device.startswith("cuda") and not torch.cuda.is_bf16_supported():
            raise ValueError("the preregistered BF16 path is unsupported on this GPU")
        runtime_dir = Path(settings["runtime_dir"]).resolve()
        loader = runtime_dir / "model_io.py"
        if hashlib.sha256(loader.read_bytes()).hexdigest() != settings["model_io_sha256"]:
            raise ValueError("remote model_io.py does not match its registered hash")
        for name, expected in settings["runtime_source_hashes"].items():
            if hashlib.sha256((runtime_dir / name).read_bytes()).hexdigest() != expected:
                raise ValueError("runtime source hash mismatch: " + name)
        sys.path.insert(0, str(runtime_dir))
        spec = importlib.util.spec_from_file_location("aic_scale_verified_model_io", loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.audit_base(settings["model_dir"])
        if hashlib.sha256((Path(settings["model_dir"]) / "tokenizer.json").read_bytes()).hexdigest() != config["parent"]["tokenizer_sha256"]:
            raise ValueError("parent tokenizer hash mismatch")
        # The historical train_arm loader creates fresh adapters. Load the actual
        # P1 banks instead, then inject only a budgeted, isolated Scale delta.
        from compact_model import load_compact
        from peft import PeftModel, LoraConfig
        base_model = load_compact(settings["model_dir"], device=device)
        for arm in ("portrait", "vertical"):
            adapter_path = Path(settings["parent_adapter_root"]) / arm / "adapter_model.safetensors"
            if hashlib.sha256(adapter_path.read_bytes()).hexdigest() != settings["parent_adapter_hashes"][arm]:
                raise ValueError("P1 adapter hash mismatch: " + arm)
        self.model = PeftModel.from_pretrained(base_model, str(Path(settings["parent_adapter_root"]) / "portrait"),
                                                adapter_name="default", is_trainable=False)
        self.model.load_adapter(str(Path(settings["parent_adapter_root"]) / "vertical"),
                                adapter_name="other_geometry", is_trainable=False)
        self.model.add_adapter("scale", LoraConfig(r=1, lora_alpha=2, lora_dropout=0.0,
            target_modules=settings["scale_target_modules"], bias="none", task_type="CAUSAL_LM"))
        self.allowed_names = config["training"]["trainable_parameter_allowlist"]
        from aic.scale_trainer import set_trainable_parameter_allowlist
        set_trainable_parameter_allowlist(self.model, self.allowed_names)
        self.parent_state_sha256 = self._parent_digest()
        self.total_inference_elements = sum(p.numel() for p in self.model.parameters()) + sum(p.numel() for p in self.model.buffers())
        if self.total_inference_elements > 9_000_000_000:
            raise ValueError("complete P1 plus Scale inference system exceeds 9B")
        if self.total_inference_elements != config["training"]["post_sft_inference_parameter_count"]:
            raise ValueError("live inference parameter inventory differs from the locked config")
        self.processor = AutoProcessor.from_pretrained(settings["model_dir"], local_files_only=True)
        self.config_arm = settings["arm"]
        self.device, self.bins = device, config["data"]["scale_bins"]["values"]
        self.image_root = Path(settings["image_root"]).resolve()
        self.adapter_name = "scale"
        self.max_image_side = settings["max_image_side"]
        self.preprocessing_contract = {"max_image_side": self.max_image_side, "alignment": 32,
                                       "color": "RGB", "resize": "PIL_BICUBIC", "prompt_version": "scale_json_v1"}
        self.gradient_checkpointing = settings["gradient_checkpointing"]
        self.model_contract = {"parent_identity_sha256": config["parent"]["identity_sha256"],
                               "training_geometry": [9, 16] if self.config_arm == "a" else [16, 9],
                               "scale_bins": self.bins, "preprocessing": self.preprocessing_contract,
                               "allowlist": self.allowed_names,
                               "runtime_sources": settings["runtime_source_hashes"]}
        self._activate(scale=True)
        if self.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.enable_input_require_grads()
        else:
            self.model.gradient_checkpointing_disable()
        self.tokenizer_report = verify_tokenizer_roundtrip(self.processor.tokenizer, len(self.bins))
        base = self.model.get_base_model()
        retained = getattr(base.lm_head, "original_token_ids", None)
        self.retained_tokens = None if retained is None else {int(value) for value in retained}
        actual_devices = {str(p.device) for p in self.model.parameters()}
        if not any(value.startswith(device.split(":")[0]) for value in actual_devices):
            raise ValueError(f"the remote loader did not place the model on {device}: {actual_devices}")

    def _samples(self, rows: list[dict]) -> list[dict]:
        samples = []
        for row in rows:
            if row.get("input_archive"):
                from zipfile import ZipFile
                path = (self.image_root / row["input_archive"]).resolve()
                if not path.is_relative_to(self.image_root):
                    raise ValueError("input archive is outside the public image root")
                if not hasattr(self, "_archives"):
                    self._archives = {}
                if path not in self._archives:
                    self._archives[path] = ZipFile(path)
                payload = self._archives[path].read(row["input_image"])
            else:
                path = (self.image_root / row["input_image"]).resolve()
                if not path.is_relative_to(self.image_root):
                    raise ValueError("input image is outside the public image root")
                payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != row["image_sha256"]:
                raise ValueError(f"input image hash mismatch: {path}")
            with Image.open(io.BytesIO(payload)) as original:
                image = original.convert("RGB")
            if row.get("transposed_geometry_augmentation"):
                image = image.transpose(Image.Transpose.TRANSPOSE)
            expected_size = row.get("media_size", [row["image_width"], row["image_height"]])
            if list(image.size) != list(expected_size):
                raise ValueError("stored image size does not match manifest preprocessing")
            width, height = image.size
            image = image.resize((max(32, round(width * self.max_image_side / max(width, height) / 32) * 32),
                                  max(32, round(height * self.max_image_side / max(width, height) / 32) * 32)),
                                 resample=Image.Resampling.BICUBIC)
            messages = [{"role": "user", "content": [{"type": "image"},
                         {"type": "text", "text": build_prompt(row, self.bins)}]}]
            prompt = self.processor.apply_chat_template(messages, tokenize=False,
                                                       add_generation_prompt=True, enable_thinking=False)
            samples.append({"prompt": prompt, "image": image})
        return samples

    def _parent_digest(self):
        digest = hashlib.sha256()
        for name, value in self.model.named_parameters():
            if "lora_" in name and ".scale." not in name:
                digest.update(name.encode())
                digest.update(value.detach().cpu().view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def _activate(self, scale: bool, ratio=None):
        from aic.scale_trainer import set_trainable_parameter_allowlist
        ratio = ratio or ([9, 16] if self.config_arm == "a" else [16, 9])
        parent = "default" if list(ratio) == [9, 16] else "other_geometry"
        self.model.base_model.set_adapter([parent, "scale"] if scale else [parent])
        if scale and not getattr(self, "reference_frozen", False):
            set_trainable_parameter_allowlist(self.model, self.allowed_names)
        else:
            self.model.requires_grad_(False)

    def train_mode(self):
        self.model.train()
        self.model.get_base_model().model.visual.eval()
        for name, module in self.model.named_modules():
            if "lora_dropout" in name and ".scale" not in name:
                module.eval()

    def autocast_context(self):
        return torch.autocast("cuda", dtype=torch.bfloat16) if self.device.startswith("cuda") else nullcontext()

    def _completion_logits(self, rows: list[dict], actions: list[int]):
        if len({tuple(r["target_ratio"]) for r in rows}) != 1:
            raise ValueError("batch must use one parent geometry adapter")
        self._activate(scale=True, ratio=rows[0]["target_ratio"])
        batch = collate_completions(self.processor, self._samples(rows), actions,
                                    self.device, self.retained_tokens)
        base = self.model.get_base_model()
        hidden = base.model(**batch["values"], use_cache=False)[0]
        selected = hidden.gather(1, batch["positions"].unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))
        logits = base.lm_head(selected).float()
        labels = batch["labels"]
        if labels[labels >= 0].max() >= logits.shape[-1]:
            raise ValueError("compact head requires a token remapping that has not been verified")
        return logits, labels, batch["token_counts"]

    def completion_loss(self, rows, actions):
        from aic.scale_objectives import causal_completion_sft_loss
        logits, labels, _ = self._completion_logits(rows, actions)
        return causal_completion_sft_loss(logits, labels, labels.ne(-100), already_shifted=True)

    def completion_logps(self, rows, actions):
        logits, labels, counts = self._completion_logits(rows, actions)
        losses = F.cross_entropy(logits.flatten(0, 1), labels.flatten(),
                                 ignore_index=-100, reduction="none").view_as(labels)
        return -losses.sum(1), counts

    def predict(self, rows: list[dict], seed: int | None = None,
                temperature: float | None = None) -> list[int]:
        if len({tuple(r["target_ratio"]) for r in rows}) != 1:
            raise ValueError("generation batch must use one geometry")
        self._activate(scale=True, ratio=rows[0]["target_ratio"])
        samples = self._samples(rows)
        tokenizer = self.processor.tokenizer
        previous_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            values = self.processor(text=[s["prompt"] for s in samples], images=[s["image"] for s in samples],
                                    add_special_tokens=False, do_resize=False, padding=True,
                                    return_tensors="pt")
        finally:
            tokenizer.padding_side = previous_side
        values = {key: value.to(self.device) for key, value in values.items()}
        rng = torch.random.fork_rng(devices=[torch.device(self.device)] if self.device.startswith("cuda") else [])
        rng.__enter__()
        if seed is not None:
            torch.manual_seed(seed)
        kwargs = {"do_sample": temperature is not None, "max_new_tokens": 32, "use_cache": True}
        if temperature is not None:
            kwargs["temperature"] = temperature
        try:
            self.model.eval()
            with torch.no_grad(), self.autocast_context():
                output = self.model.generate(**values, **kwargs)
        finally:
            rng.__exit__(None, None, None)
        generated = output[:, values["input_ids"].shape[1]:]
        return [parse_scale_action(text, len(self.bins)) for text in
                self.processor.batch_decode(generated, skip_special_tokens=True)]

    def save_adapter(self, path: Path):
        if self._parent_digest() != self.parent_state_sha256:
            raise RuntimeError("frozen P1 adapter was modified")
        self.model.save_pretrained(str(path), safe_serialization=True,
                                   selected_adapters=[self.adapter_name])


def create_backend(config: dict, device: str):
    return VideoOraScaleBackend(config, device)
