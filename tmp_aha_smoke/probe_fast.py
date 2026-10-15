import sys, types, importlib.machinery, json
sys.path.insert(0, "/root/autodl-tmp/aha_repo")
sys.path.insert(0, "/root/autodl-tmp/aha_repo/LLaVA_NeXT")
def stub(name, **attrs):
    m = types.ModuleType(name); m.__spec__ = importlib.machinery.ModuleSpec(name, None); m.__version__="0.0.0"
    for k,v in attrs.items(): setattr(m,k,v)
    sys.modules[name]=m; return m
stub("wandb"); z=stub("deepspeed.zero"); stub("deepspeed", zero=z)
class _L4b: pass
bnn=stub("bitsandbytes.nn", Linear4bit=_L4b); stub("bitsandbytes", nn=bnn)

print("=== adapter keys: connector/mm_projector ===")
from safetensors import safe_open
f = safe_open("/root/autodl-tmp/pretrained/aha_weights/adapter_model.safetensors", framework="pt")
ks = list(f.keys())
for pat in ["mm_projector","connector","image_newline"]:
    hits=[k for k in ks if pat in k]
    print("  %-14s %d  %s" % (pat, len(hits), hits[:4]))
print("  total adapter tensors:", len(ks))
nontarget=[k for k in ks if "lora_A" not in k and "lora_B" not in k]
print("  non-lora (saved modules) keys:", [k.replace("base_model.model.","") for k in nontarget])

print()
print("=== config resolution ===")
from models.live_llava.video_head_live_llava_qwen import VideoHeadLiveLlavaQwenConfig
import torch
cfg = VideoHeadLiveLlavaQwenConfig.from_pretrained(
    "/root/autodl-tmp/pretrained/llava-onevision-qwen2-7b-ov",
    vision_pretrained="google/siglip-large-patch16-384",
    frame_resolution=384, frame_token_cls=False, frame_token_pooled=[7,7], frame_num_tokens=49,
    video_pooling_stride=4, v_placeholder="<image>")
for a in ["vision_pretrained","mm_vision_tower","mm_spatial_pool_mode","mm_vision_select_feature","mm_vision_select_layer",
          "mm_patch_merge_type","frame_num_tokens","frame_token_pooled","video_pooling_stride","hidden_size","model_type"]:
    print("  %-26s = %r" % (a, getattr(cfg,a,"<ABSENT>")))
