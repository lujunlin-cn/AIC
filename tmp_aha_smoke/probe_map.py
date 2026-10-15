import sys, types, importlib.machinery, json, struct, os
sys.path.insert(0, "/root/autodl-tmp/aha_repo")
sys.path.insert(0, "/root/autodl-tmp/aha_repo/LLaVA_NeXT")
def stub(name, **attrs):
    m = types.ModuleType(name); m.__spec__ = importlib.machinery.ModuleSpec(name, None); m.__version__="0.0.0"
    for k,v in attrs.items(): setattr(m,k,v)
    sys.modules[name]=m; return m
stub("wandb"); z=stub("deepspeed.zero"); stub("deepspeed", zero=z)
class _L4b: pass
bnn=stub("bitsandbytes.nn", Linear4bit=_L4b); stub("bitsandbytes", nn=bnn)

from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionConfig, SigLipVisionModel
import torch

cfg = SigLipVisionConfig()
print("SigLipVisionConfig defaults: hidden=%s layers=%s img=%s patch=%s heads=%s" % (
    cfg.hidden_size, cfg.num_hidden_layers, cfg.image_size, cfg.patch_size, cfg.num_attention_heads))

m = SigLipVisionModel(cfg)
sd = m.state_dict()
ks = list(sd.keys())
print("instantiated SigLipVisionModel state_dict keys:", len(ks))
print("  first:", ks[0])
print("  last :", ks[-1])
lay = sorted({int(k.split("encoder.layers.")[1].split(".")[0]) for k in ks if "encoder.layers." in k})
print("  layers:", min(lay), "..", max(lay), "count", len(lay))
print("  has head.probe:", any("head.probe" in k for k in ks))
del m, sd

D="/root/autodl-tmp/pretrained/llava-onevision-qwen2-7b-ov"
raw = {}
for i in range(1,5):
    fp = os.path.join(D, f"model-0000{i}-of-00004.safetensors")
    with open(fp,"rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for k,v in json.loads(f.read(n)).items():
            if k=="__metadata__": continue
            raw[k] = v["shape"]
vis = {k:v for k,v in raw.items() if k.startswith("model.vision_tower.vision_tower.")}
print()
print("local llava vision_tower tensors:", len(vis))
stripped = {k[len("model.vision_tower.vision_tower."):]: v for k,v in vis.items()}
print("  sample stripped:", sorted(stripped)[:3])
lay2 = sorted({int(k.split("encoder.layers.")[1].split(".")[0]) for k in stripped if "encoder.layers." in k})
print("  layers:", min(lay2), "..", max(lay2), "count", len(lay2))
print("  has head.probe:", any("head.probe" in k for k in stripped))
# compare against a 26-layer model
cfg26 = SigLipVisionConfig(num_hidden_layers=26)
m26 = SigLipVisionModel(cfg26); m26.vision_model.head = torch.nn.Identity()
sd26 = m26.state_dict()
miss = [k for k in sd26 if k not in stripped]
extra = [k for k in stripped if k not in sd26]
print()
print("26-layer+Identity model keys:", len(sd26))
print("  MISSING from local:", len(miss), miss[:6])
print("  EXTRA   in local  :", len(extra), extra[:6])
shape_bad = [k for k in sd26 if k in stripped and list(sd26[k].shape) != list(stripped[k])]
print("  SHAPE MISMATCH:", len(shape_bad), shape_bad[:6])
