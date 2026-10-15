import json, os, collections
d = "/root/autodl-tmp/pretrained/llava-onevision-qwen2-7b-ov"
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
wm = idx["weight_map"]
c = collections.Counter(".".join(k.split(".")[:2]) for k in wm)
print("=== base model top-level prefix counts ===")
for k, v in sorted(c.items()):
    print("   %-40s %d" % (k, v))
print("VISION_TOWER tensors in base:", sum(1 for k in wm if "vision_tower" in k))
print("connector/mm_projector:", sum(1 for k in wm if "connector" in k or "mm_projector" in k))
print()
import glob
b = "/root/.cache/huggingface/hub/models--google--siglip-so400m-patch14-384/blobs"
for f in sorted(glob.glob(b + "/*")):
    print("%12d  %s" % (os.path.getsize(f), os.path.basename(f)))
