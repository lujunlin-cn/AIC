"""Merge AHA adapter into native BF16; optionally export an unused NF4 variant."""
import os, sys, json, time, hashlib, argparse, gc
from pathlib import Path
os.environ.update(RANK='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled')
ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--also-quantize',action='store_true'); args=ap.parse_args()
root=Path(args.out); root.mkdir(parents=True,exist_ok=True)
sys.path[:0]=['/root/autodl-tmp/aha_repo','/root/autodl-tmp/aha_repo/LLaVA_NeXT']
import torch
from transformers import BitsAndBytesConfig
from models.live_llava.video_head_live_llava_qwen import build_video_head_live_llava_qwen, VideoHeadLiveLlavaQwenForCausalLM as Model, VideoHeadLiveLlavaQwenConfig as Config
from models.tokenization_live import build_live_tokenizer_and_update_config
# This alias points to the very same module. Store its canonical model.vision_tower path once.
original_init=Model.__init__
def init_without_registered_alias(self,config):
 original_init(self,config)
 del self.vision_encoder
Model.__init__=init_without_registered_alias
base='/root/autodl-tmp/pretrained/llava-onevision-qwen2-7b-ov'
merged=root/'merged_llava_bf16_v2'; quant=root/'aha_llava_nf4'
t=time.time()
if not (merged/'config.json').exists():
 model,tok=build_video_head_live_llava_qwen(is_training=False,llm_pretrained=base,lora_pretrained='/root/autodl-tmp/pretrained/aha_weights',set_vision_inside=False,torch_dtype=torch.bfloat16,attn_implementation='eager',v_placeholder='<image>',frame_num_tokens=49,frame_resolution=384)
 model=model.merge_and_unload(safe_merge=True); model.eval()
 model.save_pretrained(merged,safe_serialization=True,max_shard_size='4GB'); tok.save_pretrained(merged)
 print('MERGED',time.time()-t,flush=True)
 del model; gc.collect(); torch.cuda.empty_cache()
if args.also_quantize and not (quant/'config.json').exists():
 qc=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16,llm_int8_skip_modules=['lm_head','informative_head','relevance_head','uncertainty_head','vision_tower','vision_encoder','mm_projector'])
 model=Model.from_pretrained(merged,config=Config.from_pretrained(merged),torch_dtype=torch.bfloat16,attn_implementation='eager',device_map='cuda',quantization_config=qc)
 model.eval(); model.currently_training=False
 model.save_pretrained(quant,safe_serialization=True,max_shard_size='4GB')
 tok=build_live_tokenizer_and_update_config(str(merged),model.config); tok.save_pretrained(quant)
 print('EXPORTED',time.time()-t,flush=True)
files=[]
exported=quant if args.also_quantize else merged
for p in sorted(exported.glob('*.safetensors')):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 files.append(dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest()))
assert files
r=dict(status='EXPORTED_RELOAD_PENDING',precision='nf4' if args.also_quantize else 'bf16',files=files,total_weight_bytes=sum(x['bytes'] for x in files),seconds=time.time()-t,upstream_commit='41473169504ae3793811066ffd398bc69094e499')
# File bytes describe storage only; eligibility is total parameters <=9B.
(root/'weight_manifest.json').write_text(json.dumps(r,indent=2)); print(json.dumps(r),flush=True)
