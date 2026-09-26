#!/usr/bin/env python3
"""Bounded compatibility probe; no training and no official-test access.
Requires checked-out upstream InternVideo2 source (recorded below) and a minimal
adapter package: flash attention import is optional, and flags disable fused ops.
"""
import argparse, hashlib, json, time, sys, types
from pathlib import Path
import cv2
import numpy as np
import torch

p=argparse.ArgumentParser(); p.add_argument('--source-root',default='/data/aic/tmp/InternVideo2'); p.add_argument('--frames',type=int,default=8); p.add_argument('--output',required=True); a=p.parse_args()
src=Path(a.source_root)/'llava-train_videochat/llava/model/multimodal_encoder/internvideo2'
# Isolated import avoids unrelated LLaVA language-model and FlashAttention imports.
pkg=types.ModuleType('aic_iv2_probe'); pkg.__path__=[str(src)]; sys.modules[pkg.__name__]=pkg
stub=types.ModuleType('aic_iv2_probe.flash_attention_class')
class DisabledFlashAttention(torch.nn.Module):
 def __init__(self,*args,**kwargs): raise RuntimeError('FlashAttention is disabled in this V100 probe')
stub.FlashAttention=DisabledFlashAttention; sys.modules[stub.__name__]=stub
from aic_iv2_probe.vit_scale_clean import PretrainVisionTransformer_clean, interpolate_pos_embed_internvideo2
ck=Path('/data/aic/pretrained/internvideo2_stage1_1b/1B_ft_k710_ft_k700_f8.pth')
manifest=Path('/data/aic/experiments/QVH_DEITS_QUICK_5PCT/train_features.jsonl')
row=json.loads(manifest.read_text().splitlines()[0]); assert row['split']=='train'
torch.set_num_threads(4)
s=torch.load(ck,map_location='cpu')['module']
# Upstream clean encoder renamed LayerScale gamma to weight. Value unchanged.
s={k.replace('.ls1.gamma','.ls1.weight').replace('.ls2.gamma','.ls2.weight'):v for k,v in s.items()}
m=PretrainVisionTransformer_clean(in_chans=3,img_size=224,patch_size=14,embed_dim=1408,depth=40,num_heads=16,mlp_ratio=48/11,qkv_bias=False,drop_path_rate=0.25,init_values=1e-5,qk_normalization=True,use_flash_attn=False,use_fused_rmsnorm=False,use_fused_mlp=False,attn_pool_num_heads=16,layerscale_no_force_fp32=False,num_frames=a.frames,tubelet_size=1,sep_pos_embed=False,sep_image_video_pos_embed=False,use_checkpoint=False,checkpoint_num=0,x_vis_return_idx=-1,x_vis_only=False)
interpolate_pos_embed_internvideo2(s,m,orig_t_size=8)
missing,extra=m.load_state_dict(s,strict=False)
assert missing==[],missing
assert set(extra)=={'fc_norm.weight','fc_norm.bias','head.weight','head.bias'},extra
fc_norm=torch.nn.LayerNorm(768,eps=1e-6)
fc_norm.load_state_dict({k.removeprefix('fc_norm.'):v for k,v in s.items() if k.startswith('fc_norm.')})
cap=cv2.VideoCapture(row['video_path']); fps=cap.get(cv2.CAP_PROP_FPS); frames=[]
for i in range(a.frames):
 cap.set(cv2.CAP_PROP_POS_FRAMES, round(i*fps/2)); ok,frame=cap.read()
 if not ok: raise ValueError('decode failed')
 frames.append(cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),(224,224),interpolation=cv2.INTER_CUBIC))
cap.release()
x=torch.tensor(np.stack(frames),dtype=torch.float32).permute(3,0,1,2).unsqueeze(0)/255
mean=torch.tensor([.485,.456,.406]).view(1,3,1,1,1); std=torch.tensor([.229,.224,.225]).view(1,3,1,1,1); x=(x-mean)/std
m=m.eval().half().cuda(); fc_norm=fc_norm.eval().half().cuda(); x=x.half().cuda()
torch.cuda.reset_peak_memory_stats(); times=[]
with torch.inference_mode():
 y=fc_norm(m(x)[1]); torch.cuda.synchronize()
 for _ in range(3):
  st=time.perf_counter(); y=fc_norm(m(x)[1]); torch.cuda.synchronize(); times.append(time.perf_counter()-st)
assert torch.isfinite(y).all()
h=hashlib.sha256()
with ck.open('rb') as f:
 for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
result={'run_id':f'INTERNVIDEO2_SMOKE_{a.frames}F_20260926','model':'InternVideo2_1B_k710_k700_f8','upstream_repo':'https://github.com/OpenGVLab/InternVideo2','upstream_commit':'3d521087215c9024199b0512370a29dee1c0fee6','weight_path':str(ck),'weight_bytes':ck.stat().st_size,'sha256':h.hexdigest(),'encoder_parameter_count':sum(p.numel() for p in m.parameters())+sum(p.numel() for p in fc_norm.parameters()),'precision':'FP16','input_shape':list(x.shape),'output_shape':list(y.shape),'finite':bool(torch.isfinite(y).all()),'mean_latency_s':float(np.mean(times)),'median_latency_s':float(np.median(times)),'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,'source_split':'QVHighlights_training','video_path':row['video_path'],'sampling_fps':2,'preprocessing':'bicubic_resize_224_ImageNet_mean_std','loaded_missing_keys':missing,'omitted_classification_keys':['head.weight','head.bias'],'fc_norm_loaded':True,'adapter':['LayerScale gamma renamed to weight','flash import stub unreachable because use_flash_attn=False','standard PyTorch attention and MLP','temporal position interpolation 8 to frames if different'],'device':torch.cuda.get_device_name(),'official_f_video':None,'competition_score':None,'status':'compatibility_smoke_only_not_candidate'}
Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
