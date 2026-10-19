#!/usr/bin/env python3
"""Minimal R87 finite-value trace for one PHD2 fragment."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np

def frames(paths):
    from PIL import Image
    out=[]
    for p in paths:
        im=Image.open(p).convert('RGB'); s=384; w,h=im.size; sc=min(s/w,s/h)
        nw,nh=max(1,round(w*sc)),max(1,round(h*sc)); im=im.resize((nw,nh))
        c=Image.new('RGB',(s,s),(0,0,0)); c.paste(im,((s-nw)//2,(s-nh)//2)); out.append(np.asarray(c).transpose(2,0,1))
    import torch
    return torch.from_numpy(np.stack(out)).contiguous()

def stat(name,x):
    import torch
    if x is None:return {'name':name,'none':True}
    if not torch.is_tensor(x):return {'name':name,'type':type(x).__name__}
    y=x.detach().float(); return {'name':name,'shape':list(x.shape),'finite':bool(torch.isfinite(y).all()),'nan':int(torch.isnan(y).sum()),'inf':int(torch.isinf(y).sum()),'min':float(torch.nan_to_num(y,nan=0.0).min()),'max':float(torch.nan_to_num(y,nan=0.0).max())}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['cache','independent'],default='cache');ap.add_argument('--transition',choices=['eos','empty'],default='eos');ap.add_argument('--vid',default='_XUcH_2xFy0_f00_r');ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    import torch
    import os
    os.environ.setdefault('RANK','0');os.environ.setdefault('WORLD_SIZE','1');os.environ.setdefault('LOCAL_RANK','0')
    from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower,SigLipVisionModel,SigLipVisionConfig
    def local_load(self,device_map=None):
        if self.is_loaded:return
        self.vision_tower=SigLipVisionModel(SigLipVisionConfig());del self.vision_tower.vision_model.encoder.layers[-1:];self.vision_tower.vision_model.head=torch.nn.Identity();self.vision_tower.requires_grad_(False);self.is_loaded=True
    SigLipVisionTower.load_model=local_load
    from models.live_llava.video_head_live_llava_qwen import build_video_head_live_llava_qwen
    from test.sink_cache import SinkCache
    base='/root/autodl-tmp/pretrained/llava-onevision-qwen2-7b-ov';adapter='/root/autodl-tmp/pretrained/aha_weights'
    model,tok=build_video_head_live_llava_qwen(is_training=False,llm_pretrained=base,lora_pretrained=adapter,vision_pretrained='google/siglip-large-patch16-384',frame_resolution=384,frame_num_tokens=49,frame_token_cls=False,frame_token_pooled=[7,7],video_pooling_stride=4,attn_implementation='eager',torch_dtype=torch.bfloat16,set_vision_inside=False,quantization=False)
    model=model.to('cuda').eval();vt=model.get_vision_tower();proc=vt.image_processor
    pool=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/frames')/a.vid;fps=sorted(pool.glob('*.jpg'),key=lambda p:float(p.stem))[:8];x=frames(fps)
    with torch.no_grad():
        pv=proc.preprocess([z.permute(1,2,0).numpy() for z in x],return_tensors='pt')['pixel_values'].to('cuda',dtype=torch.bfloat16)
        embeds=model.visual_embed(pv).split(model.config.frame_num_tokens)
    start=tok.apply_chat_template([{'role':'system','content':'A multimodal AI assistant is helping users with some activities. Below is their conversation, interleaved with the list of video frames received by the assistant.'}],return_tensors='pt').to('cuda')
    hidden=model.config.hidden_size; past=SinkCache(window_length=2048,num_sink_tokens=32) if a.mode=='cache' else None; rec={'mode':a.mode,'vid':a.vid,'processor':stat('pixel_values',pv),'visual_embeds':stat('visual_embed',torch.cat(embeds)),'frames':[]}
    for i,fe in enumerate(embeds):
        if a.mode=='independent' or i==0: ids=start
        elif a.transition=='empty': ids=torch.empty((1,0),device='cuda',dtype=torch.long)
        else: ids=torch.tensor([[tok.eos_token_id]],device='cuda',dtype=torch.long)
        ie=model.get_input_embeddings()(ids).view(1,-1,hidden); inp=torch.cat([ie,fe.view(1,-1,hidden)],1)
        with torch.no_grad():out=model(inputs_embeds=inp,use_cache=True,past_key_values=past,output_hidden_states=True,return_dict=True)
        if a.mode=='cache':past=out.past_key_values
        h=out.hidden_states[-1] if out.hidden_states else None
        rec['frames'].append({'i':i,'input':stat('input',inp),'hidden':stat('hidden',h),'relevance':stat('relevance',out.relevance_logits),'informative':stat('informative',out.informative_logits),'uncertainty':stat('uncertainty',out.uncertainty),'past_type':type(past).__name__ if past is not None else None,'past_len':int(past.get_seq_length()) if past is not None and hasattr(past,'get_seq_length') else None})
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(rec,indent=1));print(json.dumps(rec,indent=1))
if __name__=='__main__':main()
