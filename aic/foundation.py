"""Frozen video representations with a shared deterministic clip/time protocol."""
import sys
import types
from pathlib import Path
import numpy as np
import torch
from torch import nn
from aic.video import iter_sampled_frames
from aic.inference import _normalise


def anchor_windows(n, clip_len, stride=4):
    if n < 1: raise ValueError('empty input')
    anchors = np.arange(0, n, stride, dtype=np.int64)
    # Every model predicts at identical anchors; context lengths are explicit.
    offsets = np.arange(clip_len) - clip_len//2
    return anchors, np.clip(anchors[:,None]+offsets[None,:], 0, n-1)


def load_encoder(kind, root, device):
    root=Path(root)
    if kind == 'deit':
        import timm
        m=timm.create_model('deit_small_patch16_224',pretrained=False,num_classes=0)
        m.load_state_dict(torch.load(root/'deit_small_patch16_224_headless.pt',map_location='cpu',weights_only=True))
        return m.to(device).eval(),1
    if kind in ('videomae','videomae_large'):
        from transformers import AutoModel
        model_dir=root/('videomae_large' if kind=='videomae_large' else 'videomaev2')
        return AutoModel.from_pretrained(str(model_dir),trust_remote_code=True,local_files_only=True).to(device).eval(),16
    if kind not in ('internvideo','internvideo_stage2'): raise ValueError(kind)
    src=Path('/data/aic/tmp/InternVideo2/llava-train_videochat/llava/model/multimodal_encoder/internvideo2')
    pkg=types.ModuleType('aic_iv2');pkg.__path__=[str(src)];sys.modules[pkg.__name__]=pkg
    stub=types.ModuleType('aic_iv2.flash_attention_class')
    class Disabled(nn.Module):
        def __init__(self,*a,**kw):raise RuntimeError('FlashAttention forbidden in V100 adapter')
    stub.FlashAttention=Disabled;sys.modules[stub.__name__]=stub
    from aic_iv2.vit_scale_clean import PretrainVisionTransformer_clean
    stage2=kind=='internvideo_stage2'
    m=PretrainVisionTransformer_clean(in_chans=3,img_size=224,patch_size=14,embed_dim=1408,depth=40,num_heads=16,mlp_ratio=48/11,qkv_bias=False,drop_path_rate=.25,init_values=1e-5,qk_normalization=True,use_flash_attn=False,use_fused_rmsnorm=False,use_fused_mlp=False,attn_pool_num_heads=16,clip_embed_dim=768 if stage2 else None,layerscale_no_force_fp32=False,num_frames=4 if stage2 else 8,tubelet_size=1,sep_pos_embed=False,sep_image_video_pos_embed=False,use_checkpoint=False,checkpoint_num=0,x_vis_return_idx=-1,x_vis_only=False)
    if stage2:
        raw=torch.load(root/'internvideo2_stage2_1b/InternVideo2-stage2_1b-224p-f4.pt',map_location='cpu',weights_only=False)['module']
        state={k.removeprefix('vision_encoder.'):v for k,v in raw.items() if k.startswith('vision_encoder.')}
    else:
        state=torch.load(root/'internvideo2_stage1_1b/1B_ft_k710_ft_k700_f8.pth',map_location='cpu',weights_only=True)['module']
    state={k.replace('.ls1.gamma','.ls1.weight').replace('.ls2.gamma','.ls2.weight'):v for k,v in state.items()}
    missing,extra=m.load_state_dict(state,strict=False)
    allowed_extra={'fc_norm.weight','fc_norm.bias','head.weight','head.bias'}
    if stage2:
        # Stage2 checkpoints also carry clip position/decoder heads that are
        # irrelevant for frame-level representation extraction.
        allowed_extra |= {k for k in extra if k.startswith(('img_pos_embed','clip_pos_embed','clip_img_pos_embed','clip_decoder.','final_clip_decoder.'))}
    if missing or set(extra)-allowed_extra:raise ValueError((missing,extra))
    norm=None if stage2 else nn.LayerNorm(768,eps=1e-6)
    if norm is not None: norm.load_state_dict({k.removeprefix('fc_norm.'):v for k,v in state.items() if k.startswith('fc_norm.')})
    class Encoder(nn.Module):
        def __init__(self):super().__init__();self.encoder=m;self.norm=norm
        def forward(self,x):
            y=self.encoder(x)
            return y[1] if self.norm is None else self.norm(y[1])
    return Encoder().half().to(device).eval(),4 if stage2 else 8


@torch.inference_mode()
def encode_video(model, kind, path, device, batch=4):
    samples=list(iter_sampled_frames(path,sample_fps=2,size=224))
    n=len(samples);length={'deit':1,'videomae':16,'videomae_large':16,'internvideo':8,'internvideo_stage2':4}[kind]
    anchors,win=anchor_windows(n,length)
    images=_normalise([r[2] for r in samples])
    features=[]
    for start in range(0,len(anchors),batch):
        if kind=='deit':x=images[anchors[start:start+batch]]
        else:
            clips=images[win[start:start+batch]]
            # HuggingFace VideoMAE expects [B,T,C,H,W]; the custom
            # VideoMAEv2 adapter expects [B,C,T,H,W].
            x=clips if kind=='videomae_large' else clips.permute(0,2,1,3,4)
        x=x.to(device=device,dtype=next(model.parameters()).dtype)
        out=model(x)
        if hasattr(out,'last_hidden_state'):
            # Standard Transformers VideoMAE returns token features; use the
            # mean token embedding as the clip representation.
            y=out.last_hidden_state.mean(dim=1).float().cpu()
        else:
            y=out.float().cpu()
        if not torch.isfinite(y).all():raise FloatingPointError('encoder output not finite')
        features.append(y.numpy())
    return np.concatenate(features),np.asarray([samples[i][1] for i in anchors]),np.asarray([samples[i][0] for i in anchors])
