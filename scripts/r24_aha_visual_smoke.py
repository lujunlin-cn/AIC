"""Bounded real visual smoke. Failures are recorded; no installs/downloads."""
import os, sys, json, time, traceback, subprocess, hashlib
from pathlib import Path
os.environ.update(RANK='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled')
OUT=Path('/root/autodl-tmp/experiments/R24_EXEC_20261015'); start=time.time()
r={'status':'STARTED','scope':'public TRAIN visual smoke, no quality/competition eligibility claim','offline':True}
try:
    import torch
    import numpy as np
    from PIL import Image
    sys.path[:0]=['/root/autodl-tmp/aha_repo','/root/autodl-tmp/aha_repo/LLaVA_NeXT']
    from models.live_llava.video_head_live_llava_qwen import build_video_head_live_llava_qwen
    video='/root/autodl-tmp/experiments/R22_INSTRUMENT_20261015/videos/dXSlCw8AXv0.mp4'
    frames=[]
    for sec in [20,150]:
        raw=subprocess.check_output(['ffmpeg','-v','error','-ss',str(sec),'-i',video,'-frames:v','1','-vf','scale=384:384','-f','rawvideo','-pix_fmt','rgb24','pipe:1'])
        frames.append(Image.fromarray(np.frombuffer(raw,np.uint8).reshape(384,384,3)))
    r['frame_sha256']=[hashlib.sha256(f.tobytes()).hexdigest() for f in frames]
    model,tok=build_video_head_live_llava_qwen(is_training=False,
        llm_pretrained='/root/autodl-tmp/pretrained/llava-onevision-qwen2-7b-ov',
        lora_pretrained='/root/autodl-tmp/pretrained/aha_weights',
        set_vision_inside=True,torch_dtype=torch.float16,attn_implementation='eager',
        v_placeholder='<image>',frame_num_tokens=49,frame_resolution=384)
    model.eval(); model.currently_training=False
    tower=model.get_vision_tower()
    if not tower.is_loaded: tower.load_model()
    tower.to(device='cuda',dtype=torch.float16)
    prompt=tok('Assess the informativeness and highlight relevance of this video frame.',return_tensors='pt').input_ids.cuda()
    values=[]
    with torch.inference_mode():
        for f in [frames[0],frames[1],frames[0]]:
            pixel=tower.image_processor.preprocess([f],return_tensors='pt')['pixel_values'].cuda().half()
            # Exercise native vision encoder, connector and pooling; match placeholder count dynamically.
            visual=model.visual_embed(pixel)
            ids=torch.cat([prompt,torch.full((1,len(visual)),model.config.v_placeholder_id,device='cuda',dtype=torch.long)],1)
            tick=time.time()
            out=model(input_ids=ids,frames=pixel,return_dict=True,use_cache=False)
            torch.cuda.synchronize()
            v={key:getattr(out,key)[0,-1].float().cpu().tolist() for key in ['informative_logits','relevance_logits','uncertainty']}
            assert all(np.isfinite(np.asarray(x)).all() for x in v.values())
            v['seconds']=time.time()-tick; values.append(v)
    r.update(status='PASS',outputs=values,peak_vram_bytes=torch.cuda.max_memory_allocated(),actual_parameter_count=sum(p.numel() for p in model.parameters()),loaded_parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()))
except Exception as e:
    r.update(status='FAILED',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc())
finally:
    r['wall_s']=time.time()-start
    (OUT/'aha_smoke.json').write_text(json.dumps(r,indent=2))
    print(json.dumps(r,indent=2),flush=True)
