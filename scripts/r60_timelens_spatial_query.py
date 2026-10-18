#!/usr/bin/env python3
"""Run a deliberately aggressive Qwen3-VL/TimeLens crop query.

The model is already the deployed 8B temporal component.  This asks the same
checkpoint for one target-ratio crop per video.  It is intentionally a high
risk probe: responses are cached verbatim and malformed outputs are retained
for audit rather than silently converted into boxes.
"""
from __future__ import annotations
import argparse, gc, json, re
from pathlib import Path
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--video-root',type=Path,required=True); ap.add_argument('--index',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True); ap.add_argument('--start',type=int,default=0); ap.add_argument('--end',type=int,default=426)
    ap.add_argument('--fps',type=float,default=1.0); ap.add_argument('--max-pixels',type=int,default=50176); ap.add_argument('--max-new-tokens',type=int,default=128)
    args=ap.parse_args()
    idx={str(json.loads(x)['video_id']):json.loads(x) for x in args.index.read_text().splitlines() if x.strip()}
    done={}
    if args.output.exists():
        old=json.loads(args.output.read_text()); done={str(x['video_id']):x for x in old.get('results',[])}
    model=AutoModelForImageTextToText.from_pretrained(str(args.model),torch_dtype=torch.float16,device_map='auto',attn_implementation='sdpa')
    proc=AutoProcessor.from_pretrained(str(args.model))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for vi in range(args.start,args.end):
        vid=str(vi)
        if vid in done: continue
        meta=idx[vid]; rw,rh=meta['targetRatioWH']; path=args.video_root/f'{vid}.mp4'
        prompt=(f'Analyze this complete generic video. The desired output crop aspect ratio is {rw}:{rh}. '
                 'Find the main subject or interacting subjects during the most valuable highlight event. '
                 'Return exactly one JSON object and no explanation: '
                 '{"crop":[x,y,w]} where x,y,w are normalized integers from 0 to 1000, '
                 'x,y are the top-left of a crop and w is its width; the crop height is implied by the requested ratio. '
                 'Keep the whole important subject/action and avoid empty background.')
        messages=[{'role':'user','content':[{'type':'video','video':path.as_uri(),'fps':args.fps,'min_pixels':224*224,'max_pixels':args.max_pixels,'total_pixels':64*args.max_pixels},{'type':'text','text':prompt}]}]
        text=proc.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        images,videos,kwargs=process_vision_info(messages,image_patch_size=16,return_video_kwargs=True,return_video_metadata=True)
        if videos is not None:
            videos,metas=zip(*videos); videos,metas=list(videos),list(metas)
        else: metas=None
        inputs=proc(text=text,images=images,videos=videos,video_metadata=metas,do_resize=False,return_tensors='pt',cap_pixels_per_frame=True,**kwargs)
        inputs={k:(v.to(model.device) if hasattr(v,'to') else v) for k,v in inputs.items()}
        with torch.inference_mode(): out=model.generate(**inputs,max_new_tokens=args.max_new_tokens,temperature=0.01,top_p=0.001,top_k=1,repetition_penalty=1.0)
        plen=inputs['input_ids'].shape[1]; response=proc.batch_decode(out[:,plen:],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
        done[vid]={'video_id':vid,'targetRatioWH':[rw,rh],'response':response}
        args.output.write_text(json.dumps({'model':str(args.model),'fps':args.fps,'max_pixels':args.max_pixels,'prompt':prompt,'results':[done[k] for k in sorted(done,key=lambda z:int(z))]},ensure_ascii=False,indent=2)+'\n')
        del inputs,out,videos,images,kwargs,metas; gc.collect(); torch.cuda.empty_cache()
        print(json.dumps(done[vid],ensure_ascii=False),flush=True)
    args.output.write_text(json.dumps({'model':str(args.model),'fps':args.fps,'max_pixels':args.max_pixels,'prompt':prompt if 'prompt' in locals() else '','results':[done[k] for k in sorted(done,key=lambda z:int(z))]},ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__': main()
