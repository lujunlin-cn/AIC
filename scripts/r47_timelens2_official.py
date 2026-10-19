#!/usr/bin/env python3
"""Run one fixed TimeLens2 generic-highlight prompt over the 426 test videos.

The output is raw interval text only.  A separate converter creates a
submission candidate, so a stopped inference can resume without changing the
prompt or silently mixing decoder settings.
"""
from __future__ import annotations
import argparse, json, time, gc
from pathlib import Path
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

PROMPT = (
    "This is a generic video with no user query. Identify every time interval "
    "that is most worth keeping in a short highlight edit. Prefer a concrete "
    "action, outcome, reaction, reveal, or visually informative change over "
    "waiting, setup, or static repetition. Return ALL intervals in seconds "
    "as a JSON array of [start, end] pairs and no other text."
)

def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",type=Path,required=True); ap.add_argument("--video-root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True); ap.add_argument("--start",type=int,default=0); ap.add_argument("--end",type=int,default=426)
    ap.add_argument("--fps",type=float,default=2.0)
    ap.add_argument("--max-pixels",type=int,default=320*320)
    ap.add_argument("--max-new-tokens",type=int,default=256)
    ap.add_argument("--resume",action="store_true"); args=ap.parse_args()
    done={}
    if args.resume and args.output.exists():
        old=json.loads(args.output.read_text()); done={str(x["video_id"]):x for x in old.get("results",[])}
    model=AutoModelForImageTextToText.from_pretrained(str(args.model),torch_dtype=torch.float16,device_map="auto",attn_implementation="sdpa")
    processor=AutoProcessor.from_pretrained(str(args.model))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for vid in range(args.start,args.end):
        key=str(vid)
        if key in done: continue
        path=args.video_root/f"{vid}.mp4"
        if not path.exists():
            done[key]={"video_id":key,"error":"missing_video"}; continue
        messages=[{"role":"user","content":[
            # Keep the ASCII symlink path in the URI.  ``Path.resolve()``
            # expands the mount's Chinese directory name and decord on this
            # stack percent-encodes it before opening, which fails.
            {"type":"video","video":path.as_uri(),"fps":args.fps,"min_pixels":224*224,"max_pixels":args.max_pixels,"total_pixels":64*args.max_pixels},
            {"type":"text","text":PROMPT}]}]
        text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        images,videos,kwargs=process_vision_info(messages,image_patch_size=16,return_video_kwargs=True,return_video_metadata=True)
        if videos is not None:
            videos,metas=zip(*videos); videos,metas=list(videos),list(metas)
        else: metas=None
        inputs=processor(text=text,images=images,videos=videos,video_metadata=metas,do_resize=False,return_tensors="pt",cap_pixels_per_frame=True,**kwargs)
        inputs={k:(v.to(model.device) if hasattr(v,"to") else v) for k,v in inputs.items()}
        with torch.inference_mode():
            out=model.generate(**inputs,max_new_tokens=args.max_new_tokens,temperature=0.01,top_p=0.001,top_k=1,repetition_penalty=1.0)
        plen=inputs["input_ids"].shape[1]
        response=processor.batch_decode(out[:,plen:],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
        done[key]={"video_id":key,"response":response}
        args.output.write_text(json.dumps({"model":str(args.model),"prompt":PROMPT,"fps":args.fps,"max_pixels":args.max_pixels,"results":[done[k] for k in sorted(done,key=lambda x:int(x))]},ensure_ascii=False,indent=2)+"\n")
        # Release per-video tensors and allocator blocks before the next
        # independent decode. Without this, a long run can accumulate cached
        # video tensors and die after a handful of videos.
        del inputs, out, videos, images, kwargs, metas
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(json.dumps({"video_id":key,"response":response},ensure_ascii=False),flush=True)
    args.output.write_text(json.dumps({"model":str(args.model),"prompt":PROMPT,"fps":args.fps,"max_pixels":args.max_pixels,"results":[done[k] for k in sorted(done,key=lambda x:int(x))]},ensure_ascii=False,indent=2)+"\n")

if __name__=="__main__": main()
