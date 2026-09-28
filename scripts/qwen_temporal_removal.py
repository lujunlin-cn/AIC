"""T3: Qwen3-VL-32B removal-only temporal judgement; the spatial output stays N0.

The old teacher (aic/qwen_teacher.py, SUB_Q 31.20) asked for highlight
segments to KEEP, allowed "no highlight at all" (10 empty videos) and kept 61%
of frames on average.  This probe inverts the default: every frame is kept
unless the teacher marks a 1 s cell as clearly worthless for a highlight clip.

  cells     cell k = frames with display time in [k, k+1) s; its anchor is the
            decoded frame nearest k + .5 s (centre anchor, temporal contract v1)
  stage 1   overview: <=16 anchors uniformly over the whole video -> summary
  stage 2   blocks of 8 target anchors + <=2 context anchors on each side, with
            the stage-1 summary; the reply lists target ids to REMOVE
  guard     if the removed cells cover > 50% of the frames, keep everything
            (pre-declared, not tuned: removal-only assumes low-value content is
            a minority; SUB_Q evidence says over-removal is the known failure)

  --extract-only   cv env (PyAV 15): anchors -> <frames-dir>/<vid>/<cell>.png + meta.json
  otherwise        vLLM env: reads PNGs, writes <output>/<vid>.json
"""
import argparse,json,re,time
from pathlib import Path
import numpy as np

SYSTEM=("You are a professional video editor cutting a short highlight clip from a video. "
        "By default every moment stays in the clip; you only cut moments that clearly add nothing.")
OVERVIEW=("These {n} frames are sampled uniformly across a whole {dur:.1f}s video, each labelled with its time. "
          "Describe in one or two sentences what the video shows and what its main event or highlight is. "
          "Return ONLY JSON: {{\"summary\":\"...\"}}")
JUDGE=("Video summary: {summary}\n"
       "The frames below are consecutive 1-second moments of this {dur:.1f}s video, labelled [id=K t=Ts]. "
       "Frames marked CONTEXT are only there to show what comes before and after; judge only the TARGET frames {lo}..{hi}. "
       "The default is KEEP. Mark a TARGET frame REMOVE only if that moment clearly contributes nothing to a highlight clip of this video: "
       "black, blank or fading frames; title cards, credits, logos or end screens unrelated to the event; frozen or transition frames; "
       "content unrelated to the main event. If in doubt, keep it. It is normal to remove nothing. "
       "Return ONLY JSON: {{\"remove\":[ids],\"reason\":\"short\"}}")
_JSON=re.compile(r"\{.*\}",re.DOTALL)
BLOCK,CTX,OVERVIEW_N,GUARD=8,2,16,.5


def cells_and_anchors(times):
 times=np.asarray(times,float);cell=np.floor(times+1e-9).astype(int);anchors=[]
 for k in np.unique(cell):
  idx=np.flatnonzero(cell==k);anchors.append({'cell':int(k),'frame':int(idx[np.argmin(np.abs(times[idx]-(k+.5)))]),'t':float(k+.5)})
 return cell,anchors


def parse_remove(text,targets):
 m=_JSON.search(text or '')
 if not m:return [],'no_json'
 try:d=json.loads(m.group(0))
 except ValueError:return [],'bad_json'
 r=d.get('remove',[])
 if not isinstance(r,list):return [],'bad_list'
 ok=[];rep=False
 for x in r:
  try:x=int(x)
  except (TypeError,ValueError):rep=True;continue
  if x in targets:ok.append(x)
  else:rep=True
 return sorted(set(ok)),'repaired' if rep else 'ok'


def keep_frames(frame_cell,removed_cells,guard=GUARD):
 """Frames kept by the removal rule; keep-all when the guard trips."""
 fc=np.asarray(frame_cell);drop=np.isin(fc,list(removed_cells))
 if drop.mean()>guard or drop.all():return np.arange(len(fc)),True
 return np.flatnonzero(~drop),False


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--index');ap.add_argument('--records',help='JSONL with video_id, video_path')
 ap.add_argument('--frames-dir',type=Path,required=True);ap.add_argument('--output',type=Path);ap.add_argument('--model')
 ap.add_argument('--size',type=int,default=448);ap.add_argument('--tp',type=int,default=4);ap.add_argument('--chunk',type=int,default=16)
 ap.add_argument('--part',type=int,default=0);ap.add_argument('--parts',type=int,default=1);ap.add_argument('--extract-only',action='store_true');a=ap.parse_args()
 if a.index:
  from aic.inference import _record_path
  jobs=[(r['video_id'],_record_path(r,None)) for r in map(json.loads,open(a.index))]
 else:jobs=[(r['video_id'],r['video_path']) for r in map(json.loads,open(a.records))]
 jobs=jobs[a.part::a.parts]
 if a.extract_only:
  import cv2
  from aic.video import _decoded
  for vid,path in jobs:
   d=a.frames_dir/vid
   if (d/'meta.json').exists():continue
   d.mkdir(parents=True,exist_ok=True);times=[]
   try:
    for st,f in _decoded(path):times.append(st.time_seconds)
   except ValueError as e:print('SKIP',vid,e,flush=True);continue
   cell,anc=cells_and_anchors(times);want={x['frame']:x['cell'] for x in anc}
   for st,f in _decoded(path):
    if st.index in want:
     rgb=f.to_ndarray(format='rgb24');h,w=rgb.shape[:2];sc=min(1.,a.size/max(w,h))
     img=cv2.resize(rgb,(round(w*sc),round(h*sc)),interpolation=cv2.INTER_AREA) if sc<1 else rgb
     cv2.imwrite(str(d/f"{want[st.index]}.png"),cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
   (d/'meta.json').write_text(json.dumps({'n_frames':len(times),'duration':float(times[-1]) if times else 0.,'frame_cell':cell.tolist(),'anchors':anc})+'\n')
   print('EXTRACTED',vid,len(times),len(anc),flush=True)
  return
 from PIL import Image
 from vllm import LLM,SamplingParams
 llm=LLM(model=a.model,dtype='float16',tensor_parallel_size=a.tp,gpu_memory_utilization=.85,max_model_len=16384,max_num_seqs=16,limit_mm_per_prompt={'image':OVERVIEW_N,'video':0},seed=0)
 p_over=SamplingParams(temperature=0.0,max_tokens=128,seed=0);p_judge=SamplingParams(temperature=0.0,max_tokens=96,seed=0)
 a.output.mkdir(parents=True,exist_ok=True);start=time.time()
 todo=[(v,p) for v,p in jobs if not (a.output/f'{v}.json').exists()]
 for c0 in range(0,len(todo),a.chunk):
  chunk=todo[c0:c0+a.chunk];metas={};imgs={}
  for vid,_ in chunk:
   metas[vid]=json.loads((a.frames_dir/vid/'meta.json').read_text());imgs[vid]={x['cell']:Image.open(a.frames_dir/vid/f"{x['cell']}.png").convert('RGB') for x in metas[vid]['anchors']}
  def content(vid,cells,tags):
   out=[]
   for k,tag in zip(cells,tags):out+=[{'type':'text','text':f'[id={k} t={k+.5:.1f}s{tag}]'},{'type':'image_pil','image_pil':imgs[vid][k]}]
   return out
  conv=[]
  for vid,_ in chunk:
   m=metas[vid];cs=[x['cell'] for x in m['anchors']];sel=[cs[i] for i in sorted({int(round(j)) for j in np.linspace(0,len(cs)-1,min(OVERVIEW_N,len(cs)))})]
   conv.append([{'role':'system','content':SYSTEM},{'role':'user','content':content(vid,sel,['']*len(sel))+[{'type':'text','text':OVERVIEW.format(n=len(sel),dur=m['duration'])}]}])
  over=[o.outputs[0].text for o in llm.chat(conv,p_over,use_tqdm=False)];summ={}
  for (vid,_),t in zip(chunk,over):
   mm=_JSON.search(t or '')
   try:summ[vid]=str(json.loads(mm.group(0)).get('summary',''))[:400] if mm else ''
   except ValueError:summ[vid]=''
   if not summ[vid]:summ[vid]=(t or '').strip()[:400] or 'unknown'
  conv=[];blocks=[]
  for vid,_ in chunk:
   m=metas[vid];cs=[x['cell'] for x in m['anchors']]
   for b in range(0,len(cs),BLOCK):
    tg=cs[b:b+BLOCK];pre=cs[max(0,b-CTX):b];post=cs[b+BLOCK:b+BLOCK+CTX];allc=pre+tg+post
    tags=[' CONTEXT' if k in pre or k in post else ' TARGET' for k in allc]
    conv.append([{'role':'system','content':SYSTEM},{'role':'user','content':content(vid,allc,tags)+[{'type':'text','text':JUDGE.format(summary=summ[vid],dur=m['duration'],lo=tg[0],hi=tg[-1])}]}])
    blocks.append((vid,tg,pre+post))
  replies=[o.outputs[0].text for o in llm.chat(conv,p_judge,use_tqdm=False)]
  per={vid:[] for vid,_ in chunk}
  for (vid,tg,ctx),t in zip(blocks,replies):
   rm,st=parse_remove(t,set(tg));per[vid].append({'targets':tg,'context':ctx,'raw':t,'removed':rm,'status':st})
  for (vid,_),ov in zip(chunk,over):
   m=metas[vid];rc=sorted({k for b in per[vid] for k in b['removed']});kept,guard=keep_frames(m['frame_cell'],rc)
   out={'video_id':vid,'n_frames':m['n_frames'],'duration':m['duration'],'overview_raw':ov,'summary':summ[vid],'blocks':per[vid],'removed_cells':rc,
    'guard_tripped':guard,'kept_frames':len(kept),'keep_rate':len(kept)/max(1,m['n_frames']),'rule':{'block':BLOCK,'context':CTX,'overview':OVERVIEW_N,'guard':GUARD}}
   (a.output/f'{vid}.json').write_text(json.dumps(out)+'\n')
  print(json.dumps({'done':c0+len(chunk),'of':len(todo),'blocks':len(blocks),'removed_cells':sum(len(b['removed']) for v in per.values() for b in v),'wall':round(time.time()-start,1)}),flush=True)
if __name__=='__main__':main()
