"""V4 E3: segment-level event roles from Qwen3-VL-32B; the spatial output stays DENSE.

TEMP (scripts/qwen_temporal_removal.py) judges each 1 s cell only against
"clearly worthless" content (black, titles, unrelated) with 2 context cells.
E3 adds event structure: consecutive 2 s segments (cells 2m, 2m+1; the same
TEMP anchors at +.5 s, 448 px) are shown as a timeline of 12 TARGET segments
+ 1 CONTEXT segment on each side, with TEMP's stage-1 video summary, and every
TARGET segment gets one role:
  MAIN / SETUP / RESULT  -> keep  (restored if TEMP had removed a cell)
  IDLE / JUNK            -> drop
  UNSURE / unparsable    -> TEMP decision for its cells
Guard (as TEMP): if the new mask drops > 50% of the frames, the video keeps
the TEMP mask.  Output <output>/<vid>.json with roles per segment, the new
removed cell list and the diff vs TEMP.
"""
import argparse,json,re,time
from pathlib import Path
import numpy as np
from scripts.qwen_temporal_removal import keep_frames,GUARD

SYSTEM=("You are a professional video editor cutting a short highlight clip from a video. "
        "You first understand the event the video is about, then decide which moments the highlight clip needs.")
ROLES_PROMPT=("Video summary: {summary}\n"
 "The images cover {lo:.1f}-{hi:.1f}s of this {dur:.1f}s video as consecutive 2-second segments; each segment is shown by up to two frames "
 "labelled [seg=K t=A-Bs]. Segments marked CONTEXT only show what comes before and after; label only the TARGET segments {ids}.\n"
 "For each TARGET segment choose its role in the event:\n"
 "MAIN = the main event or highlight action is happening, the moment a viewer is meant to watch;\n"
 "SETUP = build-up needed to understand or enjoy the main event (approach, run-up, the moment right before it);\n"
 "RESULT = the outcome or reaction right after the main event (landing, catch, score, celebration);\n"
 "IDLE = nothing of the event is happening: waiting, wandering, repeated or aimless footage with no progress towards or from the main event;\n"
 "JUNK = black, blank or fading frames, title cards, credits, logos, end screens, frozen or transition frames, or content unrelated to the main event;\n"
 "UNSURE = you cannot tell.\n"
 "A quiet or static moment is not IDLE if it belongs to the event (a pause before a jump, a pose, a person speaking to the camera). "
 "A moment without people is not JUNK if it shows the event or its setting in use.\n"
 "Return ONLY JSON: {{\"roles\":{{\"K\":\"ROLE\"}}}} with one entry for every TARGET segment.")
KEEP_R={'MAIN','SETUP','RESULT'};DROP_R={'IDLE','JUNK'};ALL_R=KEEP_R|DROP_R|{'UNSURE'}
WIN,CTX_SEG=12,1
_JSON=re.compile(r"\{.*\}",re.DOTALL)


def segments(cells):
 seg={}
 for c in cells:seg.setdefault(c//2,[]).append(c)
 return seg


def parse_roles(text,targets):
 m=_JSON.search(text or '')
 if not m:return {t:'UNSURE' for t in targets},'no_json'
 try:d=json.loads(m.group(0))
 except ValueError:return {t:'UNSURE' for t in targets},'bad_json'
 r=d.get('roles',d)
 if not isinstance(r,dict):return {t:'UNSURE' for t in targets},'bad_roles'
 out={};rep=False
 for t in targets:
  v=r.get(str(t),r.get(t))
  v=str(v).strip().upper() if v is not None else None
  if v not in ALL_R:rep=True;v='UNSURE'
  out[t]=v
 if set(map(str,r))-set(map(str,targets)):rep=True
 return out,'repaired' if rep else 'ok'


def event_mask(frame_cell,temp_removed,roles,seg_cells,guard=GUARD):
 """Removed cells after applying segment roles on top of TEMP's effective mask
 (a TEMP video whose own guard tripped keeps everything, so its default is no
 removal); if the result trips the guard, the video keeps TEMP's effective mask."""
 base=set() if keep_frames(frame_cell,sorted(set(temp_removed)),guard)[1] else set(temp_removed);rem=set(base)
 for s,role in roles.items():
  if role in DROP_R:rem|=set(seg_cells[s])
  elif role in KEEP_R:rem-=set(seg_cells[s])
 kept,tripped=keep_frames(frame_cell,sorted(rem),guard)
 if tripped:return sorted(base),True
 return sorted(rem),False


def drop_only_mask(frame_cell,temp_removed,roles,seg_cells,guard=GUARD):
 """E3B: TEMP effective removal plus IDLE/JUNK segments; no restoration."""
 return event_mask(frame_cell,temp_removed,{s:(r if r in DROP_R else 'UNSURE') for s,r in roles.items()},seg_cells,guard)


def effective_temp(frame_cell,temp_removed,guard=GUARD):
 return [] if keep_frames(frame_cell,sorted(set(temp_removed)),guard)[1] else sorted(set(temp_removed))


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--anchors',type=Path,required=True);ap.add_argument('--temp',type=Path,required=True,help='TEMP removal dir (summary + removed cells)')
 ap.add_argument('--output',type=Path,required=True);ap.add_argument('--model',required=True);ap.add_argument('--tp',type=int,default=4);ap.add_argument('--chunk',type=int,default=16)
 ap.add_argument('--videos',nargs='*');a=ap.parse_args()
 from PIL import Image
 from vllm import LLM,SamplingParams
 vids=a.videos or sorted(p.stem for p in a.temp.glob('*.json'))
 a.output.mkdir(parents=True,exist_ok=True);todo=[v for v in vids if not (a.output/f'{v}.json').exists()]
 llm=LLM(model=a.model,dtype='float16',tensor_parallel_size=a.tp,gpu_memory_utilization=.85,max_model_len=16384,max_num_seqs=16,limit_mm_per_prompt={'image':2*(WIN+2*CTX_SEG),'video':0},seed=0)
 params=SamplingParams(temperature=0.0,max_tokens=200,seed=0);start=time.time()
 for c0 in range(0,len(todo),a.chunk):
  conv=[];blocks=[];metas={};temps={}
  for vid in todo[c0:c0+a.chunk]:
   m=json.loads((a.anchors/vid/'meta.json').read_text());t=json.loads((a.temp/f'{vid}.json').read_text());metas[vid]=m;temps[vid]=t
   cells=[x['cell'] for x in m['anchors']];seg=segments(cells);sids=sorted(seg);imgs={c:Image.open(a.anchors/vid/f'{c}.png').convert('RGB') for c in cells}
   for b in range(0,len(sids),WIN):
    tg=sids[b:b+WIN];pre=sids[max(0,b-CTX_SEG):b];post=sids[b+WIN:b+WIN+CTX_SEG];content=[]
    for s in pre+tg+post:
     tag='CONTEXT' if s in pre or s in post else 'TARGET';content.append({'type':'text','text':f'[seg={s} t={2*s}-{2*s+2}s {tag}]'})
     content+=[{'type':'image_pil','image_pil':imgs[c]} for c in seg[s]]
    lo,hi=2*tg[0],min(2*tg[-1]+2,m['duration'])
    content.append({'type':'text','text':ROLES_PROMPT.format(summary=t['summary'],lo=lo,hi=hi,dur=m['duration'],ids=','.join(map(str,tg)))})
    conv.append([{'role':'system','content':SYSTEM},{'role':'user','content':content}]);blocks.append((vid,tg,pre+post))
  res=llm.chat(conv,params,use_tqdm=False);per={}
  for (vid,tg,ctx),o in zip(blocks,res):
   roles,st=parse_roles(o.outputs[0].text,tg);per.setdefault(vid,[]).append({'targets':tg,'context':ctx,'raw':o.outputs[0].text,'roles':{str(k):v for k,v in roles.items()},'status':st,'prompt_tokens':len(o.prompt_token_ids)})
  for vid in todo[c0:c0+a.chunk]:
   m=metas[vid];t=temps[vid];seg=segments([x['cell'] for x in m['anchors']]);roles={int(k):v for b in per[vid] for k,v in b['roles'].items()}
   rem,guard=event_mask(m['frame_cell'],t['removed_cells'],roles,seg);kept,_=keep_frames(m['frame_cell'],rem,1.01);eff=effective_temp(m['frame_cell'],t['removed_cells'])
   out={'video_id':vid,'n_frames':m['n_frames'],'duration':m['duration'],'summary':t['summary'],'blocks':per[vid],'segment_cells':{str(k):v for k,v in seg.items()},
    'roles':{str(k):v for k,v in roles.items()},'temp_removed_cells':t['removed_cells'],'temp_effective_removed_cells':eff,'removed_cells':rem,'guard_tripped':guard,
    'added_removed_cells':sorted(set(rem)-set(eff)),'restored_cells':sorted(set(eff)-set(rem)),'kept_frames':len(kept),'keep_rate':len(kept)/max(1,m['n_frames']),
    'rule':{'window_segments':WIN,'context_segments':CTX_SEG,'guard':GUARD,'keep':sorted(KEEP_R),'drop':sorted(DROP_R)}}
   (a.output/f'{vid}.json').write_text(json.dumps(out)+'\n')
  print(json.dumps({'done':c0+a.chunk,'of':len(todo),'blocks':len(blocks),'wall':round(time.time()-start,1)}),flush=True)
if __name__=='__main__':main()
