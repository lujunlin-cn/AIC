"""TEMP_BOUNDARY_V3: dense 32B re-judgement only at TEMP keep/drop transitions.

TEMP (46.84) decides whole 1 s cells (scripts/qwen_temporal_removal.py).  Every
transition between a kept cell and an adjacent removed cell defines a 2 s zone
(those two cells).  The zone is re-observed at 0.25 s (8 sub-cells; anchor =
decoded frame nearest the sub-cell centre, same centre-anchor convention) with
<=2 TEMP anchors of context on each side and the stored TEMP summary; the same
32B teacher, system prompt and removal criteria list the sub-cells to REMOVE.
Refined boundary = monotone split (keep side | remove side) with the fewest
disagreements with the reply; ties -> nearest to the original cell edge.
Parse failure / no reply -> original TEMP boundary.  A frame flips from its
TEMP decision only if every zone covering it flips it; frames outside zones
keep TEMP.  Video guard: if the refined mask drops > 50% (or all) frames, the
TEMP mask is used.  Spatial output is fixed: restored frames take N0 bboxes at
packaging (scripts/temporal_boundary_release.py).

  --extract-only   cv env: zone sub-cell anchors -> <frames-dir>/<vid>/z<k>_<i>.png + zones.json
  otherwise        vLLM env: reads PNGs, writes <output>/<vid>.json
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scripts.qwen_temporal_removal import SYSTEM,parse_remove,keep_frames,GUARD

SUB,CTX=4,2
JUDGE=("Video summary: {summary}\n"
       "A coarse 1-second edit of this {dur:.1f}s video {change} at about t={edge:.2f}s. "
       "CONTEXT frames show the neighbouring 1-second moments and how the coarse edit treated them. "
       "TARGET frames 0..{hi} are 0.25 s apart around that change; judge each TARGET frame on its own. "
       "Mark a TARGET frame REMOVE only if that moment clearly contributes nothing to a highlight clip of this video: "
       "black, blank or fading frames; title cards, credits, logos or end screens unrelated to the event; frozen or transition frames; "
       "content unrelated to the main event. If in doubt, keep it. "
       "Return ONLY JSON: {{\"remove\":[ids],\"reason\":\"short\"}}")
CHANGE={'keep_to_remove':'keeps the moments before and cuts the moments after','remove_to_keep':'cuts the moments before and keeps the moments after'}


def transitions(frame_cell,removed_cells):
 """Adjacent (kept, removed) cell pairs of the effective TEMP mask."""
 fc=np.asarray(frame_cell);_,guard=keep_frames(fc,removed_cells)
 rm=set() if guard else set(removed_cells);cells=np.unique(fc);out=[]
 for c0,c1 in zip(cells[:-1].tolist(),cells[1:].tolist()):
  if c1==c0+1 and (c0 in rm)!=(c1 in rm):out.append({'left':c0,'right':c1,'dir':'keep_to_remove' if c1 in rm else 'remove_to_keep'})
 return out,guard


def subcell(times,c0):
 """Sub-cell id (0..2*SUB-1) of every frame inside the zone starting at cell c0, else -1."""
 s=np.floor((np.asarray(times,float)-c0)*SUB+1e-9).astype(int);return np.where((s>=0)&(s<2*SUB),s,-1)


def best_split(direction,labels):
 """labels: {sub-cell: 1 remove / 0 keep}; returns j (sub-cells < j on the first side)."""
 first=0 if direction=='keep_to_remove' else 1
 cost=[sum(1 for i,l in labels.items() if (l!=first if i<j else l==first)) for j in range(2*SUB+1)]
 m=min(cost);return min((j for j in range(2*SUB+1) if cost[j]==m),key=lambda j:(abs(j-SUB),j))


def refine_mask(times,frame_cell,removed_cells,zone_results):
 """TEMP keep mask with zone flips applied; returns (keep bool array, info)."""
 fc=np.asarray(frame_cell);n=len(fc);kept,_=keep_frames(fc,removed_cells);temp=np.zeros(n,bool);temp[kept]=True
 cover=np.zeros(n,int);flip=np.zeros(n,int)
 for z in zone_results:
  s=subcell(times,z['left']);inz=s>=0;cover+=inz
  if z.get('split') is None:continue
  first_keep=z['dir']=='keep_to_remove';new=np.where(s<z['split'],first_keep,not first_keep)
  flip+=inz&(new!=temp)
 keep=np.where((cover>0)&(flip==cover),~temp,temp)
 guard=bool((~keep).mean()>GUARD or not keep.any())
 if guard:keep=temp.copy()
 return keep,{'refine_guard':guard,'restored':int((keep&~temp).sum()),'newly_removed':int((~keep&temp).sum())}


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--index');ap.add_argument('--records');ap.add_argument('--anchors',type=Path,required=True);ap.add_argument('--removal',type=Path,required=True)
 ap.add_argument('--frames-dir',type=Path,required=True);ap.add_argument('--output',type=Path);ap.add_argument('--model')
 ap.add_argument('--size',type=int,default=448);ap.add_argument('--tp',type=int,default=4);ap.add_argument('--chunk',type=int,default=64)
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
   d=a.frames_dir/vid;rf=a.removal/f'{vid}.json'
   if (d/'zones.json').exists() or not rf.exists():continue
   meta=json.loads((a.anchors/vid/'meta.json').read_text());rem=json.loads(rf.read_text());tr,guard=transitions(meta['frame_cell'],rem['removed_cells'])
   d.mkdir(parents=True,exist_ok=True)
   if not tr:(d/'zones.json').write_text(json.dumps({'n_frames':meta['n_frames'],'zones':[],'times':None,'temp_guard':guard})+'\n');continue
   times=[st.time_seconds for st,_ in _decoded(path)]
   if len(times)!=meta['n_frames'] or np.any(np.floor(np.asarray(times)+1e-9).astype(int)!=np.asarray(meta['frame_cell'])):raise ValueError('time/cell mismatch '+vid)
   want={}
   for k,z in enumerate(tr):
    s=subcell(times,z['left']);z['samples']=[]
    for i in range(2*SUB):
     idx=np.flatnonzero(s==i)
     if not len(idx):continue
     f=int(idx[np.argmin(np.abs(np.asarray(times)[idx]-(z['left']+(i+.5)/SUB)))]);z['samples'].append({'sub':i,'frame':f,'t':float(times[f])});want.setdefault(f,[]).append(f'z{k}_{i}')
   for st,fr in _decoded(path):
    if st.index in want:
     rgb=fr.to_ndarray(format='rgb24');h,w=rgb.shape[:2];sc=min(1.,a.size/max(w,h))
     img=cv2.resize(rgb,(round(w*sc),round(h*sc)),interpolation=cv2.INTER_AREA) if sc<1 else rgb
     for name in want[st.index]:cv2.imwrite(str(d/f'{name}.png'),cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
   (d/'zones.json').write_text(json.dumps({'n_frames':len(times),'zones':tr,'times':times,'temp_guard':guard})+'\n');print('EXTRACTED',vid,len(tr),flush=True)
  return
 from PIL import Image
 from vllm import LLM,SamplingParams
 llm=LLM(model=a.model,dtype='float16',tensor_parallel_size=a.tp,gpu_memory_utilization=.85,max_model_len=16384,max_num_seqs=16,limit_mm_per_prompt={'image':2*SUB+2*CTX,'video':0},seed=0)
 sp=SamplingParams(temperature=0.0,max_tokens=96,seed=0);a.output.mkdir(parents=True,exist_ok=True);start=time.time()
 todo=[v for v,_ in jobs if (a.frames_dir/v/'zones.json').exists() and not (a.output/f'{v}.json').exists()];nq=0
 for c0 in range(0,len(todo),a.chunk):
  chunk=todo[c0:c0+a.chunk];conv=[];keys=[];Z={}
  for vid in chunk:
   Z[vid]=json.loads((a.frames_dir/vid/'zones.json').read_text());meta=json.loads((a.anchors/vid/'meta.json').read_text());rem=json.loads((a.removal/f'{vid}.json').read_text())
   rm=set(rem['removed_cells']);anc={x['cell']:x for x in meta['anchors']}
   for k,z in enumerate(Z[vid]['zones']):
    if not z['samples']:continue
    content=[]
    for c in [z['left']-i for i in range(CTX,0,-1)]:
     if c in anc:content+=[{'type':'text','text':f"[CONTEXT t={c+.5:.1f}s {'cut' if c in rm else 'kept'}]"},{'type':'image_pil','image_pil':Image.open(a.anchors/vid/f'{c}.png').convert('RGB')}]
    for x in z['samples']:content+=[{'type':'text','text':f"[id={x['sub']} t={x['t']:.2f}s TARGET]"},{'type':'image_pil','image_pil':Image.open(a.frames_dir/vid/f"z{k}_{x['sub']}.png").convert('RGB')}]
    for c in [z['right']+i for i in range(1,CTX+1)]:
     if c in anc:content+=[{'type':'text','text':f"[CONTEXT t={c+.5:.1f}s {'cut' if c in rm else 'kept'}]"},{'type':'image_pil','image_pil':Image.open(a.anchors/vid/f'{c}.png').convert('RGB')}]
    content.append({'type':'text','text':JUDGE.format(summary=rem['summary'],dur=meta['duration'],change=CHANGE[z['dir']],edge=float(z['right']),hi=2*SUB-1)})
    conv.append([{'role':'system','content':SYSTEM},{'role':'user','content':content}]);keys.append((vid,k))
  replies=[o.outputs[0].text for o in llm.chat(conv,sp,use_tqdm=False)] if conv else [];nq+=len(conv);R={}
  for (vid,k),t in zip(keys,replies):R[(vid,k)]=t
  for vid in chunk:
   zs=[]
   for k,z in enumerate(Z[vid]['zones']):
    t=R.get((vid,k));subs={x['sub'] for x in z['samples']}
    if t is None:zs.append({**z,'raw':None,'status':'no_samples','split':None});continue
    rmv,st=parse_remove(t,subs)
    split=None if st in ('no_json','bad_json','bad_list') else best_split(z['dir'],{i:int(i in rmv) for i in subs})
    zs.append({**z,'raw':t,'removed':rmv,'status':st,'split':split,'moved_subcells':None if split is None else split-SUB})
   (a.output/f'{vid}.json').write_text(json.dumps({'video_id':vid,'zones':zs,'rule':{'sub':SUB,'context':CTX,'guard':GUARD}})+'\n')
  print(json.dumps({'done':c0+len(chunk),'of':len(todo),'queries':nq,'wall':round(time.time()-start,1)}),flush=True)
if __name__=='__main__':main()
