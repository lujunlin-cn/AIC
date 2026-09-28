"""P2: Qwen3-VL-32B as a spatial subject locator only (no frame deletion, no scale).

For each video, keyframes are taken every --step seconds (plus every shot start
from the cached B0 reset flags).  Each keyframe is sent alone with the target
aspect ratio; the model returns one subject point in 0-1000 normalised image
coordinates.  Output is cached per video as JSON (keyframe index -> point,
status, raw).  The prompt, parser and step are fixed on RetargetVid dev before
any official inference; unparsable/out-of-range replies are recorded and later
fall back to B0.

  --cache-dir   OBS cache (for shot resets / frame count)
  --videos      *.AVI dir (RetargetVid) or --index (official, frozen inference)
  --frames-dir  optional pre-extracted keyframes <vid>/<frame>.png written by
                `--extract-only` in the project cv env (PyAV 15.1, the decoder
                B0 uses); the vLLM env ships PyAV 17.1, which rejects some
                colour-transfer tags (trc log316) on the official set.
  --prompt      point (N0/P0 interface, unchanged) | region (T1: target-ratio
                aware subject_box + context_box, mapped to a legal window later
                by aic.max_window_path.region_points; nothing else changes)
  --densify     T2: extra keyframes between the --step grid (2 = midpoints);
                the dense set is a superset of the step grid plus shot starts
  --reuse       earlier point-prompt output dirs; keyframes already answered
                with the same prompt/image are copied instead of re-queried, so
                a dense run differs from its parent only on the added keyframes
"""
import argparse,hashlib,json,re,time
from pathlib import Path
import numpy as np

SYSTEM=("You are a professional video editor reframing a video for a different screen shape. "
        "You look at one frame and decide which subject the reframed shot must keep.")
USER=("The frame is {w}x{h}. It will be cropped to the LARGEST window of aspect ratio {rw}:{rh} ({orient}), "
      "so only a part of the frame stays visible. Identify the main subject of the whole event, the thing a viewer "
      "is meant to watch. It is not always the largest face: for people interacting, keep the interaction as a whole; "
      "for sport, action or animals, keep the acting subject; for text or products, keep them. "
      "Return ONLY JSON: {{\"subject_point\":[x,y],\"uncertain\":false}} where x,y are integers in 0..1000 "
      "normalised to the image width and height (0,0 = top-left). Set uncertain to true if there is no clear subject.")
USER_REGION=("The frame is {w}x{h}. It will be reframed to aspect ratio {rw}:{rh} ({orient}) with the LARGEST such window: "
      "a {cw}x{ch} window that keeps the full frame {full} and can only slide {slide}, so only about {pct}% of the frame {dim} stays visible. "
      "Decide what the reframed shot must keep. subject_box = the main subject of the whole event, the thing a viewer is meant to watch "
      "(not always the largest face: for sport, action or animals, the acting subject; for text or products, the text or product). "
      "context_box = the subject together with what is needed to understand what is happening (the other person in an interaction, "
      "the hands and the object being used, the ball or the target); use the same box if nothing else is needed. "
      "Return ONLY JSON: {{\"subject_box\":[x1,y1,x2,y2],\"context_box\":[x1,y1,x2,y2],\"uncertain\":false}} with integers in 0..1000 "
      "normalised to the image width and height (0,0 = top-left). Set uncertain to true if there is no clear subject.")
PROMPTS={'point':(USER,64),'region':(USER_REGION,96)}
_JSON=re.compile(r"\{.*?\}",re.DOTALL)
_JSON_NESTED=re.compile(r"\{(?:[^{}]|\[[^\]]*\])*\}",re.DOTALL)


def prompt_sha(kind):
 return hashlib.sha256((SYSTEM+'\n'+PROMPTS[kind][0]).encode()).hexdigest()


def parse_region(text):
 m=_JSON_NESTED.search(text or '')
 if not m:return None,'no_json'
 try:d=json.loads(m.group(0))
 except ValueError:return None,'bad_json'
 out=[]
 for key in ('subject_box','context_box'):
  b=d.get(key,d.get('subject_box') if key=='context_box' else None)
  if not isinstance(b,(list,tuple)) or len(b)!=4:return None,'no_box'
  try:b=[float(v) for v in b]
  except (TypeError,ValueError):return None,'bad_box'
  if not all(0<=v<=1000 for v in b):return None,'out_of_range'
  if b[2]<=b[0] or b[3]<=b[1]:return None,'degenerate'
  out.append([v/1000 for v in b])
 return {'subject':out[0],'context':out[1],'uncertain':bool(d.get('uncertain',False))},'ok'


def region_text(W,H,r):
 rw,rh=r;w=min(W,H*rw/rh);h=w*rh/rw
 if w<W-1e-9:return dict(cw=round(w),ch=round(h),full='height',slide='left or right',pct=round(100*w/W),dim='width')
 return dict(cw=round(w),ch=round(h),full='width',slide='up or down',pct=round(100*h/H),dim='height')


def parse(text):
 m=_JSON.search(text or '')
 if not m:return None,'no_json'
 try:d=json.loads(m.group(0))
 except ValueError:return None,'bad_json'
 p=d.get('subject_point')
 if not isinstance(p,(list,tuple)) or len(p)!=2:return None,'no_point'
 try:x,y=float(p[0]),float(p[1])
 except (TypeError,ValueError):return None,'bad_point'
 if not (0<=x<=1000 and 0<=y<=1000):return None,'out_of_range'
 return {'x':x/1000,'y':y/1000,'uncertain':bool(d.get('uncertain',False))},'ok'


def keyframes(reset,fps,step,densify=1):
 n=len(reset);s=max(1,int(round(fps*step)))
 grid=range(0,n,s) if densify==1 else {int(np.floor(i*s/densify+.5)) for i in range(int(np.ceil(n*densify/s))+1)}
 k={i for i in grid if i<n}|{i for i in range(n) if reset[i]}
 return sorted(k)


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--videos');ap.add_argument('--index');ap.add_argument('--ratio-keys',default='1-3,3-1')
 ap.add_argument('--cache-dir',type=Path,required=True);ap.add_argument('--model',required=True);ap.add_argument('--output',type=Path,required=True)
 ap.add_argument('--step',type=float,default=1.0);ap.add_argument('--size',type=int,default=640);ap.add_argument('--tp',type=int,default=4)
 ap.add_argument('--part',type=int,default=0);ap.add_argument('--parts',type=int,default=1)
 ap.add_argument('--frames-dir',type=Path);ap.add_argument('--extract-only',action='store_true')
 ap.add_argument('--prompt',choices=sorted(PROMPTS),default='point');ap.add_argument('--densify',type=int,default=1)
 ap.add_argument('--reuse',type=Path,nargs='*',default=[]);a=ap.parse_args()
 if a.reuse and a.prompt!='point':raise ValueError('--reuse only copies point-prompt replies')
 import av,cv2
 from PIL import Image
 a.output.mkdir(parents=True,exist_ok=True)
 if a.index:
  from aic.inference import _record_path
  jobs=[(r['video_id'],_record_path(r,None),{'t':r['targetRatioWH']}) for r in map(json.loads,open(a.index))]
 else:
  rk={k:[int(x) for x in k.split('-')] for k in a.ratio_keys.split(',')};jobs=[(v.stem,str(v),rk) for v in sorted(Path(a.videos).glob('*.AVI'))]
 jobs=jobs[a.part::a.parts]
 if a.extract_only:
  from aic.video import _decoded
  for vid,path,ratios in jobs:
   z=np.load(a.cache_dir/f'{vid}_{next(iter(ratios))}.npz');W,H=int(z['W']),int(z['H'])
   with av.open(str(path)) as c:fps=float(c.streams.video[0].average_rate or 30)
   keys=keyframes(z['reset'],fps,a.step,a.densify);want=set(keys);d=a.frames_dir/vid;d.mkdir(parents=True,exist_ok=True)
   if (d/'meta.json').exists() and json.loads((d/'meta.json').read_text())['keyframes']==keys:continue
   for st,f in _decoded(path):
    if st.index in want:
     rgb=f.to_ndarray(format='rgb24');sc=min(1.,a.size/max(W,H))
     img=cv2.resize(rgb,(round(W*sc),round(H*sc)),interpolation=cv2.INTER_AREA) if sc<1 else rgb
     cv2.imwrite(str(d/f'{st.index}.png'),cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
   (d/'meta.json').write_text(json.dumps({'fps':fps,'keyframes':keys,'W':W,'H':H})+'\n');print('EXTRACTED',vid,len(keys),flush=True)
  return
 from vllm import LLM,SamplingParams
 llm=LLM(model=a.model,dtype='float16',tensor_parallel_size=a.tp,gpu_memory_utilization=.85,max_model_len=8192,max_num_seqs=16,limit_mm_per_prompt={'image':1,'video':0},seed=0)
 template,max_tokens=PROMPTS[a.prompt];params=SamplingParams(temperature=0.0,max_tokens=max_tokens,seed=0);start=time.time()
 for vid,path,ratios in jobs:
  dest=a.output/f'{vid}.json'
  if dest.exists():continue
  t0=time.time();k0=next(iter(ratios));z=np.load(a.cache_dir/f'{vid}_{k0}.npz');reset=z['reset'];W,H=int(z['W']),int(z['H'])
  if a.frames_dir:
   mt=json.loads((a.frames_dir/vid/'meta.json').read_text());fps=mt['fps'];keys=mt['keyframes']
   imgs={i:Image.open(a.frames_dir/vid/f'{i}.png').convert('RGB') for i in keys}
  else:
   with av.open(str(path)) as c:
    s=c.streams.video[0];fps=float(s.average_rate or 30)
    keys=keyframes(reset,fps,a.step,a.densify);want=set(keys);imgs={}
    for i,f in enumerate(c.decode(s)):
     if i in want:
      rgb=f.to_ndarray(format='rgb24');sc=min(1.,a.size/max(W,H))
      imgs[i]=Image.fromarray(cv2.resize(rgb,(round(W*sc),round(H*sc)),interpolation=cv2.INTER_AREA) if sc<1 else rgb)
  out={'video_id':vid,'W':W,'H':H,'fps':fps,'step':a.step,'densify':a.densify,'prompt':a.prompt,'prompt_sha256':prompt_sha(a.prompt),'keyframes':keys,'ratios':{}}
  prev=[json.loads((d/f'{vid}.json').read_text()) for d in a.reuse if (d/f'{vid}.json').exists()]
  queried=0
  for k,r in ratios.items():
   orient='portrait, narrower than the frame' if r[0]/r[1]<W/H else 'landscape, wider than the frame'
   old={}
   for p in prev:
    if k in p['ratios']:old.update({i:t for i,t in zip(p['keyframes'],p['ratios'][k]['raw'])})
   ask=[i for i in keys if i not in old];queried+=len(ask)
   text=template.format(w=W,h=H,rw=r[0],rh=r[1],orient=orient,**(region_text(W,H,r) if a.prompt=='region' else {}))
   conv=[[{'role':'system','content':SYSTEM},{'role':'user','content':[{'type':'image_pil','image_pil':imgs[i]},{'type':'text','text':text}]}] for i in ask]
   new=dict(zip(ask,[o.outputs[0].text for o in llm.chat(conv,params,use_tqdm=False)])) if ask else {}
   replies=[old.get(i,new.get(i)) for i in keys];pts=[];st=[]
   for i,t in zip(keys,replies):
    if a.prompt=='point':p,s_=parse(t);pts.append(None if p is None else [p['x'],p['y'],p['uncertain']])
    else:p,s_=parse_region(t);pts.append(None if p is None else [p['subject'],p['context'],p['uncertain']])
    st.append(s_)
   out['ratios'][k]={'ratio':r,'points' if a.prompt=='point' else 'regions':pts,'status':st,'raw':replies,'reused':[i in old for i in keys]}
  out['queried']=queried;out['seconds']=time.time()-t0;dest.write_text(json.dumps(out)+'\n')
  print(json.dumps({'vid':vid,'keys':len(keys),'queried':queried,'ok':{k:v['status'].count('ok') for k,v in out['ratios'].items()},'s':round(out['seconds'],1),'wall':round(time.time()-start,1)}),flush=True)
if __name__=='__main__':main()
