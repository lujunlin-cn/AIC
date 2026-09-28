"""V4 E2: the same DENSE keyframes, but the 32B teacher also sees nearby frames
of the same shot before naming the subject point of the target frame.

Only change vs the DENSE point run (scripts/qwen_subject_point.py, --densify 2):
up to 2 earlier + 2 later DENSE keyframes of the same shot within 1.05 s of the
target (0.5 s grid -> a ~2 s clip, 5 frames) are added as CONTEXT images, in
time order, each labelled with its offset from the target.  The target image
is the same 640 px PNG as before; context images are downscaled to 320 px long
side.  Same SYSTEM text, same JSON answer and the frozen parser
(scripts.qwen_subject_point.parse); the point must be in target-frame
coordinates.  Replies are only requested where they can change the output:
  consumed   the keyframe's hold span [t, next keyframe) contains a no-face
             frame (shot starts are keyframes, so the span never crosses a cut)
  context    at least one same-shot neighbour exists
Everything else keeps the DENSE reply byte-identically; parse failures and
uncertain answers also fall back to the DENSE reply (never to B0 directly).
Output = point JSON in the DENSE format (keyframes, ratios[k].points/raw/status)
plus per-keyframe 'source', so max_window_release consumes it unchanged.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from scripts.qwen_subject_point import SYSTEM,USER,parse

CTX_S,CTX_SIDE,CTX_PX=1.05,2,320
EXTRA=("You also get nearby CONTEXT frames from the same shot (smaller images, labelled with their time offset from the TARGET frame). "
       "Use them only to understand what is happening: who is acting, what the main event is and where it is going. "
       "The answer is about the TARGET frame (offset 0.0 s) and x,y refer to the TARGET frame.")


def prompt_sha():
 return hashlib.sha256((SYSTEM+'\n'+USER+' '+EXTRA).encode()).hexdigest()


def consumed(keys,n,face):
 out=[]
 for j,t in enumerate(keys):
  e=keys[j+1] if j+1<len(keys) else n;out.append(bool((~face[t:e]).any()))
 return out


def neighbours(keys,j,shot,fps):
 t=keys[j];pre=[k for k in keys[:j] if shot[k]==shot[t] and (t-k)/fps<=CTX_S][-CTX_SIDE:]
 post=[k for k in keys[j+1:] if shot[k]==shot[t] and (k-t)/fps<=CTX_S][:CTX_SIDE]
 return pre,post


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--videos',nargs='*');ap.add_argument('--index');ap.add_argument('--ratio-keys',default='1-3,3-1')
 ap.add_argument('--cache-dir',type=Path,required=True);ap.add_argument('--frames-dir',type=Path,required=True);ap.add_argument('--dense',type=Path,required=True)
 ap.add_argument('--model',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--tp',type=int,default=4);a=ap.parse_args()
 from PIL import Image
 from vllm import LLM,SamplingParams
 if a.index:jobs=[(r['video_id'],{'t':r['targetRatioWH']}) for r in map(json.loads,open(a.index))]
 else:
  rk={k:[int(x) for x in k.split('-')] for k in a.ratio_keys.split(',')};jobs=[(v,rk) for v in a.videos]
 a.output.mkdir(parents=True,exist_ok=True)
 llm=LLM(model=a.model,dtype='float16',tensor_parallel_size=a.tp,gpu_memory_utilization=.85,max_model_len=8192,max_num_seqs=16,limit_mm_per_prompt={'image':1+2*CTX_SIDE,'video':0},seed=0)
 params=SamplingParams(temperature=0.0,max_tokens=64,seed=0);start=time.time()
 for vid,ratios in jobs:
  dest=a.output/f'{vid}.json'
  if dest.exists():continue
  t0=time.time();d2=json.loads((a.dense/f'{vid}.json').read_text());mt=json.loads((a.frames_dir/vid/'meta.json').read_text())
  keys=d2['keyframes'];fps=mt['fps']
  if mt['keyframes']!=keys:raise ValueError('frames dir keyframes != DENSE keyframes '+vid)
  imgs={};small={}
  def img(i):
   if i not in imgs:
    imgs[i]=Image.open(a.frames_dir/vid/f'{i}.png').convert('RGB');w,h=imgs[i].size;sc=min(1.,CTX_PX/max(w,h))
    small[i]=imgs[i].resize((round(w*sc),round(h*sc)),Image.BICUBIC) if sc<1 else imgs[i]
   return imgs[i]
  out={k:v for k,v in d2.items() if k not in ('ratios','queried','seconds')}
  out.update({'prompt':'context_point','dense_prompt_sha256':d2.get('prompt_sha256'),'prompt_sha256':prompt_sha(),'context':{'max_s':CTX_S,'per_side':CTX_SIDE,'px':CTX_PX},'dense_parent':str(a.dense/f'{vid}.json'),'ratios':{}})
  queried=0;ptok=[]
  for k,r in ratios.items():
   z=np.load(a.cache_dir/f'{vid}_{k}.npz');W,H=int(z['W']),int(z['H']);n=len(z['reset']);shot=np.cumsum(z['reset'].astype(bool));face=z['chosen']>=0
   orient='portrait, narrower than the frame' if r[0]/r[1]<W/H else 'landscape, wider than the frame'
   base=USER.format(w=W,h=H,rw=r[0],rh=r[1],orient=orient)+' '+EXTRA
   use=consumed(keys,n,face);ask=[];conv=[];ctxinfo={}
   for j,t in enumerate(keys):
    if not use[j]:continue
    pre,post=neighbours(keys,j,shot,fps)
    if not pre and not post:continue
    content=[]
    for c in pre+[t]+post:
     if c==t:content+=[{'type':'text','text':'[TARGET t=+0.0s]'},{'type':'image_pil','image_pil':img(t)}]
     else:img(c);content+=[{'type':'text','text':f'[CONTEXT t={(c-t)/fps:+.1f}s]'},{'type':'image_pil','image_pil':small[c]}]
    content.append({'type':'text','text':base});conv.append([{'role':'system','content':SYSTEM},{'role':'user','content':content}]);ask.append(j)
    ctxinfo[j]={'pre':pre,'post':post,'offsets_s':[round((c-t)/fps,3) for c in pre+post]}
   res=llm.chat(conv,params,use_tqdm=False) if conv else [];queried+=len(ask)
   new={j:o.outputs[0].text for j,o in zip(ask,res)};ptok+=[len(o.prompt_token_ids) for o in res]
   old=d2['ratios'][k];pts=[];st=[];raw=[];src=[];ctx=[]
   for j,t in enumerate(keys):
    if j in new:
     p,s_=parse(new[j]);raw.append(new[j]);ctx.append(ctxinfo[j])
     if p is None or p['uncertain']:pts.append(old['points'][j]);st.append('fallback_dense_'+('uncertain' if p else s_));src.append('dense_fallback')
     else:pts.append([p['x'],p['y'],False]);st.append('ok');src.append('context')
    else:
     pts.append(old['points'][j]);st.append(old['status'][j]);raw.append(old['raw'][j]);ctx.append(None);src.append('dense_unconsumed' if not use[j] else 'dense_no_context')
   out['ratios'][k]={'ratio':r,'points':pts,'status':st,'raw':raw,'source':src,'context':ctx,'dense_raw':old['raw']}
  out['queried']=queried;out['prompt_tokens']={'min':min(ptok),'max':max(ptok),'mean':float(np.mean(ptok))} if ptok else None;out['seconds']=time.time()-t0
  out['target_px']=list(next(iter(imgs.values())).size) if imgs else None;dest.write_text(json.dumps(out)+'\n')
  print(json.dumps({'vid':vid,'keys':len(keys),'queried':queried,'ctx_ok':{k:v['source'].count('context') for k,v in out['ratios'].items()},'fallback':{k:v['source'].count('dense_fallback') for k,v in out['ratios'].items()},
   'tok':out['prompt_tokens'],'s':round(out['seconds'],1),'wall':round(time.time()-start,1)}),flush=True)
if __name__=='__main__':main()
