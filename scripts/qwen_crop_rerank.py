"""V4 E4 pilot: Qwen3-VL-32B looks at the actual candidate crops and picks one.

At every DENSE keyframe whose DENSE point is valid and whose hold span contains
a no-face frame, fixed-size legal windows along the free axis are built around
the DENSE window centre c0: c0 + d*win for d in (-.8,-.4,0,.4,.8), clipped to
the legal range and de-duplicated (< .15 win apart; c0 always kept).  The
teacher sees the full frame (the same 640 px DENSE PNG) and the candidate crops
cut from it (native pixels, no resize), labelled A.. in spatial order, and
answers {"choice":"A","uncertain":false}.  Each keyframe is asked twice, in
spatial order and reversed; the chosen window is accepted only if both orders
pick the same window and neither is uncertain, otherwise the DENSE point stays.
Output = point JSON in the DENSE format (the free-axis coordinate replaced by
the accepted centre) + per-keyframe candidates/choices, consumed by
max_window_release unchanged.
"""
import argparse,hashlib,json,re,time
from pathlib import Path
import numpy as np
from scripts.qwen_context_point import consumed

OFFS=(-.8,-.4,0.,.4,.8);MIN_GAP=.15;LABELS='ABCDE'
SYSTEM=("You are a professional video editor reframing a video for a different screen shape. "
        "You compare candidate crops of one frame and pick the one the reframed shot should show.")
USER=("The first image is the full {w}x{h} frame. It must be cropped to a {rw}:{rh} window of fixed size that can only slide {slide}. "
      "The next {n} images are candidate crops of exactly that size, labelled {labels}, ordered {order}. "
      "Choose the crop that best keeps the main subject of the whole event, the thing a viewer is meant to watch, together with what is needed "
      "to understand it (the other person in an interaction, the hands and the object being used, the ball or the target). "
      "Do not prefer a crop because it is centred or looks nicer. "
      "Return ONLY JSON: {{\"choice\":\"A\",\"uncertain\":false}}. Set uncertain to true if two or more crops are equally good.")
_JSON=re.compile(r"\{.*?\}",re.DOTALL)


def prompt_sha():return hashlib.sha256((SYSTEM+'\n'+USER).encode()).hexdigest()


def candidates(p,win):
 lo,hi=win/2,1-win/2;c0=float(np.clip(p,lo,hi));out=[c0]
 for d in sorted(OFFS,key=abs)[1:]:
  c=float(np.clip(c0+d*win,lo,hi))
  if all(abs(c-x)>=MIN_GAP*win for x in out):out.append(c)
 return sorted(out),c0


def parse_choice(text,n):
 m=_JSON.search(text or '')
 if not m:return None,'no_json'
 try:d=json.loads(m.group(0))
 except ValueError:return None,'bad_json'
 c=str(d.get('choice','')).strip().upper()[:1]
 if c not in LABELS[:n]:return None,'bad_choice'
 if bool(d.get('uncertain',False)):return None,'uncertain'
 return LABELS.index(c),'ok'


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--videos',nargs='*');ap.add_argument('--index');ap.add_argument('--ratio-keys',default='1-3,3-1')
 ap.add_argument('--cache-dir',type=Path,required=True);ap.add_argument('--frames-dir',type=Path,required=True);ap.add_argument('--dense',type=Path,required=True)
 ap.add_argument('--model',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--tp',type=int,default=4);a=ap.parse_args()
 from PIL import Image
 from vllm import LLM,SamplingParams
 from aic.max_window_path import geometry
 if a.index:jobs=[(r['video_id'],{'t':r['targetRatioWH']}) for r in map(json.loads,open(a.index))]
 else:
  rk={k:[int(x) for x in k.split('-')] for k in a.ratio_keys.split(',')};jobs=[(v,rk) for v in a.videos]
 a.output.mkdir(parents=True,exist_ok=True)
 llm=LLM(model=a.model,dtype='float16',tensor_parallel_size=a.tp,gpu_memory_utilization=.85,max_model_len=8192,max_num_seqs=16,limit_mm_per_prompt={'image':1+len(OFFS),'video':0},seed=0)
 params=SamplingParams(temperature=0.0,max_tokens=32,seed=0);start=time.time()
 for vid,ratios in jobs:
  dest=a.output/f'{vid}.json'
  if dest.exists():continue
  t0=time.time();d2=json.loads((a.dense/f'{vid}.json').read_text());mt=json.loads((a.frames_dir/vid/'meta.json').read_text());keys=d2['keyframes']
  if mt['keyframes']!=keys:raise ValueError('frames dir keyframes != DENSE keyframes '+vid)
  out={k:v for k,v in d2.items() if k not in ('ratios','queried','seconds')}
  out.update({'prompt':'crop_rerank','prompt_sha256':prompt_sha(),'dense_prompt_sha256':d2.get('prompt_sha256'),'offsets':OFFS,'min_gap':MIN_GAP,'dense_parent':str(a.dense/f'{vid}.json'),'ratios':{}})
  queried=0;imgs={}
  for k,r in ratios.items():
   z=np.load(a.cache_dir/f'{vid}_{k}.npz');W,H=int(z['W']),int(z['H']);n=len(z['reset']);face=z['chosen']>=0
   w,h,axis=geometry(W,H,r);old=d2['ratios'][k];pts=[None if p is None else list(p) for p in old['points']];info=[None]*len(keys);src=['dense_unconsumed']*len(keys)
   if axis is None:out['ratios'][k]={'ratio':r,'points':pts,'status':old['status'],'raw':old['raw'],'source':src,'rerank':info};continue
   comp=0 if axis==0 else 1;win=(w/W) if comp==0 else (h/H);use=consumed(keys,n,face)
   slide='left or right' if comp==0 else 'up or down';orders=('from left to right','from right to left') if comp==0 else ('from top to bottom','from bottom to top')
   conv=[];meta=[]
   for j,t in enumerate(keys):
    p=old['points'][j]
    if not use[j]:continue
    if p is None or p[2]:src[j]='dense_invalid';continue
    cs,c0=candidates(p[comp],win)
    if len(cs)<2:src[j]='single_candidate';continue
    if t not in imgs:imgs[t]=Image.open(a.frames_dir/vid/f'{t}.png').convert('RGB')
    im=imgs[t];iw,ih=im.size;L=iw if comp==0 else ih
    crops=[]
    for c in cs:
     lo=int(round((c-win/2)*L));hi=min(L,lo+int(round(win*L)))
     crops.append(im.crop((lo,0,hi,ih)) if comp==0 else im.crop((0,lo,iw,hi)))
    for o,order in enumerate(orders):
     seq=list(range(len(cs))) if o==0 else list(range(len(cs)))[::-1]
     content=[{'type':'text','text':'[FULL FRAME]'},{'type':'image_pil','image_pil':im}]
     for lab,i in zip(LABELS,seq):content+=[{'type':'text','text':f'[{lab}]'},{'type':'image_pil','image_pil':crops[i]}]
     content.append({'type':'text','text':USER.format(w=W,h=H,rw=r[0],rh=r[1],slide=slide,n=len(cs),labels=', '.join(LABELS[:len(cs)]),order=order)})
     conv.append([{'role':'system','content':SYSTEM},{'role':'user','content':content}]);meta.append((j,o,seq))
    info[j]={'cands':cs,'c0':c0,'dense_index':cs.index(c0)}
   res=llm.chat(conv,params,use_tqdm=False) if conv else [];queried+=len(conv)
   for (j,o,seq),res_o in zip(meta,res):
    txt=res_o.outputs[0].text;ci,st=parse_choice(txt,len(seq));info[j][f'raw{o}']=txt;info[j][f'status{o}']=st;info[j][f'pick{o}']=None if ci is None else seq[ci]
   for j,x in enumerate(info):
    if x is None:continue
    if x.get('pick0') is not None and x.get('pick0')==x.get('pick1'):
     pts[j][comp]=float(x['cands'][x['pick0']]);src[j]='rerank_agree'
    else:src[j]='rerank_disagree' if x.get('pick0') is not None and x.get('pick1') is not None else 'rerank_fail'
   out['ratios'][k]={'ratio':r,'points':pts,'status':old['status'],'raw':old['raw'],'source':src,'rerank':info}
  out['queried']=queried;out['seconds']=time.time()-t0;dest.write_text(json.dumps(out)+'\n')
  cnt={k:{s:v['source'].count(s) for s in ('rerank_agree','rerank_disagree','rerank_fail')} for k,v in out['ratios'].items()}
  print(json.dumps({'vid':vid,'keys':len(keys),'queried':queried,'src':cnt,'s':round(out['seconds'],1),'wall':round(time.time()-start,1)}),flush=True)
if __name__=='__main__':main()
