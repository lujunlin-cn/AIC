"""V5 P2 pilot: Qwen names the important subjects, GroundingDINO localises them.

Guide P2 route on a 25-video stratified slice of RetargetVid confirm2 (strata
axis x motion from the P0 per-video table).  Stages:
  qwen   32B teacher names 1-4 subjects from <=6 sampled DENSE keyframes
  detect GroundingDINO-tiny grounds those phrases on every DENSE keyframe PNG
  eval   identity selection (coverage, then mean score) -> free-axis centre per
         keyframe, DENSE fallback on misses, INTERP pipeline, paired vs mother
Only localisation is explored; no event/class label enters anything.
"""
import argparse,json,time
from pathlib import Path
import numpy as np

E=Path('/data/aic/experiments')
FRAMES=E/'QWEN_RV200_KEYFRAMES/d2';CACHE=E/'OBS_CACHE_RETARGET_ALL200';DENSE=E/'QWEN_RV200_POINT_D2/points'
ANN=Path('/data/aic/datasets/RetargetVid/annotations_all')
OUT=E/'V5_P2_GROUND'
SYS=("You help a professional video editor reframe a shot for a different screen shape. "
     "You name the subjects a crop of this shot must keep.")
USR=("These are sampled frames of one shot. List the 1-4 most important subjects "
     "(the people, animals or objects the shot is about, including what they interact with) "
     "as short lowercase English noun phrases. Return ONLY JSON: {\"subjects\":[\"...\",\"...\"]}")


def load_split(n=25,seed=20260929):
 m=json.load(open(E/'V5_P0_XRERANK/eval/metrics.json'))
 rows=[r for r in m['per_video'] if r['split']=='confirm2']
 by={}
 for r in rows:
  for rr in r['per_ratio'].values():by.setdefault((rr['axis'],rr['motion']),[]).append(r)
 quota={('x','fast'):7,('x','slow'):6,('y','fast'):6,('y','slow'):6}
 rng=np.random.default_rng(seed);sel={};strata={}
 for k,q in quota.items():
  vids=[r['vid'] for r in by.get(k,[])]
  for v in (rng.choice(vids,min(q,len(vids)),replace=False) if vids else []):sel[str(v)]=k;strata.setdefault(f'{k[0]}_{k[1]}',[]).append(str(v))
 return sel,strata


def stage_qwen(videos):
 from vllm import LLM,SamplingParams
 from PIL import Image
 dest=OUT/'qwen_subjects.json'
 subs=json.loads(dest.read_text()) if dest.exists() else {}
 todo=[v for v in videos if v not in subs]
 if not todo:print(json.dumps({'all_done':len(subs)}),flush=True);return
 conv=[];meta=[]
 for vid in todo:
  qq=json.loads((DENSE/f'{vid}.json').read_text());keys=qq['keyframes']
  idx=np.linspace(0,len(keys)-1,min(6,len(keys))).round().astype(int)
  imgs=[FRAMES/vid/f'{keys[i]}.png' for i in idx]
  content=[{'type':'image_pil','image_pil':Image.open(p).convert('RGB')} for p in imgs]
  content.append({'type':'text','text':USR})
  conv.append([{'role':'system','content':SYS},{'role':'user','content':content}]);meta.append(vid)
 llm=LLM(model='/data/aic/pretrained/qwen3_vl_32b_instruct',dtype='float16',tensor_parallel_size=4,gpu_memory_utilization=.85,max_model_len=8192,max_num_seqs=16,limit_mm_per_prompt={'image':6,'video':0},seed=0)
 res=llm.chat(conv,SamplingParams(temperature=0.,max_tokens=96,seed=0),use_tqdm=False)
 for vid,r in zip(meta,res):
  txt=r.outputs[0].text;ss=None
  try:
   m=json.loads(txt[txt.index('{'):txt.rindex('}')+1]);ss=[str(x).strip().lower() for x in m.get('subjects',[])][:4]
  except Exception:pass
  subs[vid]={'subjects':[s for s in (ss or []) if s],'raw':txt}
  dest.write_text(json.dumps(subs,indent=1)+'\n')
  print(json.dumps({'vid':vid,'subjects':subs[vid]['subjects']}),flush=True)


def stage_detect(videos):
 import torch
 from transformers import AutoProcessor,GroundingDinoForObjectDetection
 from PIL import Image
 sub=json.loads((OUT/'qwen_subjects.json').read_text())
 dev='cuda:4';proc=AutoProcessor.from_pretrained('/data/aic/pretrained/groundingdino_tiny')
 model=GroundingDinoForObjectDetection.from_pretrained('/data/aic/pretrained/groundingdino_tiny',torch_dtype=torch.float32).to(dev).eval()
 det={}
 for n,vid in enumerate(videos):
  qq=json.loads((DENSE/f'{vid}.json').read_text());keys=qq['keyframes'];phrases=sub[vid]['subjects']
  text='. '.join(phrases)+' .' if phrases else 'object .'
  dd={}
  for t in keys:
   im=Image.open(FRAMES/vid/f'{t}.png').convert('RGB')
   inputs=proc(images=im,text=text,return_tensors='pt');inputs={k:v.to(dev) for k,v in inputs.items()}
   with torch.no_grad():outp=model(**inputs)
   res=proc.post_process_grounded_object_detection(outp,inputs['input_ids'],threshold=.25,text_threshold=.25)[0]
   labs=res.get('text_labels') or [model.config.id2label[int(i)] for i in res['labels']]
   by={}
   for ph,sc,box in zip(labs,res['scores'].tolist(),res['boxes'].tolist()):
    ph=ph.strip().lower()
    if ph in phrases:by.setdefault(ph,[]).append([round(sc,4)]+[round(x,1) for x in box])
   dd[t]=by
  det[vid]={'phrases':phrases,'text':text,'per_keyframe':dd}
  print(json.dumps({'vid':vid,'n':n+1,'phrases':phrases,'kf_with_det':sum(1 for v in dd.values() if v)}),flush=True)
 (OUT/'detections.json').write_text(json.dumps(det)+'\n')


def select_points(vid,comp,rk,win=None):
 d=json.loads((OUT/'detections.json').read_text())[vid];q=json.loads((DENSE/f'{vid}.json').read_text())
 keys=q['keyframes'];pts=[None if p is None else list(p) for p in q['ratios'][rk]['points']]
 det=d['per_keyframe'];phrases=d['phrases']
 if not phrases:return pts,{'mode':'no_subjects'}
 cov={ph:0 for ph in phrases};sc={ph:[] for ph in phrases}
 for t in keys:
  for ph,bs in det.get(str(t),{}).items():
   if bs:cov[ph]+=1;sc[ph].append(max(b[0] for b in bs))
 elig=[ph for ph in phrases if cov[ph]>=.6*len(keys)]  # qwen order = importance; coverage gate kills backdrop phrases
 top=elig[0] if elig else None
 if top is None:return pts,{'mode':'no_detection'}
 hit=0
 for i,t in enumerate(keys):
  p=pts[i]
  if p is None or p[2]:continue
  bs=det.get(str(t),{}).get(top)
  if bs:
   b=max(bs,key=lambda x:x[0]);nc=float((b[1]+b[3])/2 if comp==0 else (b[2]+b[4])/2)  # transformers boxes are already normalised
   if win is None or abs(nc-p[comp])<=.75*win:p[comp]=nc;hit+=1
 return pts,{'mode':'grounded','top':top,'coverage':cov[top]/len(keys),'keyframes_replaced':hit}


def stage_eval(videos,sel):
 from aic.max_window_path import geometry,qwen_centres_interp,ema_offsets
 from scripts.max_window_path_eval import load_gt,RATIOS
 from scripts.teacher_diag_eval import frame_iou,boot
 rows=[]
 for vid in videos:
  q=json.loads((DENSE/f'{vid}.json').read_text());keys=q['keyframes'];per={}
  strat='_'.join(sel[vid])
  for r,ratio in RATIOS.items():
   z=np.load(CACHE/f'{vid}_{r}.npz');gt=load_gt(ANN,vid,r);W,H=int(z['W']),int(z['H']);w,h,axis=geometry(W,H,ratio);comp=0 if axis==0 else 1
   win=(w/W) if comp==0 else (h/H)
   reset=z['reset'].astype(bool);raw=z['raw'][:,comp];face=z['chosen']>=0
   def run(p):
    c,_=qwen_centres_interp(p,keys,reset,raw,comp);c=np.where(face,raw,c)
    return float(frame_iou(ema_offsets(c,reset,W,H,ratio),W,H,ratio,gt).mean())
   spt,info=select_points(vid,comp,r,win)
   per[r]={'axis':'x' if comp==0 else 'y','dense_iou':run(q['ratios'][r]['points']),'ground_iou':run(spt),**{f'sel_{k}':v for k,v in info.items() if not isinstance(v,dict)}}
  rows.append({'vid':vid,'stratum':strat,**{r:per[r] for r in RATIOS},
   'dense':float(np.mean([per[r]['dense_iou'] for r in RATIOS])),'ground':float(np.mean([per[r]['ground_iou'] for r in RATIOS])),
   'xchg':any(per[r].get('sel_keyframes_replaced',0)>0 for r in RATIOS)})
 d=np.array([r['ground']-r['dense'] for r in rows]);bs=np.random.default_rng(20260929).choice(d,(5000,len(d))).mean(1)
 out={'videos':len(rows),'mean_dense':float(np.mean([r['dense'] for r in rows])),'mean_ground':float(np.mean([r['ground'] for r in rows])),
  'delta_mean':float(d.mean()),'ci95':[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))],
  'better':int((d>1e-4).sum()),'worse':int((d<-1e-4).sum()),'videos_with_replacement':int(sum(r['xchg'] for r in rows)),
  'by_stratum':{s:{'delta':float(np.mean([r['ground']-r['dense'] for r in rows if r['stratum']==s])),'n':sum(1 for r in rows if r['stratum']==s)} for s in sorted({r['stratum'] for r in rows})},
  'per_video':rows,'official_f_video':None}
 (OUT/'eval.json').write_text(json.dumps(out,indent=1)+'\n')
 print(json.dumps({k:v for k,v in out.items() if k!='per_video'},default=str))


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True,choices=('split','qwen','detect','eval'));ap.add_argument('--n',type=int,default=25)
 ap.add_argument('--videos',nargs='*');a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 sel,strata=load_split(a.n)
 if not (OUT/'strata.json').exists() or a.stage=='split':
  (OUT/'strata.json').write_text(json.dumps({'sel':sel,'strata':strata},indent=1))
  if a.stage=='split':return
 videos=a.videos or sorted(sel)
 if a.stage=='qwen':stage_qwen(videos)
 elif a.stage=='detect':stage_detect(videos)
 else:stage_eval(videos,sel)
if __name__=='__main__':main()
