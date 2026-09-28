"""P1b/P2: fixed B0 path (EMA .25 + B0 shot resets), only the observation changes.

Candidates per frame, all at the max window:
  B0      cached B0 crop (YuNet chosen face, else weighted gradient saliency)
  CENTER  centred window
  COV_m   window maximising prior-weighted coverage of COCO detections + faces,
          fallback to B0's observation unless it beats B0's window by factor m
  QWEN    Qwen3-VL subject point per 1 s keyframe (held within shot), fallback B0
Oracles (diagnostic only, never used for selection): per-shot and per-frame best
of {B0, CENTER, COV}.  m is chosen on dev (001-020); confirm (021-030) once.
"""
import argparse,json
from pathlib import Path
import numpy as np
from aic.max_window_path import geometry,centre_to_offset,to_crops,ema_offsets,coverage_centre,qwen_centres,shots
from scripts.benchmark_spatial import iou
from scripts.max_window_path_eval import boxes,load_gt,RATIOS


def per_frame_iou(c,ratio,gt):
 return iou(boxes(np.asarray(c),ratio)[None],gt).mean(0)


def candidates(z,margins,qwen=None,key=None):
 W,H=int(z['W']),int(z['H']);ratio=z['ratio'].tolist();reset=z['reset'].astype(bool);w,h,axis=geometry(W,H,ratio)
 comp=0 if axis==0 else 1;raw=z['raw'][:,comp];out={'B0':np.asarray(z['b0']),'CENTER':np.asarray(to_crops(W,H,ratio,centre_to_offset(np.full(len(raw),.5),W,H,ratio)))};diag={}
 for m in margins:
  cc=[];src=[]
  for t in range(len(raw)):c,s=coverage_centre(z,t,raw[t],keep_margin=m);cc.append(c);src.append(s)
  out[f'COV_m{m}']=np.asarray(to_crops(W,H,ratio,ema_offsets(np.asarray(cc),reset,W,H,ratio)))
  diag[f'COV_m{m}']={s:src.count(s)/len(src) for s in set(src)}
  # COVNF: keep B0's observation whenever YuNet chose a face; detections only on no-face frames
  nf=np.where(z['chosen']>=0,raw,np.asarray(cc))
  out[f'COVNF_m{m}']=np.asarray(to_crops(W,H,ratio,ema_offsets(nf,reset,W,H,ratio)))
 if qwen is not None and key in qwen['ratios']:
  q=qwen['ratios'][key];qc,src=qwen_centres(q['points'],qwen['keyframes'],reset,raw,comp)
  out['QWEN']=np.asarray(to_crops(W,H,ratio,ema_offsets(qc,reset,W,H,ratio)))
  # QWENNF: keep B0's YuNet observation on face frames; Qwen point only replaces the saliency fallback
  out['QWENNF']=np.asarray(to_crops(W,H,ratio,ema_offsets(np.where(z['chosen']>=0,raw,qc),reset,W,H,ratio)))
  diag['QWEN']={'qwen_frame_rate':float((src=='qwen').mean()),'parse_ok':q['status'].count('ok')/len(q['status']),'uncertain':sum(1 for p in q['points'] if p and p[2])/len(q['points'])}
 # B0 reproduction check: EMA over cached raw must equal cached B0 crop
 rep=np.asarray(to_crops(W,H,ratio,ema_offsets(raw,reset,W,H,ratio)));diag['b0_repro_max_abs']=float(np.abs(rep-out['B0']).max())
 return out,diag


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--annotations',required=True);ap.add_argument('--output',type=Path,required=True)
 ap.add_argument('--qwen',type=Path);ap.add_argument('--margins',default='1.0,1.1,1.3');ap.add_argument('--dev',default='001-020');ap.add_argument('--confirm',default='021-030');a=ap.parse_args()
 rng=lambda s:[f'{i:03d}' for i in range(int(s.split('-')[0]),int(s.split('-')[1])+1)];margins=[float(x) for x in a.margins.split(',')]
 rows=[];diags={};oracle_sets=[['B0','CENTER'],['B0','COV'],['B0','CENTER','COV']]
 for split,vids in (('dev',rng(a.dev)),('confirm',rng(a.confirm))):
  for vid in vids:
   qf=a.qwen/f'{vid}.json' if a.qwen else None;qwen=json.loads(qf.read_text()) if qf and qf.exists() else None
   for r,ratio in RATIOS.items():
    z=np.load(a.cache/f'{vid}_{r}.npz');gt=load_gt(a.annotations,vid,r);c,dg=candidates(z,margins,qwen,r);diags[f'{vid}_{r}']=dg
    pf={k:per_frame_iou(v,ratio,gt) for k,v in c.items()}
    for k,v in pf.items():rows.append({'split':split,'video_id':vid,'ratio':r,'policy':k,'iou':float(v.mean())})
    reset=z['reset'].astype(bool)
    for m in margins:
     for S in oracle_sets:
      keys=[k if k!='COV' else f'COV_m{m}' for k in S];st=np.stack([pf[k] for k in keys])
      shot=np.concatenate([st[:,a_:b_].mean(1).max()*np.ones(b_-a_) for a_,b_ in shots(reset)])
      tag='+'.join(S)+f'_m{m}' if 'COV' in S else '+'.join(S)
      rows.append({'split':split,'video_id':vid,'ratio':r,'policy':f'ORACLE_SHOT[{tag}]','iou':float(shot.mean())})
      rows.append({'split':split,'video_id':vid,'ratio':r,'policy':f'ORACLE_FRAME[{tag}]','iou':float(st.max(0).mean())})
 rows=[dict(t) for t in {tuple(sorted(x.items())) for x in rows}]
 names=sorted({x['policy'] for x in rows});summ={}
 for split in ('dev','confirm'):
  for n in names:
   v=[x for x in rows if x['split']==split and x['policy']==n]
   if v:summ[f'{split}:{n}']={'iou':float(np.mean([x['iou'] for x in v])),**{r:float(np.mean([x['iou'] for x in v if x['ratio']==r])) for r in RATIOS},'n':len(v)}
 def paired(split,n,base='B0'):
  bx={(y['video_id'],y['ratio']):y['iou'] for y in rows if y['split']==split and y['policy']==base}
  pv={}
  for x in rows:
   if x['split']==split and x['policy']==n:pv.setdefault(x['video_id'],[]).append(x['iou']-bx[(x['video_id'],x['ratio'])])
  if not pv:return None
  d=np.concatenate(list(pv.values()));dv=np.array([np.mean(v) for v in pv.values()]);b=np.random.default_rng(20260928);bs=[b.choice(dv,len(dv)).mean() for _ in range(5000)]
  return {'mean':float(d.mean()),'ci95_video':[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))],'improved_pairs':int((d>1e-4).sum()),'worse_pairs':int((d<-1e-4).sum()),'pairs':len(d),'worst':float(d.min()),'best':float(d.max())}
 covs=[f'{p}_m{m}' for p in ('COV','COVNF') for m in margins];chosen=max(covs,key=lambda n:summ[f'dev:{n}']['iou'])
 out={'protocol':'RetargetVid native IoU, max window, B0 EMA path fixed; only observation changes','chosen_cov_on_dev':chosen,'summary':summ,
  'paired_vs_B0':{f'{s}:{n}':paired(s,n) for s in ('dev','confirm') for n in names if n!='B0'},'diagnostics':diags,'per_video':sorted(rows,key=lambda x:(x['split'],x['video_id'],x['ratio'],x['policy'])),'official_f_video':None}
 groups={}
 for split in ('dev','confirm'):
  for g,cond in (('face_rate>=.5',lambda f:f>=.5),('face_rate<.5',lambda f:f<.5)):
   keys={k for k,d in diags.items() if cond(np.load(a.cache/f'{k}.npz')['chosen'].__ge__(0).mean())}
   for n in ('CENTER',chosen,'COV_m1.1','COVNF_m1.1','QWEN','QWENNF'):
    d=[x['iou']-y['iou'] for x in rows for y in rows if x['split']==y['split']==split and x['policy']==n and y['policy']=='B0' and x['video_id']==y['video_id'] and x['ratio']==y['ratio'] and f"{x['video_id']}_{x['ratio']}" in keys]
    if d:groups[f'{split}:{g}:{n}-B0']={'mean':float(np.mean(d)),'pairs':len(d),'improved':int(sum(v>1e-4 for v in d))}
 pooled={}
 for n in names:
  if n=='B0' or n.startswith('ORACLE'):continue
  pv={}
  for x in rows:
   if x['policy']==n:
    b=next(y['iou'] for y in rows if y['policy']=='B0' and y['split']==x['split'] and y['video_id']==x['video_id'] and y['ratio']==x['ratio']);pv.setdefault(x['video_id'],[]).append(x['iou']-b)
  dv=np.array([np.mean(v) for v in pv.values()]);bb=np.random.default_rng(20260928);bs=[bb.choice(dv,len(dv)).mean() for _ in range(5000)]
  pooled[n]={'videos':len(dv),'mean':float(dv.mean()),'ci95_video':[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))],'improved_videos':int((dv>1e-4).sum()),'worse_videos':int((dv<-1e-4).sum())}
 out['pooled_30_vs_B0']=pooled
 out['groups']=groups
 a.output.mkdir(parents=True,exist_ok=True);(a.output/'metrics.json').write_text(json.dumps(out,indent=1)+'\n')
 for s in ('dev','confirm'):
  for n in ['B0','CENTER',chosen,'COV_m1.1','COVNF_m1.1','QWEN',f'ORACLE_SHOT[B0+CENTER]',f'ORACLE_SHOT[B0+COV_m{chosen[5:]}]',f'ORACLE_SHOT[B0+CENTER+COV_m{chosen[5:]}]']:
   if f'{s}:{n}' in summ:print(s,n,json.dumps({k:round(v,4) for k,v in summ[f'{s}:{n}'].items()}),json.dumps(out['paired_vs_B0'].get(f'{s}:{n}')))
 print(json.dumps(pooled));print('chosen',chosen,'b0_repro_max',max(d['b0_repro_max_abs'] for d in diags.values()))
if __name__=='__main__':main()
