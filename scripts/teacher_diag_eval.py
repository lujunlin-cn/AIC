"""T0/T1/T2 on RetargetVid human crops: where does the 32B teacher lose IoU?

Same protocol as scripts/max_window_candidates_eval.py (rounded boxes,
inclusive pixels, mean over 6 annotators, max legal window, B0 EMA path and
shot resets).  Splits (frozen before any T1/T2 reply was read):
  used     001-030  (dev/confirm of the previous round, now diagnostic only)
  dev2     031-100
  confirm2 601-700  (first use; evaluated once with the pre-declared primaries)
Policies (N0 = platform 42.83 recipe on this data):
  B0, CENTER, P0 (Qwen point everywhere), N0 (point on no-face frames),
  N0_NOEMA (diagnostic), REGION*_{fit,subject,context} (T1), DENSE*/DENSENF (T2)
Oracles (public GT only, never on the official set): per-frame / per-shot /
per-video best of a 65-position legal grid (129 for the discretisation check),
and per-frame / per-shot best of the candidates {B0, N0, P0, REGIONNF_fit}.
Keyframe decomposition (frames where the policy actually consumes a Qwen reply):
  subject_miss  point outside the GT window of >= half the annotators
  placement     point inside, but the window centred on it trails the grid
                oracle at that frame by > .1 IoU
  ok            otherwise
Propagation: at the dense-only keyframes (0.5 s after a 1 s keyframe in the
same shot) compare the held 1 s point with the fresh reply, no EMA.
"""
import argparse,json
from pathlib import Path
import numpy as np
from aic.max_window_path import geometry,centre_to_offset,ema_offsets,qwen_centres,region_points,shots
from scripts.benchmark_spatial import iou
from scripts.max_window_path_eval import load_gt,RATIOS

SPLITS={'used':[f'{i:03d}' for i in range(1,31)],'dev2':[f'{i:03d}' for i in range(31,101)],'confirm2':[str(i) for i in range(601,701)]}
PRIMARY={'T1':'REGIONNF_fit','T2':'DENSENF'}


def win_boxes(off,W,H,ratio):
 w,h,axis=geometry(W,H,ratio);off=np.asarray(off,float)
 x=off if axis==0 else np.zeros_like(off);y=off if axis==1 else np.zeros_like(off)
 return np.maximum(np.rint(np.column_stack([x,y,x+w,y+h])).astype(int),0)


def frame_iou(off,W,H,ratio,gt):
 return iou(win_boxes(off,W,H,ratio)[None],gt).mean(0)


def boot(dv,seed=20260928,n=5000):
 dv=np.asarray(dv,float);b=np.random.default_rng(seed);bs=b.choice(dv,(n,len(dv))).mean(1)
 return [float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]


def video_eval(vid,r,ratio,z,gt,q1,q2,rg):
 W,H=int(z['W']),int(z['H']);w,h,axis=geometry(W,H,ratio);comp=0 if axis==0 else 1
 L,s=(W,w) if axis==0 else (H,h);win=s/L;reset=z['reset'].astype(bool);face=z['chosen']>=0;raw=z['raw'][:,comp];n=len(raw)
 ema=lambda c:ema_offsets(np.asarray(c,float),reset,W,H,ratio)
 off={'B0':np.asarray(z['b0'])[:,comp],'CENTER':np.full(n,(L-s)/2)}
 srcs={}
 def add(name,pts,keys):
  c,src=qwen_centres(pts,keys,reset,raw,comp);srcs[name]=src
  off[name.replace('@','')]=ema(c);off[name.replace('@','NF')]=ema(np.where(face,raw,c))
  return c
 c1=add('P0@',q1['ratios'][r]['points'],q1['keyframes']);off['N0']=off.pop('P0NF')
 off['N0_NOEMA']=centre_to_offset(np.where(face,raw,c1),W,H,ratio)
 add('DENSE@',q2['ratios'][r]['points'],q2['keyframes'])
 if rg is not None:
  for m in ('fit','subject','context'):add(f'REGION@_{m}',region_points(rg['ratios'][r]['regions'],comp,win,m),rg['keyframes'])
 pf={k:frame_iou(v,W,H,ratio,gt) for k,v in off.items()}
 if np.abs(np.asarray(z['b0'])[:,comp]-ema(raw)).max()>1e-6:raise ValueError('B0 replay mismatch '+vid)
 # legal grids
 G={}
 for k in (65,129):
  g=np.linspace(0,L-s,k);G[k]=np.stack([frame_iou(np.full(n,o),W,H,ratio,gt) for o in g])
 sh=shots(reset);res={}
 for k,M in G.items():
  res[f'ORACLE_FRAME_GRID{k}']=M.max(0)
  res[f'ORACLE_SHOT_GRID{k}']=np.concatenate([np.full(b-a,M[:,a:b].mean(1).max()) for a,b in sh])
  res[f'ORACLE_VIDEO_GRID{k}']=np.full(n,M.mean(1).max())
 cand=['B0','N0','P0']+(['REGIONNF_fit'] if rg is not None else []);st=np.stack([pf[c] for c in cand])
 res['ORACLE_FRAME_CAND']=st.max(0);res['ORACLE_SHOT_CAND']=np.concatenate([np.full(b-a,st[:,a:b].mean(1).max()) for a,b in sh])
 pf.update(res)
 # keyframe decomposition for the point teacher (P0 consumes every valid reply; N0 only on no-face frames)
 gl=gt[...,comp];gh=gt[...,comp+2];kf=[];orc=G[65].max(0)
 def keyrows(tag,keys,cents,extra=None):
  for j,(t,c) in enumerate(zip(keys,cents)):
   if c is None:continue
   o=float(centre_to_offset(np.array([c]),W,H,ratio)[0]);v=float(frame_iou(np.array([o]),W,H,ratio,gt[:,t:t+1])[0]);px=c*L
   inside=float(((gl[:,t]<=px)&(px<=gh[:,t])).mean());gap=float(orc[t]-v)
   cls='subject_miss' if inside<.5 else ('placement' if gap>.1 else 'ok')
   kf.append({'tag':tag,'frame':int(t),'face':bool(face[t]),'iou':v,'oracle':float(orc[t]),'inside':inside,'class':cls,**(extra[j] if extra else {})})
 pts=q1['ratios'][r]['points'];keyrows('point',q1['keyframes'],[None if p is None or p[2] else p[comp] for p in pts])
 if rg is not None:
  rp=region_points(rg['ratios'][r]['regions'],comp,win,'fit');regs=rg['ratios'][r]['regions']
  ex=[{'context_fits':bool(x is not None and x[1][comp+2]-x[1][comp]<=win+1e-9),'subject_fits':bool(x is not None and x[0][comp+2]-x[0][comp]<=win+1e-9)} for x in regs]
  keyrows('region_fit',rg['keyframes'],[None if p is None or p[2] else p[comp] for p in rp],ex)
 # propagation at dense-only keyframes: held 1 s reply vs fresh reply, same shot, no EMA
 prop=[];k1=q1['keyframes'];p1=q1['ratios'][r]['points'];shot_id=np.cumsum(reset)
 for t,p,reu in zip(q2['keyframes'],q2['ratios'][r]['points'],q2['ratios'][r].get('reused',[True]*len(q2['keyframes']))):
  if reu or t in set(k1):continue
  j=max(i for i,k in enumerate(k1) if k<t);prev=p1[j]
  if shot_id[k1[j]]!=shot_id[t] or prev is None or prev[2] or p is None or p[2]:continue
  oh,of=centre_to_offset(np.array([prev[comp],p[comp]]),W,H,ratio);ih,i_f=frame_iou(np.array([oh,of]),W,H,ratio,np.repeat(gt[:,t:t+1],2,1))
  prop.append({'frame':int(t),'face':bool(face[t]),'held':float(ih),'fresh':float(i_f),'shift_win':float(abs(prev[comp]-p[comp])/win)})
 # per-frame context for N0 errors
 is_key=np.zeros(n,bool);is_key[[k for k in k1 if k<n]]=True
 near_cut=np.zeros(n,bool)
 for a,b in sh[1:]:near_cut[a:min(b,a+15)]=True
 gc=((gl+gh)/2).mean(0)/L;motion=float(np.abs(np.diff(gc))[~reset[1:]].mean()) if n>1 else 0.
 ag=[]
 for i in range(6):
  for j in range(i+1,6):ag.append(iou(gt[i],gt[j]).mean())
 det=z['det'];persons=[int(((det[z['det_off'][t]:z['det_off'][t+1],4]>=.5)&(det[z['det_off'][t]:z['det_off'][t+1],5]==1)).sum()) for t in range(n)]
 meta={'frames':n,'axis':'x' if axis==0 else 'y','face_rate':float(face.mean()),'shots':len(sh),'gt_motion':motion,'annotator_agreement':float(np.mean(ag)),
  'persons_median':float(np.median(persons)),'qwen_nf_frame_rate':float(((srcs['P0@']=='qwen')&~face).mean()),
  'n0_gap_key_nf':float((orc-pf['N0'])[is_key&~face].mean()) if (is_key&~face).any() else None,'n0_gap_nonkey_nf':float((orc-pf['N0'])[~is_key&~face].mean()) if (~is_key&~face).any() else None,
  'n0_gap_near_cut':float((orc-pf['N0'])[near_cut].mean()) if near_cut.any() else None,'n0_gap_far_cut':float((orc-pf['N0'])[~near_cut].mean()),
  'n0_iou_face':float(pf['N0'][face].mean()) if face.any() else None,'p0_iou_face':float(pf['P0'][face].mean()) if face.any() else None,
  'n0_iou_nf':float(pf['N0'][~face].mean()) if (~face).any() else None,'b0_iou_nf':float(pf['B0'][~face].mean()) if (~face).any() else None,'face_frames':int(face.sum())}
 return {k:float(v.mean()) for k,v in pf.items()},meta,kf,prop


def strata(m,thr):
 return {'axis':m['axis'],'face':'none' if m['face_rate']==0 else ('low<.5' if m['face_rate']<.5 else 'high>=.5'),
  'motion':'fast' if m['gt_motion']>thr['motion'] else 'slow','cuts':'single_shot' if m['shots']==1 else 'multi_shot',
  'persons':'0' if m['persons_median']<.5 else ('1' if m['persons_median']<1.5 else '2+'),'agreement':'low' if m['annotator_agreement']<thr['agreement'] else 'high'}


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--annotations',type=Path,required=True)
 ap.add_argument('--point',type=Path,required=True);ap.add_argument('--dense',type=Path,required=True);ap.add_argument('--region',type=Path)
 ap.add_argument('--output',type=Path,required=True);ap.add_argument('--splits',default=','.join(SPLITS));a=ap.parse_args()
 for k in [k for k in SPLITS if k not in a.splits.split(',')]:del SPLITS[k]
 rows=[];metas={};kfs=[];props=[]
 for split,vids in SPLITS.items():
  for vid in vids:
   q1=json.loads((a.point/f'{vid}.json').read_text());q2=json.loads((a.dense/f'{vid}.json').read_text())
   rf=a.region/f'{vid}.json' if a.region else None;rg=json.loads(rf.read_text()) if rf and rf.exists() else None
   for r,ratio in RATIOS.items():
    z=np.load(a.cache/f'{vid}_{r}.npz');gt=load_gt(a.annotations,vid,r)
    if gt.shape[1]!=len(z['b0']):raise ValueError(f'{vid} frame mismatch')
    v,m,kf,pr=video_eval(vid,r,ratio,z,gt,q1,q2,rg);metas[f'{vid}_{r}']={**m,'split':split}
    rows+=[{'split':split,'video_id':vid,'ratio':r,'policy':k,'iou':x} for k,x in v.items()]
    kfs+=[{'split':split,'video_id':vid,'ratio':r,**x} for x in kf];props+=[{'split':split,'video_id':vid,'ratio':r,**x} for x in pr]
 tab={(x['split'],x['video_id'],x['ratio'],x['policy']):x['iou'] for x in rows};pols=sorted({x['policy'] for x in rows})
 def per_video(split,p,base=None,keys=None):
  out={}
  for vid in SPLITS[split]:
   d=[tab[(split,vid,r,p)]-(tab[(split,vid,r,base)] if base else 0) for r in RATIOS if (split,vid,r,p) in tab and (keys is None or f'{vid}_{r}' in keys)]
   if d:out[vid]=float(np.mean(d))
  return out
 def paired(split,p,base,keys=None):
  pv=per_video(split,p,base,keys)
  if not pv:return None
  dv=np.array(list(pv.values()));return {'mean':float(dv.mean()),'ci95_video':boot(dv),'better':int((dv>1e-4).sum()),'worse':int((dv<-1e-4).sum()),'videos':len(dv),'worst':float(dv.min())}
 summ={f'{s}:{p}':{'iou':float(np.mean(list(per_video(s,p).values()))),**{r:float(np.mean([tab[(s,v,r,p)] for v in SPLITS[s] if (s,v,r,p) in tab])) for r in RATIOS},
  'worst_quartile':float(np.percentile(list(per_video(s,p).values()),25)),'videos':len(per_video(s,p))} for s in SPLITS for p in pols if per_video(s,p)}
 comps=[('N0','B0'),('P0','B0'),('N0','P0'),('REGIONNF_fit','N0'),('REGION_fit','P0'),('REGIONNF_subject','N0'),('REGIONNF_context','N0'),('DENSENF','N0'),('DENSE','P0'),('N0_NOEMA','N0'),('CENTER','B0')]
 pair={f'{s}:{p}-{b}':paired(s,p,b) for s in SPLITS for p,b in comps}
 # stratification thresholds fixed on the used split
 used=[m for m in metas.values() if m['split']=='used'];thr={'motion':float(np.median([m['gt_motion'] for m in used])),'agreement':float(np.median([m['annotator_agreement'] for m in used]))}
 st={}
 for s in ('dev2','confirm2','used'):
  for dim in ('axis','face','motion','cuts','persons','agreement'):
   for val in sorted({strata(m,thr)[dim] for k,m in metas.items() if m['split']==s}):
    keys={k for k,m in metas.items() if m['split']==s and strata(m,thr)[dim]==val}
    for p,b in comps[:5]+[('DENSENF','N0')]:
     res=paired(s,p,b,keys)
     if res:st[f'{s}:{dim}={val}:{p}-{b}']={'pairs':len(keys),**res}
 # keyframe decomposition + propagation
 dec={}
 for s in SPLITS:
  for tag in ('point','region_fit'):
   for fc in (None,False,True):
    ks=[x for x in kfs if x['split']==s and x['tag']==tag and (fc is None or x['face']==fc)]
    if ks:dec[f'{s}:{tag}:{"all" if fc is None else ("face" if fc else "noface")}']={'n':len(ks),**{c:float(np.mean([x['class']==c for x in ks])) for c in ('subject_miss','placement','ok')},
     'iou':float(np.mean([x['iou'] for x in ks])),'oracle':float(np.mean([x['oracle'] for x in ks])),'iou_given_ok':float(np.mean([x['iou'] for x in ks if x['class']=='ok'] or [np.nan]))}
 pro={}
 for s in SPLITS:
  ps=[x for x in props if x['split']==s]
  if ps:
   sh=np.array([x['shift_win'] for x in ps]);pro[s]={'n':len(ps),'held_iou':float(np.mean([x['held'] for x in ps])),'fresh_iou':float(np.mean([x['fresh'] for x in ps])),
    'shift_win_median':float(np.median(sh)),'shift_gt_quarter_window':float((sh>.25).mean()),'fresh_better_by_.1':float(np.mean([x['fresh']-x['held']>.1 for x in ps])),'held_better_by_.1':float(np.mean([x['held']-x['fresh']>.1 for x in ps]))}
 gaps={s:{k:float(np.nanmean([m[k] if m[k] is not None else np.nan for m in metas.values() if m['split']==s])) for k in ('n0_gap_key_nf','n0_gap_nonkey_nf','n0_gap_near_cut','n0_gap_far_cut')} for s in SPLITS}
 face_split={s:{k:float(np.nanmean([m[k] if m[k] is not None else np.nan for m in metas.values() if m['split']==s])) for k in ('n0_iou_face','p0_iou_face','n0_iou_nf','b0_iou_nf')} for s in SPLITS}
 cost={}
 for s in SPLITS:
  q1=[json.loads((a.point/f'{v}.json').read_text()) for v in SPLITS[s]];q2=[json.loads((a.dense/f'{v}.json').read_text()) for v in SPLITS[s]]
  cost[s]={'keyframes_point':sum(len(x['keyframes']) for x in q1),'keyframes_dense':sum(len(x['keyframes']) for x in q2),'frames':sum(metas[f'{v}_1-3']['frames'] for v in SPLITS[s])}
 out={'protocol':'RetargetVid native IoU (6 annotators), max legal window, B0 EMA/resets; splits used=001-030, dev2=031-100, confirm2=601-700','primary':PRIMARY,'strata_thresholds':thr,
  'summary':summ,'paired':pair,'strata':st,'keyframe_decomposition':dec,'propagation_dense_vs_held':pro,'n0_gap_by_position':gaps,'face_vs_noface_iou':face_split,'cost':cost,
  'video_meta':metas,'per_video':rows,'official_f_video':None}
 a.output.mkdir(parents=True,exist_ok=True);(a.output/'metrics.json').write_text(json.dumps(out,indent=1)+'\n')
 (a.output/'keyframes.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in kfs))
 for s in SPLITS:
  for p in ('B0','CENTER','P0','N0','REGIONNF_fit','DENSENF','ORACLE_SHOT_CAND','ORACLE_FRAME_GRID65','ORACLE_SHOT_GRID65','ORACLE_VIDEO_GRID65','ORACLE_FRAME_GRID129'):
   if f'{s}:{p}' in summ:print(s,p,json.dumps({k:round(v,4) for k,v in summ[f'{s}:{p}'].items()}))
  for p,b in comps:
   if pair.get(f'{s}:{p}-{b}'):print(' ',s,f'{p}-{b}',json.dumps({k:(round(v,4) if isinstance(v,float) else v) for k,v in pair[f'{s}:{p}-{b}'].items()}))
 print(json.dumps(dec,indent=0));print(json.dumps(pro));print(json.dumps(gaps));print(json.dumps(face_split));print(json.dumps(cost))
if __name__=='__main__':main()
