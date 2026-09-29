"""V5 H3 step 3: G / V / V+Q candidate scorers on frozen DINOv2 features.

Prereg configs/V5_H3_VISUAL_PREREG.json.  Same splits, candidates, targets and
pipeline for all three heads; only the input features differ:
  G    26 H2-style geometry/detector/saliency/point-distance features
  V    base geometry (7) + frozen-DINOv2 pooled candidate features (12x768->256)
  V+Q  26 H2 features + the same visual projection
Training: Huber(0.25) on mean-annotator IoU, dataset-balanced batches, AdamW
3e-4, wd 1e-3/0, warmup 5% cosine, clip 1.0, <=2000 updates, best by dev
keyframe IoU (rv_dev+live_dev dataset-equal).  Evaluation: keyframe IoU
(mother / G / V / V+Q / candidate oracle, same candidate set) and the full
INTERP pipeline (qwen_centres_interp, NOFACE gate, B0 EMA) per split.
"""
import argparse,hashlib,json,math,pickle,time
from pathlib import Path
import numpy as np

BASE=('rel_pos','win','axis','log_aspect','ds_rv','ds_live','is_mother')


def load_all(tables,visual,names):
 out={}
 for name in names:
  units=pickle.loads((tables/f'{name}.pkl').read_bytes())
  for ui,u in enumerate(units):
   z=np.load(visual/f'{name}_{ui:04d}.npz')
   K=len(u['J']);V=np.zeros((K,u['offs'].shape[1],12,768),np.float16)
   rows=z['rows']
   if len(rows):V[rows]=z['pooled']
   u['V']=V.reshape(K,u['offs'].shape[1],-1)
  out[name]=units
 return out


def unit_tensors(u,dev,mu,sd,proj,kind):
 import torch
 K=len(u['J']);C=u['offs'].shape[1]
 gi=torch.tensor(u['X'],dtype=torch.float32,device=dev) if kind!='V' else None  # V6 E0: V reads no X (official units may lack it; identical numerics when X exists)
 base=np.zeros((K,C,len(BASE)),np.float32)
 offn=u['offs']/max(u['L']-u['s'],1e-9)
 base[...,0]=offn;base[...,1]=u['s']/u['L'];base[...,2]=float(u['comp']);base[...,3]=math.log(u['W']/u['H'])
 base[...,4]=1. if u['ds']=='rv' else 0.;base[...,5]=1. if u['ds']=='live' else 0.;base[...,6]=u['is_mother'].astype(np.float32)
 bt=torch.tensor(base,dtype=torch.float32,device=dev)
 vt=torch.tensor(u['V'].astype(np.float32),dtype=torch.float32,device=dev)
 if kind=='G':x=(gi-mu)/sd
 elif kind=='V':x=torch.cat([bt,proj(vt)],-1)
 else:x=torch.cat([(gi-mu)/sd,proj(vt)],-1)
 y=torch.tensor(u['Y'],dtype=torch.float32,device=dev)
 return x,y


def make_head(kind,dev):
 import torch.nn as nn
 proj=nn.Linear(12*768,256).to(dev) if kind in ('V','VQ') else None
 nin=(7 if kind=='V' else 26)+(256 if proj is not None else 0)
 net=nn.Sequential(nn.Linear(nin,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,1)).to(dev)
 return net,proj


def params_of(net,proj):return list(net.parameters())+(list(proj.parameters()) if proj is not None else [])


def dev_keyframe_iou(net,proj,kind,units,dev,mu,sd):
 import torch
 per={}
 for u in units:
  if not u['has_gt'].any():continue
  x,y=unit_tensors(u,dev,mu,sd,proj,kind)
  with torch.no_grad():s=net(x).squeeze(-1)
  m=torch.tensor(u['has_gt'],device=dev);k=int(m.sum());idx=s[m].argmax(1)
  per.setdefault(u['ds'],[]).append(float(y[m][torch.arange(k,device=dev),idx].mean()))
 return float(np.mean([np.mean(v) for v in per.values()])),{k2:round(float(np.mean(v)),4) for k2,v in per.items()}


def train_head(kind,train,devu,dev,seed,steps,log=print):
 import torch,torch.nn as nn
 torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed)
 allx=np.concatenate([u['X'].reshape(-1,26) for u in train if u['has_gt'].any()])
 mu=torch.tensor(allx.mean(0),dtype=torch.float32,device=dev);sd=torch.tensor(allx.std(0),dtype=torch.float32,device=dev);sd[sd<1e-6]=1.
 net,proj=make_head(kind,dev);params=params_of(net,proj)
 opt=torch.optim.AdamW([{'params':[p for p in params if p.ndim==2],'weight_decay':1e-3},{'params':[p for p in params if p.ndim<2],'weight_decay':0.}],lr=3e-4)
 warm=max(1,int(.05*steps));sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:(s+1)/warm if s<warm else .5*(1+math.cos(math.pi*(s-warm)/max(1,steps-warm))))
 pools={ds:[u for u in train if u['ds']==ds and u['has_gt'].any()] for ds in ('rv','live')};pools={k:v for k,v in pools.items() if v}
 per_ds=max(1,8//len(pools));best=(-1,None,None,None);hist=[];npar=sum(int(p.numel()) for p in params)
 for step in range(steps):
  xs=[];ys=[]
  for pool in pools.values():
   for ui in rng.integers(0,len(pool),per_ds):
    u=pool[ui];ok=np.flatnonzero(u['has_gt']);kk=rng.choice(ok,8,replace=len(ok)<8)
    x,y=unit_tensors(u,dev,mu,sd,proj,kind);xs.append(x[kk]);ys.append(y[torch.tensor(kk,device=dev)])
  x=torch.stack(xs);y=torch.stack(ys)
  loss=torch.nn.functional.huber_loss(net(x).squeeze(-1),y,delta=.25)
  opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(params,1.);opt.step();sch.step()
  if (step+1)%250==0:
   d,per=dev_keyframe_iou(net,proj,kind,devu,dev,mu,sd)
   hist.append({'step':step+1,'loss':round(float(loss.detach()),4),'dev_keyframe_iou':round(d,4),'per_ds':per});log(json.dumps({'head':kind,**hist[-1]}))
   if d>best[0]:best=(d,step+1,{k:v.detach().clone() for k,v in net.state_dict().items()},({k:v.detach().clone() for k,v in proj.state_dict().items()} if proj is not None else None))
 net.load_state_dict(best[2])
 if proj is not None:proj.load_state_dict(best[3])
 info={'best_step':best[1],'best_dev_keyframe_iou':best[0],'params':int(npar),'history':hist}
 return net,proj,(mu,sd),info


def score_unit(net,proj,kind,u,dev,mu,sd):
 import torch
 x,_=unit_tensors(u,dev,mu,sd,proj,kind)
 with torch.no_grad():return net(x).squeeze(-1).cpu().numpy()


def keyframe_eval(units,scores):
 per=[]
 for u,s in zip(units,scores):
  m=u['has_gt']
  if not m.any():continue
  Y=u['Y'][m];sel=s[m];moidx=u['is_mother'][m].argmax(1);si=sel.argmax(1)
  per.append({'ds':u['ds'],'axis':int(u['comp']),'vid':u['vid'],
   'mother':float(Y[np.arange(len(Y)),moidx].mean()),'selector':float(Y[np.arange(len(Y)),si].mean()),
   'oracle':float(Y.max(1).mean()),'changed_frac':float((si!=moidx).mean())})
 def agg(rows):
  if not rows:return None
  return {'units':len(rows),'mother':float(np.mean([r['mother'] for r in rows])),'selector':float(np.mean([r['selector'] for r in rows])),
   'oracle':float(np.mean([r['oracle'] for r in rows])),'regret':float(np.mean([r['oracle']-r['selector'] for r in rows])),
   'sel_minus_mother':float(np.mean([r['selector']-r['mother'] for r in rows])),'changed_frac':float(np.mean([r['changed_frac'] for r in rows]))}
 out={};groups={}
 for r in per:groups.setdefault((r['ds'],r['axis']),[]).append(r)
 for (ds,ax),rows in groups.items():out[f'{ds}:axis{ax}']=agg(rows)
 for ds in ('rv','live'):
  rows=[r for r in per if r['ds']==ds]
  if rows:out[f'{ds}:all']=agg(rows)
 out['ALL']=agg(per)
 return out,per


def pipeline_eval(units,scores):
 from aic.max_window_path import qwen_centres_interp,ema_offsets
 from scripts.teacher_diag_eval import frame_iou
 per={}
 for i,u in enumerate(units):
  s=scores[i];pts=[list(p) if p is not None else None for p in u['pts']]
  for k,j in enumerate(u['J']):pts[j][u['comp']]=float((u['offs'][k,s[k].argmax()]+u['s']/2)/u['L'])
  def run(p):
   c,_=qwen_centres_interp(p,u['keys'],u['reset'],u['raw'],u['comp']);c=np.where(u['face'],u['raw'],c)
   off=ema_offsets(c,u['reset'],u['W'],u['H'],u['ratio'])[u['gf']]
   return float(frame_iou(off,u['W'],u['H'],u['ratio'],u['gt']).mean())
  a=run([list(p) if p is not None else None for p in u['pts']]);b=run(pts)
  per.setdefault((u['ds'],int(u['comp'])),{}).setdefault(u['vid'],[]).append((a,b))
 vids={k:{v:float(np.mean(np.array(x),0)[1]-np.mean(np.array(x),0)[0]) for v,x in d.items()} for k,d in per.items()}
 def agg(sel):
  d=np.array([v for k in vids for v in vids[k].values() if sel(k)])
  if not len(d):return None
  bs=np.random.default_rng(20260929).choice(d,(5000,len(d))).mean(1)
  return {'videos':len(d),'mean':float(d.mean()),'ci95':[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))],'better':int((d>1e-4).sum()),'worse':int((d<-1e-4).sum()),'worst':float(d.min())}
 return {'all':agg(lambda k:True),'axis0':agg(lambda k:k[1]==0),'axis1':agg(lambda k:k[1]==1),
  'by_group':{f'{k[0]}:axis{k[1]}':agg(lambda q,qk=k:q==qk) for k in vids}}


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tables',type=Path,default=Path('/data/aic/experiments/V5_H3_VISUAL/tables'))
 ap.add_argument('--visual',type=Path,default=Path('/data/aic/experiments/V5_H3_VISUAL/features'));ap.add_argument('--prereg',default='configs/V5_H3_VISUAL_PREREG.json')
 ap.add_argument('--output',type=Path,required=True);ap.add_argument('--seed',type=int,default=1);ap.add_argument('--steps',type=int,default=2000)
 ap.add_argument('--cuda',default='cuda:2');a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 import torch
 dev=a.cuda if torch.cuda.is_available() else 'cpu'
 logf=open(a.output/'train.log','a');log=lambda s:(print(s,flush=True),logf.write(s+'\n'),logf.flush())
 log(json.dumps({'prereg_sha256':hashlib.sha256(Path(a.prereg).read_bytes()).hexdigest(),'seed':a.seed,'steps':a.steps}))
 D=load_all(a.tables,a.visual,['rv_train','rv_dev','rv_confirm2','live_train','live_dev','live_val'])
 train=D['rv_train']+D['live_train'];devu=D['rv_dev']+D['live_dev']
 log(json.dumps({'train_units':len(train),'dev_units':len(devu),'train_kf':int(sum(u['has_gt'].sum() for u in train))}))
 heads={}
 for kind in ('G','V','VQ'):
  t0=time.time();net,proj,norm,info=train_head(kind,train,devu,dev,a.seed,a.steps,log);info['seconds']=round(time.time()-t0,1)
  heads[kind]=(net,proj,norm,info);log(json.dumps({'head':kind,'trained':{'best_step':info['best_step'],'best_dev_keyframe_iou':info['best_dev_keyframe_iou'],'params':info['params'],'s':info['seconds']}}))
 res={'seed':a.seed,'steps':a.steps,'prereg':a.prereg,'prereg_sha256':hashlib.sha256(Path(a.prereg).read_bytes()).hexdigest(),
  'heads':{k:v[3] for k,v in heads.items()},'keyframe':{},'pipeline':{}}
 csv=['split,head,group,units,mother,selector,oracle,regret,sel_minus_mother,changed_frac']
 for name in ('rv_dev','rv_confirm2','live_dev','live_val'):
  units=D[name];res['keyframe'][name]={}
  for k in heads:
   s=[score_unit(heads[k][0],heads[k][1],k,u,dev,*heads[k][2]) for u in units]
   out,rows=keyframe_eval(units,s);res['keyframe'][name][k]=out
   for g,v in out.items():
    csv.append(f"{name},{k},{g},{v['units']},{v['mother']:.4f},{v['selector']:.4f},{v['oracle']:.4f},{v['regret']:.4f},{v['sel_minus_mother']:.4f},{v['changed_frac']:.4f}")
  log(json.dumps({'keyframe':name,'G_ALL':res['keyframe'][name]['G']['ALL'],'VQ_ALL':res['keyframe'][name]['VQ']['ALL']}))
 for name in ('rv_dev','rv_confirm2','live_dev','live_val'):
  units=D[name];res['pipeline'][name]={}
  for k in heads:
   s=[score_unit(heads[k][0],heads[k][1],k,u,dev,*heads[k][2]) for u in units]
   res['pipeline'][name][k]=pipeline_eval(units,s)
   log(json.dumps({'pipeline':name,'head':k,'all':res['pipeline'][name][k]['all'],'axis0':res['pipeline'][name][k]['axis0'],'axis1':res['pipeline'][name][k]['axis1']}))
 # prereg decision
 def dev_equal(kind):
  rv=res['keyframe']['rv_dev'][kind];lv=res['keyframe']['live_dev'][kind]
  if rv is None or lv is None or rv.get('rv:all') is None or lv.get('live:all') is None:return None
  return (rv['rv:all']['selector']+lv['live:all']['selector'])/2
 deveq={k:dev_equal(k) for k in ('G','V','VQ')}
 gate1={'dev_equal':deveq,'best_visual_minus_G':max((deveq[k]-deveq['G'] for k in ('V','VQ') if deveq[k] is not None and deveq['G'] is not None),default=None),
  'rv_dev_only':{k:(None if res['keyframe']['rv_dev'][k].get('rv:all') is None else res['keyframe']['rv_dev'][k]['rv:all']['sel_minus_mother']) for k in ('G','V','VQ')}}
 pi=res['pipeline']
 def g2(name,kind):
  p=pi[name][kind];a=p['all']
  return {'mean':a['mean'],'ci_low':a['ci95'][0],'axis1_mean':(p['axis1']['mean'] if p['axis1'] else None)}
 gate2={'rv_dev':{k:g2('rv_dev',k) for k in ('V','VQ')},'confirm':{k:g2('rv_confirm2',k) for k in ('V','VQ')}}
 res['decision']={'gate1_visual_value':gate1,'gate2_pipeline':gate2}
 ok1=gate1['best_visual_minus_G'] is not None and gate1['best_visual_minus_G']>0
 ok2=False;bestkind=None
 for k in ('V','VQ'):
  d=g2('rv_dev',k)
  if d['mean']>0 and d['ci_low']>-0.002 and (d['axis1_mean'] is None or d['axis1_mean']>=-0.002):
   c=g2('rv_confirm2',k)
   if c['mean']>0 and c['ci_low']>-0.002 and (c['axis1_mean'] is None or c['axis1_mean']>=-0.002):
    ok2=True;bestkind=k
 res['decision']['promote_to_package']=bool(ok1 and ok2);res['decision']['chosen_head']=bestkind
 res['official_f_video']=None
 torch.save({k:{'net':heads[k][0].state_dict(),'proj':(heads[k][1].state_dict() if heads[k][1] is not None else None),'mu':heads[k][2][0].cpu(),'sd':heads[k][2][1].cpu(),'kind':k} for k in heads},a.output/'h3_heads.pt')
 (a.output/'metrics.json').write_text(json.dumps(res,indent=1,default=float)+'\n')
 (a.output/'ablation.csv').write_text('\n'.join(csv)+'\n')
 log(json.dumps({'decision':res['decision']}));log('done')
if __name__=='__main__':main()
