"""T5_CROPHEAD_V3 probe: can public human crops teach the point->window mapping?

Prereg: configs/T5_CROPHEAD_V3_PREREG.json.  Every valid 1 s Qwen keyframe point
(N0 protocol) gets 65 legal max-window positions + the N0 point-centred one.
Target y(c) = mean over annotators of IoU(window c, GT_r) at that keyframe
(same rounded, inclusive-pixel IoU as scripts/teacher_diag_eval.py).  Features
use inference-time observations only (see FEATS); GT never enters them.
  H0  N0 point-centred window
  H1  centre = a*p + (1-a)*.5 + b*win per axis (grid fit on train keyframes)
  H2  MLP 2x64 -> y(c), MSE, AdamW (see prereg), argmax at inference
Evaluation = full N0 pipeline (hold within shot, face gate, B0 EMA) with only
the keyframe centres replaced; per-video IoU vs H0, video bootstrap.
Also reported: keyframe candidate oracle, the keyframe oracle pushed through
the pipeline, and a per-annotator vs mean-annotator oracle check.
"""
import argparse,hashlib,json,math,pickle
from pathlib import Path
import numpy as np
from aic.max_window_path import geometry,centre_to_offset,ema_offsets,qwen_centres,frame_boxes
from scripts.benchmark_spatial import iou
from scripts.max_window_path_eval import load_gt,RATIOS
from scripts.teacher_diag_eval import win_boxes

K=65
E=Path('/data/aic/experiments/T5_CROPHEAD_V3')
RV_ANN=Path('/data/aic/datasets/RetargetVid/annotations_all');RV_CACHE=Path('/data/aic/experiments/OBS_CACHE_RETARGET_ALL200');RV_PTS=Path('/data/aic/experiments/QWEN_RV200_POINT_D1/points')
FEATS=['axis','win','log_aspect','rel_pos','d_pt','abs_d_pt','d_pt2','d_pc','pt_edge','pt_other','d_sal','sal_conf','d_raw','face','box_has','box_lo','box_hi','box_w','box_cov','n_person','cov_rel','cov_mass','d_next','d_prev','has_next','has_prev']


def live_rows(split):
 rows=[json.loads(l) for l in open(E/f'live_{"val" if split=="val" else "train"}_index.jsonl')]
 fold=lambda v:int(hashlib.sha1(v.encode()).hexdigest(),16)%10==0
 return [r for r in rows if split=='val' or (fold(r['video_id'])==(split=='dev'))]


def split_units(name):
 if name=='rv_train':return [('rv',f'{i:03d}',r,None) for i in range(1,81) for r in RATIOS]
 if name=='rv_dev':return [('rv',f'{i:03d}',r,None) for i in range(81,101) for r in RATIOS]
 if name=='rv_confirm2':return [('rv',str(i),r,None) for i in range(601,701) for r in RATIOS]
 sub={'live_train':'train','live_dev':'dev','live_confirm3':'val'}[name]
 return [('live',r['video_id'],'t',{**r,'sub':'val' if sub=='val' else 'train'}) for r in live_rows(sub)]


def kf_features(z,t,p,nb,comp,W,H,ratio,offs):
 w,h,_=geometry(W,H,ratio);L,s=(W,w) if comp==0 else (H,h);win=s/L;cc=(offs+s/2)/L;pa,pb=p[comp],p[1-comp]
 p0=(centre_to_offset(np.array([pa]),W,H,ratio)[0]+s/2)/L;sal=z['sal'][t];raw=float(z['raw'][t,comp]);face=float(z['chosen'][t]>=0)
 d=z['det'][z['det_off'][t]:z['det_off'][t+1]];d=d[d[:,4]>=.5];px,py=p[0]*W,p[1]*H
 inside=d[(d[:,0]<=px)&(px<=d[:,2])&(d[:,1]<=py)&(py<=d[:,3])] if len(d) else d
 n=len(offs);f=np.zeros((n,len(FEATS)),np.float32);F=dict(zip(FEATS,range(len(FEATS))))
 f[:,F['axis']]=comp;f[:,F['win']]=win;f[:,F['log_aspect']]=math.log(W/H);f[:,F['rel_pos']]=offs/max(L-s,1e-9)
 f[:,F['d_pt']]=(cc-pa)/win;f[:,F['abs_d_pt']]=np.abs(cc-pa)/win;f[:,F['d_pt2']]=((cc-pa)/win)**2;f[:,F['d_pc']]=(cc-p0)/win;f[:,F['pt_edge']]=(p0-pa)/win;f[:,F['pt_other']]=pb
 f[:,F['d_sal']]=(cc-sal[comp])/win;f[:,F['sal_conf']]=sal[2];f[:,F['d_raw']]=(cc-raw)/win;f[:,F['face']]=face
 if len(inside):
  b=inside[np.argmax(inside[:,4])];lo,hi=b[comp]/L,b[comp+2]/L;wl,wh=cc-win/2,cc+win/2
  f[:,F['box_has']]=1;f[:,F['box_lo']]=(lo-wl)/win;f[:,F['box_hi']]=(hi-wh)/win;f[:,F['box_w']]=(hi-lo)/win;f[:,F['box_cov']]=np.clip(np.minimum(hi,wh)-np.maximum(lo,wl),0,None)/max(hi-lo,1e-9)
 f[:,F['n_person']]=math.log1p(int((d[:,5]==1).sum()) if len(d) else 0)
 ext=frame_boxes(z,t)
 if len(ext):
  lo,hi,wt=ext.T;ln=np.maximum(hi-lo,1.);sc=np.array([(wt*np.clip(np.minimum(hi,o+s)-np.maximum(lo,o),0,None)/ln).sum() for o in offs])
  f[:,F['cov_rel']]=sc/max(sc.max(),1e-9);f[:,F['cov_mass']]=math.log1p(float(wt.sum()))
 if nb[1] is not None:f[:,F['d_next']]=(nb[1]-pa)/win;f[:,F['has_next']]=1
 if nb[0] is not None:f[:,F['d_prev']]=(pa-nb[0])/win;f[:,F['has_prev']]=1
 return f


def load_unit(u):
 ds,vid,r,row=u
 if ds=='rv':
  z=np.load(RV_CACHE/f'{vid}_{r}.npz');qq=json.loads((RV_PTS/f'{vid}.json').read_text());gt=load_gt(RV_ANN,vid,r);gf=np.arange(gt.shape[1])
  if gt.shape[1]!=len(z['b0']):raise ValueError('frames '+vid)
 else:
  z=np.load(E/f"obs_cache_live_{row['sub']}"/f'{vid}_t.npz');qq=json.loads((E/f"points_{row['sub']}"/f'{vid}.json').read_text())
  a=json.loads(Path(row['annotation']).read_text());W,H=int(z['W']),int(z['H'])
  b=np.array([[x['left'],x['top'],x['right'],x['bottom']] for x in a['raw_ltrb']],float);b[:,[0,2]]=np.clip(b[:,[0,2]],0,W-1);b[:,[1,3]]=np.clip(b[:,[1,3]],0,H-1)
  gf=np.array(a['frames']);ok=gf<len(z['b0']);gf,gt=gf[ok],b[ok][None]
 W,H=int(z['W']),int(z['H']);ratio=z['ratio'].tolist();w,h,axis=geometry(W,H,ratio)
 if axis is None:return None
 comp=0 if axis==0 else 1;L,s=(W,w) if comp==0 else (H,h);reset=z['reset'].astype(bool);keys=qq['keyframes'];pts=qq['ratios'][r]['points']
 shot=np.cumsum(reset);valid=[j for j,p in enumerate(pts) if p is not None and not p[2]];gpos={int(f):i for i,f in enumerate(gf)}
 X=[];Y=[];OF=[];J=[]
 for jj,j in enumerate(valid):
  t=keys[j];p=pts[j];pv=[k for k in valid if k<j and shot[keys[k]]==shot[t]];nx=[k for k in valid if k>j and shot[keys[k]]==shot[t]]
  nb=(pts[pv[-1]][comp] if pv else None,pts[nx[0]][comp] if nx else None)
  offs=np.append(np.linspace(0,L-s,K),centre_to_offset(np.array([p[comp]]),W,H,ratio)[0])
  X.append(kf_features(z,t,p,nb,comp,W,H,ratio,offs));OF.append(offs);J.append(j)
  Y.append(iou(win_boxes(offs,W,H,ratio)[None],gt[:,gpos[t]][:,None]).mean(0) if t in gpos else np.full(len(offs),np.nan))
 C=K+1;return {'ds':ds,'vid':vid,'r':r,'W':W,'H':H,'ratio':ratio,'comp':comp,'L':L,'s':s,'keys':keys,'pts':pts,'reset':reset,'raw':z['raw'][:,comp].copy(),'face':z['chosen']>=0,
  'gt':gt,'gf':gf,'X':np.array(X,np.float32).reshape(-1,C,len(FEATS)),'Y':np.array(Y,np.float32).reshape(-1,C),'offs':np.array(OF).reshape(-1,C),'J':np.array(J,int),
  'pa':np.array([pts[j][comp] for j in J]) if J else np.zeros(0)}


def table(name,cache_dir):
 f=cache_dir/f'{name}.pkl'
 if f.exists():return pickle.loads(f.read_bytes())
 us=[x for x in (load_unit(u) for u in split_units(name)) if x is not None];f.write_bytes(pickle.dumps(us));return us


def pipe(u,centres=None):
 pts=[list(p) if p is not None else None for p in u['pts']]
 if centres is not None:
  for j,c in zip(u['J'],centres):pts[j][u['comp']]=float(c)
 c,_=qwen_centres(pts,u['keys'],u['reset'],u['raw'],u['comp']);c=np.where(u['face'],u['raw'],c)
 off=ema_offsets(c,u['reset'],u['W'],u['H'],u['ratio'])[u['gf']]
 return float(iou(win_boxes(off,u['W'],u['H'],u['ratio'])[None],u['gt']).mean())


def centres_from_idx(u,idx):return (u['offs'][np.arange(len(idx)),idx]+u['s']/2)/u['L']


def h1_centres(u,ab):
 a,b=ab[u['comp']];return np.clip(a*u['pa']+(1-a)*.5+b*u['s']/u['L'],0,1)


def fit_h1(units):
 A=np.linspace(.5,1.5,21);B=np.linspace(-.5,.5,41);G=np.array([(a,b) for a in A for b in B]);out={}
 for comp in (0,1):
  per={}
  for u in units:
   m=~np.isnan(u['Y'][:,0])
   if u['comp']!=comp or not m.any():continue
   win=u['s']/u['L'];c=np.clip(G[:,:1]*u['pa'][m][None]+(1-G[:,:1])*.5+G[:,1:]*win,0,1)
   off=centre_to_offset(c.ravel(),u['W'],u['H'],u['ratio']);bx=win_boxes(off,u['W'],u['H'],u['ratio']).reshape(len(G),-1,4)
   gpos={int(f):i for i,f in enumerate(u['gf'])};g=u['gt'][:,[gpos[u['keys'][j]] for j in u['J'][m]]]
   per.setdefault(u['ds'],[]).append(iou(bx[None],g[:,None]).mean(0).mean(1))
  if not per:out[comp]=(1.,0.);continue
  obj=np.mean([np.mean(v,0) for v in per.values()],0);k=int(np.argmax(obj));out[comp]=(float(G[k,0]),float(G[k,1]))
 return out


def standardise(train):
 X=np.concatenate([u['X'].reshape(-1,len(FEATS)) for u in train]);mu=X.mean(0);sd=X.std(0);sd[sd<1e-6]=1;return mu,sd


def train_h2(train,dev,seed,steps=3000,log=print):
 import torch,torch.nn as nn
 torch.manual_seed(seed);rng=np.random.default_rng(seed);mu,sd=standardise(train);T=lambda a:torch.tensor((a-mu)/sd,dtype=torch.float32)
 pools={ds:[u for u in train if u['ds']==ds and (~np.isnan(u['Y'][:,0])).any()] for ds in ('rv','live')};pools={k:v for k,v in pools.items() if v}
 net=nn.Sequential(nn.Linear(len(FEATS),64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1))
 opt=torch.optim.AdamW([{'params':[p for p in net.parameters() if p.ndim==2],'weight_decay':1e-3},{'params':[p for p in net.parameters() if p.ndim<2],'weight_decay':0.}],lr=3e-4)
 warm=max(1,int(.05*steps));sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:(s+1)/warm if s<warm else .5*(1+math.cos(math.pi*(s-warm)/max(1,steps-warm))))
 per_ds=16//len(pools);best=(-1,None,None);hist=[]
 def dev_score():
  with torch.no_grad():
   sc={}
   for u in dev:
    m=~np.isnan(u['Y'][:,0])
    if m.any():i=net(T(u['X'][m])).squeeze(-1).argmax(1).numpy();sc.setdefault(u['ds'],[]).append(float(u['Y'][m][np.arange(m.sum()),i].mean()))
   return float(np.mean([np.mean(v) for v in sc.values()]))
 for step in range(steps):
  xs=[];ys=[]
  for ds,pool in pools.items():
   for ui in rng.integers(0,len(pool),per_ds):
    u=pool[ui];ok=np.flatnonzero(~np.isnan(u['Y'][:,0]));k=rng.choice(ok,8);xs.append(u['X'][k]);ys.append(u['Y'][k])
  x=T(np.stack(xs));y=torch.tensor(np.stack(ys));loss=((net(x).squeeze(-1)-y)**2).mean()
  opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();sch.step()
  if (step+1)%250==0:
   d=dev_score();hist.append({'step':step+1,'loss':float(loss.detach()),'dev_keyframe_iou':d});log(json.dumps(hist[-1]))
   if d>best[0]:best=(d,step+1,{k:v.clone() for k,v in net.state_dict().items()})
 net.load_state_dict(best[2]);return net,(mu,sd),{'best_step':best[1],'best_dev_keyframe_iou':best[0],'history':hist}


def h2_centres(net,norm,u):
 import torch
 if not len(u['J']):return np.zeros(0)
 mu,sd=norm
 with torch.no_grad():i=net(torch.tensor((u['X']-mu)/sd,dtype=torch.float32)).squeeze(-1).argmax(1).numpy()
 return centres_from_idx(u,i)


def boot(dv,seed=20260928,n=5000):
 dv=np.asarray(dv,float);b=np.random.default_rng(seed).choice(dv,(n,len(dv))).mean(1);return [float(np.percentile(b,2.5)),float(np.percentile(b,97.5))]


def per_video(units,vals,comp=None):
 out={}
 for u,v in zip(units,vals):
  if comp is None or u['comp']==comp:out.setdefault((u['ds'],u['vid']),[]).append(v)
 return {k:float(np.mean(v)) for k,v in out.items()}


def summarise(units,base,new,comp=None):
 b=per_video(units,base,comp);n=per_video(units,new,comp)
 if not b:return None
 d=np.array([n[k]-b[k] for k in b]);return {'videos':len(d),'base':float(np.mean(list(b.values()))),'new':float(np.mean(list(n.values()))),'mean':float(d.mean()),'ci95':boot(d),'better':int((d>1e-4).sum()),'worse':int((d<-1e-4).sum()),'worst':float(d.min())}


def keyframe_oracle(units):
 """Keyframe candidate oracle vs H0, and per-annotator vs mean-annotator oracle (RV only)."""
 rows={}
 for u in units:
  m=~np.isnan(u['Y'][:,0])
  if not m.any():continue
  Y=u['Y'][m];rows.setdefault((u['ds'],u['comp']),[]).append((float(Y[:,-1].mean()),float(Y[:,:K].max(1).mean()),float(Y.max(1).mean())))
 out={f'{ds}:axis{c}':dict(zip(('h0','oracle_grid65','oracle_grid65_plus_h0'),np.mean(v,0).round(4).tolist()),units=len(v)) for (ds,c),v in rows.items()}
 pa=[]
 for u in units:
  if u['ds']!='rv' or not len(u['J']):continue
  gpos={int(f):i for i,f in enumerate(u['gf'])};g=u['gt'][:,[gpos[u['keys'][j]] for j in u['J']]]
  per=np.stack([iou(win_boxes(o,u['W'],u['H'],u['ratio'])[None],g[:,k:k+1]) for k,o in enumerate(u['offs'])],1)  # [A,nk,C]
  pa.append((float(per.mean(0).max(1).mean()),float(per.max(2).mean())))
 if pa:out['rv:mean_annotator_oracle_vs_per_annotator_oracle']=np.mean(pa,0).round(4).tolist()
 return out


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=0);ap.add_argument('--steps',type=int,default=3000);ap.add_argument('--output',type=Path,required=True)
 ap.add_argument('--tables',type=Path,default=E/'tables');a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);a.tables.mkdir(parents=True,exist_ok=True)
 logf=open(a.output/'train.log','a');log=lambda s:(print(s,flush=True),logf.write(s+'\n'),logf.flush())
 S={n:table(n,a.tables) for n in ('rv_train','rv_dev','rv_confirm2','live_train','live_dev','live_confirm3')}
 log(json.dumps({n:{'units':len(v),'keyframes':int(sum(len(u['J']) for u in v)),'keyframes_with_gt':int(sum((~np.isnan(u['Y'][:,0])).sum() for u in v))} for n,v in S.items()}))
 train=S['rv_train']+S['live_train'];dev=S['rv_dev']+S['live_dev']
 ab=fit_h1(train);log(json.dumps({'H1':ab}))
 net,norm,th=train_h2(train,dev,a.seed,a.steps,log)
 res={'prereg_sha256':hashlib.sha256(Path('configs/T5_CROPHEAD_V3_PREREG.json').read_bytes()).hexdigest(),'seed':a.seed,'H1':{str(k):v for k,v in ab.items()},'H2_train':th,'features':FEATS}
 pol={'H0':lambda u:None,'H1':lambda u:h1_centres(u,ab),'H2':lambda u:h2_centres(net,norm,u),'ORACLE_KF':lambda u:None}
 ev={}
 for name,us in S.items():
  if name.endswith('train'):continue
  vals={'H0':[pipe(u) for u in us],'H1':[pipe(u,h1_centres(u,ab)) for u in us],'H2':[pipe(u,h2_centres(net,norm,u)) for u in us]}
  orc=[]
  for u in us:
   m=~np.isnan(u['Y'][:,0]);c=u['pa'].copy();c[m]=centres_from_idx(u,u['Y'][m].argmax(1)) if m.any() else c[m];orc.append(pipe(u,c) if len(u['J']) else pipe(u))
  vals['ORACLE_KF']=orc;ev[name]=vals
  res[name]={p:{'all':summarise(us,vals['H0'],vals[p]),'axis0':summarise(us,vals['H0'],vals[p],0),'axis1':summarise(us,vals['H0'],vals[p],1)} for p in ('H1','H2','ORACLE_KF')}
  res[name]['H0_iou']=float(np.mean(list(per_video(us,vals['H0']).values())))
  log(json.dumps({name:{p:{k:(None if v is None else {'mean':round(v['mean'],4),'ci95':[round(x,4) for x in v['ci95']],'videos':v['videos']}) for k,v in d.items()} for p,d in res[name].items() if p!='H0_iou'},'H0_iou':round(res[name]['H0_iou'],4)}))
 res['keyframe_oracle']={n:keyframe_oracle(S[n]) for n in ('rv_dev','rv_confirm2','live_dev','live_confirm3')}
 # dev selection (prereg): dataset-equal dev gain, then per-axis application
 def dev_gain(p,comp=None):
  g=[res[n][p]['all' if comp is None else f'axis{comp}'] for n in ('rv_dev','live_dev')];g=[x['mean'] for x in g if x];return float(np.mean(g)) if g else None
 chosen='H1' if (dev_gain('H1') or -1)>=(dev_gain('H2') or -1) else 'H2';axes=[c for c in (0,1) if (dev_gain(chosen,c) or -1)>0]
 res['selection']={'dev_gain':{p:{'all':dev_gain(p),'axis0':dev_gain(p,0),'axis1':dev_gain(p,1)} for p in ('H1','H2')},'chosen':chosen,'applied_axes':axes}
 # confirmation with the chosen model on the applied axes only
 conf={}
 for name in ('live_confirm3','rv_confirm2'):
  us=S[name];new=[ev[name][chosen][i] if us[i]['comp'] in axes else ev[name]['H0'][i] for i in range(len(us))]
  conf[name]={'all':summarise(us,ev[name]['H0'],new),'axis0':summarise(us,ev[name]['H0'],new,0),'axis1':summarise(us,ev[name]['H0'],new,1)}
 ok=bool(axes) and conf['live_confirm3']['all']['mean']>0 and conf['live_confirm3']['all']['ci95'][0]>0
 for c in axes:
  x=conf['live_confirm3' if c==0 else 'rv_confirm2'][f'axis{c}'];ok=ok and x is not None and x['mean']>0 and x['ci95'][0]>0
 res['confirmation']=conf;res['promote']=bool(ok);res['official_f_video']=None
 import torch
 torch.save({'state_dict':net.state_dict(),'mu':norm[0],'sd':norm[1],'features':FEATS,'H1':ab,'seed':a.seed},a.output/'crophead.pt')
 (a.output/'metrics.json').write_text(json.dumps(res,indent=1)+'\n');log(json.dumps({'selection':res['selection'],'confirmation':{k:{kk:(None if vv is None else {'mean':round(vv['mean'],4),'ci95':[round(x,4) for x in vv['ci95']]}) for kk,vv in v.items()} for k,v in conf.items()},'promote':res['promote']}))
if __name__=='__main__':main()
