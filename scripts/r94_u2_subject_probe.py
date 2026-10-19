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


def load_u2_model(path, device):
    import sys, torch
    sys.path.insert(0, '/data/aic/experiments')
    import R94_U2_H1_u2net_model as um
    net=um.U2NETP(3,1)
    ck=torch.load(path,map_location='cpu',weights_only=True)
    net.load_state_dict(ck,strict=True); net.eval().to(device)
    return net

U2_NAMES=['u2_center_x','u2_center_y','u2_area','u2_conf','u2_lo','u2_hi','u2_extent','u2_cand_delta','u2_cand_abs_delta','u2_cand_cov','u2_bbox_center_delta','u2_bbox_edge']

def u2_stats_batch(net, paths, device):
    import torch, cv2
    ims=[]; shapes=[]
    for p in paths:
        im=cv2.imread(str(p),cv2.IMREAD_COLOR)
        if im is None: raise FileNotFoundError(str(p))
        im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB); h,w=im.shape[:2]; shapes.append((w,h))
        im=cv2.resize(im,(320,320),interpolation=cv2.INTER_LINEAR).astype('float32')/255.
        im=(im-.5)/.5; ims.append(torch.from_numpy(im).permute(2,0,1))
    x=torch.stack(ims).to(device)
    with torch.inference_mode():
        y=net(x)[0].sigmoid().detach().float().cpu().numpy()[:,0]
    out=[]
    for m,(w,h) in zip(y,shapes):
        m=np.asarray(m,float); m=np.maximum(m,0); den=float(m.sum())+1e-8
        yy,xx=np.mgrid[0:m.shape[0],0:m.shape[1]]
        cx=float((m*xx).sum()/den)/(m.shape[1]-1); cy=float((m*yy).sum()/den)/(m.shape[0]-1)
        area=float(m.mean()); conf=float(np.percentile(m,95))
        nz=np.argwhere(m>=max(0.35,0.5*float(m.max())))
        if len(nz):
            lo=float(nz[:,1].min()/(m.shape[1]-1)); hi=float(nz[:,1].max()/(m.shape[1]-1)); ext=float((hi-lo))
        else: lo=hi=cx; ext=0.
        out.append(np.array([cx,cy,area,conf,lo,hi,ext],np.float32))
    return np.stack(out)

def attach_u2(units, split_name, net, device, root):
    import torch
    vals=[]; paths=[]; refs=[]
    subdir='keyframes_val' if split_name=='live_confirm3' else 'keyframes_train'
    for ui,u in enumerate(units):
        if u['ds']!='live': raise ValueError('U2 pilot is live-only')
        for ki,j in enumerate(u['J']):
            key=int(u['keys'][j]); p=Path(root)/subdir/u['vid']/f'{key}.png'
            if not p.is_file(): raise FileNotFoundError(str(p))
            paths.append(p); refs.append((ui,ki,u))
    allstats=[]
    for i in range(0,len(paths),32): allstats.append(u2_stats_batch(net,paths[i:i+32],device))
    allstats=np.concatenate(allstats,0)
    by=[[] for _ in units]
    for st,(ui,ki,u) in zip(allstats,refs): by[ui].append(st)
    for u,sts in zip(units,by):
        S=np.stack(sts) if sts else np.zeros((0,7),np.float32); C=u['X'].shape[1]; n=len(S)
        F=np.zeros((n,C,len(U2_NAMES)),np.float32)
        if n:
            comp=u['comp']; win=u['s']/u['L']; cand=(u['offs'][:n]+u['s']/2)/u['L']
            F[:,:,0]=S[:,0,None]; F[:,:,1]=S[:,1,None]; F[:,:,2]=S[:,2,None]; F[:,:,3]=S[:,3,None]
            F[:,:,4]=S[:,4,None]; F[:,:,5]=S[:,5,None]; F[:,:,6]=S[:,6,None]
            delta=(cand-S[:,0,None])/max(win,1e-8); F[:,:,7]=delta; F[:,:,8]=np.abs(delta)
            F[:,:,9]=np.clip(np.minimum(cand+win/2,S[:,5,None])-np.maximum(cand-win/2,S[:,4,None]),0,None)/np.maximum(S[:,6,None],1e-6)
            F[:,:,10]=(cand-S[:,0,None])/max(win,1e-8); F[:,:,11]=((cand-win/2-S[:,4,None])+(S[:,5,None]-(cand+win/2)))/max(win,1e-8)
        u['X_U2']=np.concatenate([u['X'],F],axis=2).astype(np.float32)
        u['u2_stats']=S
    return units

def fit_model(train, dev, seed, use_u2, steps=3000, log=print):
    import torch, torch.nn as nn
    torch.manual_seed(seed); rng=np.random.default_rng(seed); D=train[0]['X_U2'].shape[2] if use_u2 else train[0]['X'].shape[2]
    key='X_U2' if use_u2 else 'X'; Xall=np.concatenate([u[key].reshape(-1,D) for u in train]); mu=Xall.mean(0); sd=Xall.std(0); sd[sd<1e-6]=1
    T=lambda a:torch.tensor((a-mu)/sd,dtype=torch.float32)
    pool=[u for u in train if (~np.isnan(u['Y'][:,0])).any()]; net=nn.Sequential(nn.Linear(D,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1))
    opt=torch.optim.AdamW([{'params':[p for p in net.parameters() if p.ndim==2],'weight_decay':1e-3},{'params':[p for p in net.parameters() if p.ndim<2],'weight_decay':0.}],lr=3e-4)
    warm=max(1,int(.05*steps)); sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:(s+1)/warm if s<warm else .5*(1+math.cos(math.pi*(s-warm)/max(1,steps-warm))))
    best=(-1,None,None); hist=[]
    def score():
        net.eval(); vv=[]
        with torch.no_grad():
            for u in dev:
                m=~np.isnan(u['Y'][:,0]);
                if not m.any(): continue
                pred=net(T(u[key][m])).squeeze(-1).numpy().reshape(m.sum(),-1); idx=pred.argmax(1); vv.append(float(u['Y'][m][np.arange(m.sum()),idx].mean()))
        return float(np.mean(vv)) if vv else float('nan')
    for step in range(steps):
        xs=[]; ys=[]
        for _ in range(16):
            u=pool[int(rng.integers(len(pool)))]; ok=np.flatnonzero(~np.isnan(u['Y'][:,0])); kk=rng.choice(ok,min(8,len(ok)),replace=False); xs.append(u[key][kk]); ys.append(u['Y'][kk])
        x=T(np.concatenate(xs)); y=torch.tensor(np.concatenate(ys),dtype=torch.float32); net.train(); loss=((net(x).squeeze(-1)-y)**2).mean(); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.); opt.step(); sch.step()
        if (step+1)%250==0:
            d=score(); hist.append({'step':step+1,'loss':float(loss.detach()),'dev_keyframe_iou':d}); log(json.dumps({'use_u2':use_u2,**hist[-1]}))
            if d>best[0]: best=(d,step+1,{k:v.clone() for k,v in net.state_dict().items()})
    net.load_state_dict(best[2]); net.eval(); return net,(mu,sd),{'best_step':best[1],'best_dev_keyframe_iou':best[0],'history':hist,'input_dim':D,'parameter_count':sum(p.numel() for p in net.parameters())}

def centres_model(net,norm,u,use_u2):
    import torch
    key='X_U2' if use_u2 else 'X'; mu,sd=norm
    with torch.no_grad(): pred=net(torch.tensor((u[key]-mu)/sd,dtype=torch.float32)).squeeze(-1).numpy(); idx=pred.argmax(1)
    return centres_from_idx(u,idx)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--u2',type=Path,required=True); ap.add_argument('--steps',type=int,default=3000); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--tables',type=Path,default=E/'tables'); ap.add_argument('--keyframes',type=Path,default=E); a=ap.parse_args(); a.output.mkdir(parents=True,exist_ok=False); a.tables.mkdir(parents=True,exist_ok=True)
    import torch
    device='cuda' if torch.cuda.is_available() else 'cpu'; net_u2=load_u2_model(str(a.u2),device)
    S={n:table(n,a.tables) for n in ('live_train','live_dev','live_confirm3')}
    for n in S: attach_u2(S[n],n,net_u2,device,a.keyframes)
    logf=open(a.output/'train.log','w'); log=lambda s:(print(s,flush=True),logf.write(s+'\n'),logf.flush())
    log(json.dumps({'device':device,'units':{n:len(v) for n,v in S.items()},'keyframes':{n:int(sum(len(u['J']) for u in v)) for n,v in S.items()},'u2_parameters':sum(p.numel() for p in net_u2.parameters())}))
    train=S['live_train']; dev=S['live_dev']; models={}
    for use in (False,True): models[use]=fit_model(train,dev,a.seed,use,a.steps,log)
    out={'experiment_id':'R94_U2_H1_LIVE','hypothesis':'Frozen U2NETP foreground extent and candidate relation add crop utility beyond identical 26D control at fixed T5 candidate set','pdf_source':'研究1/视觉研究结论.pdf','device':device,'u2_weight_sha256':hashlib.sha256(a.u2.read_bytes()).hexdigest(),'u2_parameters':sum(p.numel() for p in net_u2.parameters()),'models':{str(k):models[k][2] for k in models},'splits':{n:len(v) for n,v in S.items()},'metrics':{}}
    for n,us in S.items():
        base=[pipe(u) for u in us]; vals={False:[pipe(u,centres_model(models[False][0],models[False][1],u,False)) for u in us], True:[pipe(u,centres_model(models[True][0],models[True][1],u,True)) for u in us]}
        out['metrics'][n]={'H0':float(np.mean(base)),'control_zero_u2':summarise(us,base,vals[False]),'u2_full':summarise(us,base,vals[True]),'u2_vs_control':summarise(us,vals[False],vals[True])}
        log(json.dumps({n:{k:(v if not isinstance(v,dict) else {'mean':v['mean'],'ci95':v['ci95'],'videos':v['videos']}) for k,v in out['metrics'][n].items()}}))
    # Save only heads and feature stats; U2 itself is an external frozen dependency and not copied to git.
    for use,(net,norm,meta) in models.items(): torch.save({'state_dict':net.state_dict(),'mu':norm[0],'sd':norm[1],'use_u2':use,'seed':a.seed,'meta':meta},a.output/f'head_{"u2" if use else "control"}.pt')
    out['decision']={'primary':'u2_full_vs_control','gate_delta':0.01,'pass_dev_independent':bool(out['metrics']['live_dev']['u2_vs_control']['mean']>=0.01 and out['metrics']['live_dev']['u2_vs_control']['ci95'][0]>0),'official_f_video':None}
    (a.output/'metrics.json').write_text(json.dumps(out,indent=2)+'\n')
    log(json.dumps(out['decision']))
if __name__=='__main__': main()
