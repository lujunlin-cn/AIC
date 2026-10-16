"""R31 low-dimensional VPB event-state probe; no DEV tuning."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x): Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def fs(mask,order,k):
 m=mask.sum(1); tp=mask[:,order[:k]].sum(1); return float(np.divide(2*tp,k+m,out=np.zeros(len(m)),where=k+m>0).mean())
def fit(rows,kind,ridge):
 x=np.concatenate([r['feat'][kind] for r in rows]); y=np.concatenate([r['sample_y'] for r in rows]); w=np.concatenate([np.full(12,1/(len(rows)*12)) for _ in rows])
 mu=(x*w[:,None]).sum(0); sd=np.sqrt(((x-mu)**2*w[:,None]).sum(0)).clip(1e-6); xx=(x-mu)/sd; reg=np.ones(xx.shape[1]); reg[0]=0
 def fun(b):
  z=xx@b; loss=(w*(np.logaddexp(0,z)-y*z)).sum()+ridge*(reg*b*b).sum()/2; g=xx.T@(w*(expit(z)-y))+ridge*reg*b; return loss,g
 q=minimize(fun,np.zeros(xx.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':500,'ftol':1e-12,'gtol':1e-8})
 if not q.success: raise RuntimeError(q.message)
 return {'kind':kind,'beta':q.x.tolist(),'mu':mu.tolist(),'sd':sd.tolist(),'ridge':ridge,'iterations':int(q.nit)}
def pred(r,m): return expit(((r['feat'][m['kind']]-m['mu'])/m['sd'])@np.asarray(m['beta']))
def boot(d):
 rng=np.random.default_rng(20261016); z=rng.choice(d,(10000,len(d)),replace=True).mean(1); return {'mean':float(np.mean(d)),'ci95':np.quantile(z,[.025,.975]).tolist(),'sources':len(d)}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); a=ap.parse_args(); c=json.loads(Path(a.config).read_text()); out=Path(c['output']);
 if out.exists(): raise FileExistsError(out)
 out.mkdir(parents=True); dump(out/'config.json',c); tick=time.time()
 ident=json.loads(Path(c['identity_metrics']).read_text()); report=json.loads(Path(c['axis_report']).read_text()); axes={x['video_id']:x for x in report['axis']}; bind={x['video_id']:x for x in json.loads(Path(c['bindings']).read_text())}
 raw=[]; train_ids=ident['train_ids']
 for split in ('train','dev'):
  for vid in ident[f'{split}_ids']:
   ar=axes[vid]; axis=Path(c['data_root'])/ar['axis_array_relpath']; assert sha(axis)==ar['axis_array_sha256']; z=np.load(axis); pts=z['timestamps']; mask=z['raw_summary_membership'].astype(bool); ix=np.asarray(bind[vid]['sample_frame_ids']); assert np.array_equal(pts[ix],bind[vid]['actual_pts'])
   e=np.load(Path(c['feature_root'])/f'{vid}.npz')['vpb'].astype(float); assert e.shape==(12,768) and np.isfinite(e).all()
   raw.append((vid,split,pts,mask,ix,e))
 # PCA is unsupervised and fit on TRAIN only.
 E=np.concatenate([r[5] for r in raw if r[1]=='train']); em=E.mean(0); _,_,V=np.linalg.svd(E-em,full_matrices=False); P=V[:c['pca_components']]
 rows=[]
 for vid,split,pts,mask,ix,e in raw:
  nrm=np.linalg.norm(e,axis=1); d=np.diff(e,axis=0,prepend=e[:1]); dn=np.linalg.norm(d,axis=1); acc=np.linalg.norm(np.diff(d,axis=0,prepend=d[:1]),axis=1); cos=np.ones(12); cos[1:]=1-np.sum(e[1:]*e[:-1],1)/(nrm[1:]*nrm[:-1]+1e-8)
  def roll(a,w=3): return np.convolve(np.pad(a,(w//2,w//2),mode='edge'),np.ones(w)/w,mode='valid')
  t=2*(pts[ix]-pts[0])/max(pts[-1]-pts[0],1e-8)-1
  pos=np.stack([np.ones(12),t,t*t,t*t*t],1)
  low=np.stack([np.ones(12),nrm,dn,acc,cos,roll(dn),roll(cos),*([t] if True else [])],1)
  pca=np.concatenate([np.ones((12,1)),(e-em)@P.T],1)
  both=np.concatenate([low,pca[:,1:]],1)
  owner=np.abs(pts[:,None]-pts[ix][None]).argmin(1)
  rows.append({'id':vid,'split':split,'mask':mask,'owner':owner,'sample_y':mask.mean(0)[ix],'feat':{'position':pos,'low':low,'pca':pca,'low_pca':both}})
 train=[r for r in rows if r['split']=='train']; dev=[r for r in rows if r['split']=='dev']; models={k:fit(train,k,c['ridge']) for k in c['arms']}; rec=[]
 for r in rows:
  k=round(c['keep']*len(r['owner'])); center=np.argsort(abs(np.arange(len(r['owner']))-(len(r['owner'])-1)/2),kind='stable'); rec.append({'video_id':r['id'],'split':r['split'],'arm':'center85','f':fs(r['mask'],center,k),'kept':k,'frames':len(r['owner'])})
  for name,m in models.items():
   order=np.argsort(-pred(r,m)[r['owner']],kind='stable'); rec.append({'video_id':r['id'],'split':r['split'],'arm':name+'85','f':fs(r['mask'],order,k),'kept':k,'frames':len(r['owner'])})
 arms={}
 for split in ('train','dev'):
  arms[split]={}
  for name in sorted(set(x['arm'] for x in rec)):
   q=[x for x in rec if x['split']==split and x['arm']==name]; arms[split][name]={'f':float(np.mean([x['f'] for x in q])),'videos':q}
 contrasts={}
 for cand,ctrl in c['contrasts']:
  x={q['video_id']:q['f'] for q in arms['dev'][cand]['videos']}; y={q['video_id']:q['f'] for q in arms['dev'][ctrl]['videos']}; contrasts[cand+'_minus_'+ctrl]=boot(np.array([x[v]-y[v] for v in x]))
 result={'status':'COMPLETED','experiment_id':c['experiment_id'],'config_sha256':sha(a.config),'script_sha256':sha(__file__),'train_sources':len(train),'dev_sources':len(dev),'protected_opened':False,'pca_components':c['pca_components'],'models':models,'arms':arms,'contrasts':contrasts,'seconds':time.time()-tick,'scope':c['scope'],'limitations':c['limitations']}; dump(out/'metrics.json',result); dump(out/'records.json',rec); print(json.dumps({'seconds':result['seconds'],'dev':{k:v['f'] for k,v in arms['dev'].items()},'contrasts':contrasts},indent=2))
if __name__=='__main__': main()
