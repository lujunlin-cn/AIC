#!/usr/bin/env python3
"""R51 probe: gate D3 residual by persistent two-sided motion.

The gate is fixed per held-out video: a residual is active only when the
mean change before and after the frame are both at least their within-video
medians and their ratio is at least the within-video median ratio.  This is a
structural cut-noise test, not a threshold sweep.  Source-held-out Ridge and
eta=.20 remain unchanged.
"""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.r45_highfreq_semantic_residual import decode_highfreq,highfreq_features
from scripts.r44_local_state_residual import GRADE,_rank01,proxy_f,select

def read(ps):
 out=[]
 for p in ps:
  with Path(p).open(newline='') as f: out+=list(csv.DictReader(f))
 return out
def pred(x,y,c,xt):
 s=StandardScaler().fit(x); A=np.c_[np.ones(len(c)),c]; co=np.linalg.lstsq(A,y,rcond=None)[0]; m=Ridge(alpha=10.).fit(s.transform(x),y-A@co); return m.predict(s.transform(xt))
def evaluate(p,y,t,gate):
 c=1-np.abs(t-(len(t)-1)/2)/max((len(t)-1)/2,1); k=int(round(.85*len(t))); kc=select(c,k); fused=_rank01(c)+.20*(_rank01(p)-.5)*gate; kr=select(fused,k)
 return {'center':proxy_f(kc,y,True),'residual':proxy_f(kr,y,True),'delta':proxy_f(kr,y,True)-proxy_f(kc,y,True),'center_binary':proxy_f(kc,y,False),'residual_binary':proxy_f(kr,y,False),'delta_binary':proxy_f(kr,y,False)-proxy_f(kc,y,False),'swaps':int(len(set(kc.tolist())^set(kr.tolist()))//2),'gate_count':int(gate.sum())}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--labels',nargs=2,type=Path,required=True); ap.add_argument('--videos',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args(); rs=read(a.labels); by={}; data={}
 for r in rs: by.setdefault(r['video_id'],[]).append(r)
 for v in sorted(by):
  fr,_,_=decode_highfreq(a.videos/f'{v}.mp4'); data[v]=highfreq_features(fr).astype(np.float32)
 source=[]
 for held in sorted(by):
  tr=[r for r in rs if r['video_id']!=held]; X=[];Y=[];C=[]
  for r in tr:
   v=r['video_id']; t=min(int(float(r['time_sec'])),29); j=min(4*t,len(data[v])-1); X.append(data[v][j]); Y.append(GRADE[r['generic_highlight']]); C.append(1-abs(t-14.5)/14.5)
  X=np.asarray(X,np.float32); Y=np.asarray(Y,float); C=np.asarray(C,float); grouped={}
  for r in by[held]: grouped.setdefault(int(float(r['time_sec'])),[]).append(r)
  t=np.asarray(sorted(grouped),int); y=np.asarray([np.mean([GRADE[r['generic_highlight']] for r in grouped[u]]) for u in t],float); h=data[held]; j=np.minimum(4*t,len(h)-1); p=pred(X,Y,C,h[j])
  bm=h[j,11]; am=h[j,12]; energy=.5*(bm+am); ratio=np.minimum(bm,am)/(np.maximum(bm,am)+1e-6); gate=((energy>=np.median(energy))&(ratio>=np.median(ratio))).astype(float)
  source.append({'video_id':held,'d3':evaluate(p,y,t,np.ones(len(t))), 'persistent':evaluate(p,y,t,gate)})
 out={'experiment_id':'R51_PERSISTENT_TRANSITION_GATE','status':'PUBLIC_SOURCE_HELD_OUT_COMPLETED','rows':len(rs),'sources':len(by),'eta':.20,'alpha':10.,'gate':'energy >= within-video median AND two-sided continuity ratio >= within-video median','official_gt':False,'macro':{}}
 for n in ('d3','persistent'):
  vals=[x[n] for x in source]; out['macro'][n]={'center':float(np.mean([z['center'] for z in vals])),'residual':float(np.mean([z['residual'] for z in vals])),'delta':float(np.mean([z['delta'] for z in vals])),'positive_sources':int(sum(z['delta']>0 for z in vals)),'center_binary':float(np.mean([z['center_binary'] for z in vals])),'residual_binary':float(np.mean([z['residual_binary'] for z in vals])),'delta_binary':float(np.mean([z['delta_binary'] for z in vals])),'positive_sources_binary':int(sum(z['delta_binary']>0 for z in vals))}
 out['source']=source; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(out['macro'],ensure_ascii=False,indent=2))
if __name__=='__main__': main()
