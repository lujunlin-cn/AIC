#!/usr/bin/env python3
"""Controlled ablation for R50 global trajectory context.

Compares the existing D3 local contract, D3 plus full trajectory context, and
the same context with raw normalized position removed.  All arms use the same
source-held-out Ridge, centre anchor, eta=.20 and exact 85% proxy decision.
"""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import cv2,numpy as np
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
def gfeat(fr):
 g=np.stack([cv2.cvtColor(x,cv2.COLOR_BGR2GRAY).astype(np.float32)/255 for x in fr]); n=len(g)
 d=np.mean(np.abs(g[1:]-g[:-1]),axis=(1,2)) if n>1 else np.zeros(0,np.float32); cum=np.r_[0.,np.cumsum(d)]; tot=max(float(cum[-1]),1e-6); med=np.median(g,axis=0); first,last=g[0],g[-1]; z=[]
 for i in range(n):
  b=cum[i]/tot; a=(tot-cum[i])/tot
  z.append([np.mean(np.abs(g[i]-first)),np.mean(np.abs(g[i]-last)),np.mean(np.abs(g[i]-med)),b,a,b-a,i/max(n-1,1),np.mean(d[max(0,i-8):i+1]) if len(d) else 0.,np.mean(d[i:min(len(d),i+8)]) if len(d) else 0.])
 return np.asarray(z,np.float32)
def pred(x,y,c,xt):
 s=StandardScaler().fit(x); A=np.c_[np.ones(len(c)),c]; co=np.linalg.lstsq(A,y,rcond=None)[0]; m=Ridge(alpha=10.).fit(s.transform(x),y-A@co); return m.predict(s.transform(xt))
def score(q,y,t,eta=.2):
 c=1-np.abs(t-(len(t)-1)/2)/max((len(t)-1)/2,1); k=int(round(.85*len(t))); kc=select(c,k); kr=select(_rank01(c)+eta*(_rank01(q)-.5),k)
 return proxy_f(kc,y,True),proxy_f(kr,y,True),proxy_f(kc,y,False),proxy_f(kr,y,False),int(len(set(kc.tolist())^set(kr.tolist()))//2)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--labels',nargs=2,type=Path,required=True); ap.add_argument('--videos',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args(); rs=read(a.labels); by={}; data={}
 for r in rs: by.setdefault(r['video_id'],[]).append(r)
 for v in sorted(by):
  fr,_,_=decode_highfreq(a.videos/f'{v}.mp4'); h=highfreq_features(fr).astype(np.float32); data[v]=(h,gfeat(fr))
 arms={'D3_LOCAL':[],'GLOBAL_FULL':[],'GLOBAL_NO_POSITION':[]}
 for held in sorted(by):
  tr=[r for r in rs if r['video_id']!=held]; grouped={}
  for r in by[held]: grouped.setdefault(int(float(r['time_sec'])),[]).append(r)
  t=np.asarray(sorted(grouped),int); y=np.asarray([np.mean([GRADE[r['generic_highlight']] for r in grouped[u]]) for u in t],float); c=1-np.abs(t-14.5)/14.5
  Xloc=[];Xfull=[];Xnop=[];Y=[];C=[]
  for r in tr:
   v=r['video_id']; u=min(int(float(r['time_sec'])),29); j=min(4*u,len(data[v][0])-1); h,g=data[v]; Xloc.append(h[j]); Xfull.append(np.r_[h[j],g[j]]); Xnop.append(np.r_[h[j],g[j,:6],g[j,7:]]); Y.append(GRADE[r['generic_highlight']]); C.append(1-abs(u-14.5)/14.5)
  h,g=data[held]; j=np.minimum(4*t,len(h)-1); xt={'D3_LOCAL':h[j],'GLOBAL_FULL':np.c_[h[j],g[j]],'GLOBAL_NO_POSITION':np.c_[h[j],g[j,:6],g[j,7:]]}; trxs={'D3_LOCAL':np.asarray(Xloc,np.float32),'GLOBAL_FULL':np.asarray(Xfull,np.float32),'GLOBAL_NO_POSITION':np.asarray(Xnop,np.float32)}; Y=np.asarray(Y,float); C=np.asarray(C,float)
  for name in arms:
   p=pred(trxs[name],Y,C,xt[name]); z=score(p,y,t); arms[name].append({'video_id':held,'center':z[0],'residual':z[1],'delta':z[1]-z[0],'center_binary':z[2],'residual_binary':z[3],'delta_binary':z[3]-z[2],'swaps':z[4]})
 out={'experiment_id':'R50_GLOBAL_CONTEXT_ABLATION','status':'PUBLIC_SOURCE_HELD_OUT_COMPLETED','rows':len(rs),'sources':len(by),'eta':.2,'alpha':10.,'arms':{}}
 for name,vals in arms.items(): out['arms'][name]={'macro_center':float(np.mean([x['center'] for x in vals])),'macro_residual':float(np.mean([x['residual'] for x in vals])),'macro_delta':float(np.mean([x['delta'] for x in vals])),'positive_sources':int(sum(x['delta']>0 for x in vals)),'macro_center_binary':float(np.mean([x['center_binary'] for x in vals])),'macro_residual_binary':float(np.mean([x['residual_binary'] for x in vals])),'macro_delta_binary':float(np.mean([x['delta_binary'] for x in vals])),'positive_sources_binary':int(sum(x['delta_binary']>0 for x in vals)),'source':vals}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(out['arms'],ensure_ascii=False,indent=2))
if __name__=='__main__': main()
