#!/usr/bin/env python3
"""R52 probe: optical-flow motion residual versus grayscale D3.

Mean pixel difference confounds camera exposure, compression and cuts with
object motion.  This arm adds fixed Farneback flow statistics (magnitude,
tail, coherence and divergence) at the same 4-fps anchors.  It uses one
source-held-out Ridge and fixed eta=.20/alpha=10; no feature or weight sweep.
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

def flow_features(frames):
 gray=[cv2.resize(cv2.cvtColor(x,cv2.COLOR_BGR2GRAY),(160,90),interpolation=cv2.INTER_AREA).astype(np.float32)/255 for x in frames]
 flows=[]
 for i in range(max(len(gray)-1,0)):
  f=cv2.calcOpticalFlowFarneback(gray[i],gray[i+1],None,.5,2,15,2,5,1.2,0)
  mag,ang=cv2.cartToPolar(f[...,0],f[...,1]);
  # Direction coherence is the length of the mean unit vector; it is high
  # for camera/object translation and low for unstructured noise.
  coh=float(np.hypot(np.mean(np.cos(ang)),np.mean(np.sin(ang))))
  dx=cv2.Sobel(f[...,0],cv2.CV_32F,1,0); dy=cv2.Sobel(f[...,1],cv2.CV_32F,0,1)
  flows.append([float(np.mean(mag)),float(np.std(mag)),float(np.percentile(mag,90)),float(np.mean(mag>1.0)),coh,float(np.mean(np.abs(dx+dy)))])
 flows=np.asarray(flows,np.float32); out=[]
 for i in range(len(frames)):
  b=flows[i-1] if i>0 and len(flows) else np.zeros(6,np.float32); a=flows[i] if i<len(flows) else np.zeros(6,np.float32)
  lo=max(0,i-2); hi=min(len(flows),i+2); local=flows[lo:hi] if hi>lo else np.zeros((1,6),np.float32)
  out.append(np.r_[b,a,.5*(b+a),a-b,np.max(local,axis=0),np.mean(local,axis=0)])
 return np.asarray(out,np.float32)

def fit(x,y,c,xt):
 s=StandardScaler().fit(x); A=np.c_[np.ones(len(c)),c]; co=np.linalg.lstsq(A,y,rcond=None)[0]; m=Ridge(alpha=10.).fit(s.transform(x),y-A@co); return m.predict(s.transform(xt))
def score(p,y,t):
 c=1-np.abs(t-(len(t)-1)/2)/max((len(t)-1)/2,1); k=int(round(.85*len(t))); kc=select(c,k); kr=select(_rank01(c)+.2*(_rank01(p)-.5),k)
 return {'center':proxy_f(kc,y,True),'residual':proxy_f(kr,y,True),'delta':proxy_f(kr,y,True)-proxy_f(kc,y,True),'center_binary':proxy_f(kc,y,False),'residual_binary':proxy_f(kr,y,False),'delta_binary':proxy_f(kr,y,False)-proxy_f(kc,y,False),'swaps':int(len(set(kc.tolist())^set(kr.tolist()))//2)}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--labels',nargs=2,type=Path,required=True); ap.add_argument('--videos',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args(); rs=read(a.labels); by={}; data={}
 for r in rs: by.setdefault(r['video_id'],[]).append(r)
 for v in sorted(by):
  fr,_,_=decode_highfreq(a.videos/f'{v}.mp4'); h=highfreq_features(fr).astype(np.float32); f=flow_features(fr); data[v]=(h,f)
 source=[]
 for held in sorted(by):
  tr=[r for r in rs if r['video_id']!=held]; Xd=[]; Xf=[]; Y=[]; C=[]
  for r in tr:
   v=r['video_id']; t=min(int(float(r['time_sec'])),29); j=min(4*t,len(data[v][0])-1); Xd.append(data[v][0][j]); Xf.append(np.r_[data[v][0][j],data[v][1][j]]); Y.append(GRADE[r['generic_highlight']]); C.append(1-abs(t-14.5)/14.5)
  grouped={}
  for r in by[held]: grouped.setdefault(int(float(r['time_sec'])),[]).append(r)
  t=np.asarray(sorted(grouped),int); y=np.asarray([np.mean([GRADE[r['generic_highlight']] for r in grouped[u]]) for u in t],float); h,f=data[held]; j=np.minimum(4*t,len(h)-1); Xd=np.asarray(Xd,np.float32); Xf=np.asarray(Xf,np.float32); Y=np.asarray(Y,float); C=np.asarray(C,float)
  source.append({'video_id':held,'d3':score(fit(Xd,Y,C,h[j]),y,t),'flow':score(fit(Xf,Y,C,np.c_[h[j],f[j]]),y,t)})
 out={'experiment_id':'R52_OPTICAL_FLOW_RESIDUAL_PROBE','status':'PUBLIC_SOURCE_HELD_OUT_COMPLETED','rows':len(rs),'sources':len(by),'eta':.20,'alpha':10.,'feature_contract':'D3 4fps local state + fixed Farneback flow magnitude/tail/coherence/divergence statistics','official_gt':False,'source':source,'macro':{}}
 for n in ('d3','flow'):
  z=[x[n] for x in source]; out['macro'][n]={'center':float(np.mean([q['center'] for q in z])),'residual':float(np.mean([q['residual'] for q in z])),'delta':float(np.mean([q['delta'] for q in z])),'positive_sources':int(sum(q['delta']>0 for q in z)),'center_binary':float(np.mean([q['center_binary'] for q in z])),'residual_binary':float(np.mean([q['residual_binary'] for q in z])),'delta_binary':float(np.mean([q['delta_binary'] for q in z])),'positive_sources_binary':int(sum(q['delta_binary']>0 for q in z))}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(out['macro'],ensure_ascii=False,indent=2))
if __name__=='__main__': main()
