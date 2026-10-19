#!/usr/bin/env python3
"""Public held-out probe for the exact R58 audio frame-swap mechanism."""
from __future__ import annotations
import argparse,csv,json,subprocess
from pathlib import Path
import numpy as np

def decode(p):
 try:
  raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(p),'-f','f32le','-ac','1','-ar','16000','pipe:1'])
  return np.frombuffer(raw,dtype='<f4').astype(np.float32)
 except Exception:return np.zeros(0,np.float32)

def nov(p,n=30,sr=16000):
 x=decode(p); hz=20.; t=(np.arange(max(1,int(n*hz)))+.5)/hz; w=int(.1*sr); lr=np.zeros(len(t)); cent=np.zeros(len(t))
 for i,u in enumerate(t):
  j=int(round(u*sr)); y=x[max(0,j-w//2):min(len(x),j+w//2)]
  if len(y)<8:continue
  y=y-np.mean(y); rms=np.sqrt(np.mean(y*y)+1e-12); lr[i]=np.log1p(1000*rms)
  z=np.abs(np.fft.rfft(y[:min(len(y),2048)],n=1024))**2; cent[i]=(np.arange(len(z))*z).sum()/(z.sum()+1e-9)/(len(z)-1)
 d=np.maximum(np.diff(lr,prepend=lr[0]),0); local=lr-np.convolve(lr,np.ones(21)/21,mode='same'); s=d+.5*np.maximum(local,0)+.15*cent
 return np.asarray([np.max(s[max(0,int(i*hz-5)):min(len(s),int(i*hz+6))]) for i in range(n)],float)

def f(sel,y,soft):
 yy=y/3 if soft else (y>=2).astype(float); return 2*yy[sel].sum()/(len(sel)+yy.sum())

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--labels',nargs=2,type=Path,required=True); ap.add_argument('--videos',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
 rows=[]
 for si,p in enumerate(a.labels):
  for r in csv.DictReader(p.open()): r['_stream']=si; rows.append(r)
 by={}
 for r in rows:by.setdefault(r['video_id'],[]).append(r)
 out=[]; ds=[]; db=[]
 for vid,rs in sorted(by.items()):
  p=a.videos/f'{vid}.mp4'; s=nov(p); ts=[]; ys=[]; streams=[]
  for r in rs:
   ts.append(int(float(r['time_sec']))); ys.append({'0普通':0.,'1有信息':1.,'2值得保留':2.,'3高光核心':3.}[r['generic_highlight']]); streams.append(int(r['_stream']))
  for st in sorted(set(streams)):
   ix=np.asarray([i for i,x in enumerate(streams) if x==st]); order=np.argsort(np.asarray(ts)[ix]); ix=ix[order]; y=np.asarray(ys)[ix]; n=len(ix); k=round(.85*n); center=1-np.abs(np.arange(n)-(n-1)/2)/max((n-1)/2,1); aud=s[np.asarray(ts)[ix]]; sel=np.argsort(-center)[:k]; base=f(sel,y,True)
   aa=set(sel.tolist()); outside=sorted(set(range(n))-aa,key=lambda i:(aud[i],-i),reverse=True); inside=sorted(aa,key=lambda i:(aud[i],i)); m=min(round(.05*k),len(outside)); drop=set(inside[:m]); add=set(outside[:m]); hybrid=np.asarray(sorted((aa-drop)|add)); val=f(hybrid,y,True); vb=f(hybrid,y,False); b=f(sel,y,False)
   out.append({'video_id':vid,'stream':st,'base_soft':base,'audio_soft':val,'delta_soft':val-base,'base_binary':b,'audio_binary':vb,'delta_binary':vb-b,'swaps':m}); ds.append(val-base);db.append(vb-b)
 res={'experiment_id':'R58_PUBLIC_AUDIO_NOVELTY_SWAP','rows':len(rows),'views':len(out),'macro_delta_soft':float(np.mean(ds)),'positive_views_soft':int(sum(x>0 for x in ds)),'macro_delta_binary':float(np.mean(db)),'positive_views_binary':int(sum(x>0 for x in db)),'per_view':out}; a.output.write_text(json.dumps(res,ensure_ascii=False,indent=2)+'\n'); print(json.dumps({k:res[k] for k in res if k!='per_view'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
