#!/usr/bin/env python3
"""Extract dense, time aligned audio novelty features from AIC test videos.

This is deliberately model free: 20 Hz RMS/peak/ZCR plus positive novelty and
spectral band statistics.  It is an offline diagnostic/generation input and
does not access labels.  A small output JSON is enough to drive a frame-set
swap against a TimeLens candidate.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import av

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--video-root',type=Path,required=True); ap.add_argument('--index',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--hz',type=float,default=20.0); args=ap.parse_args()
    idx={str(x['video_id']):x for x in (json.loads(l) for l in args.index.read_text().splitlines() if l.strip())}
    out={}
    for vid in sorted(idx,key=lambda x:int(x) if x.isdigit() else x):
        meta=idx[vid]; p=args.video_root/f'{vid}.mp4'; rec={'video_id':vid,'present':False}
        if not p.exists(): rec['error']='missing'; out[vid]=rec; continue
        try:
            c=av.open(str(p)); streams=c.streams.audio
            if not streams: c.close(); out[vid]=rec; continue
            st=streams[0]; rate=int(st.rate or 48000); vals=[]; times=[]
            for fr in c.decode(audio=0):
                arr=fr.to_ndarray()
                if arr.ndim>1: arr=arr.mean(axis=0)
                x=arr.astype(np.float32)
                # PyAV float AAC is usually [-1,1], integer PCM is scaled.
                if np.max(np.abs(x),initial=0)>2: x=x/32768.0
                t0=float(fr.pts*fr.time_base) if fr.pts is not None else (times[-1]+len(x)/rate if times else 0.0)
                vals.append(x); times.append(t0)
            c.close()
            if not vals: out[vid]=rec; continue
            x=np.concatenate(vals); sr=rate; dur=max(float(meta.get('duration',len(x)/sr)),len(x)/sr)
            hz=float(args.hz); n=max(1,int(math.ceil(dur*hz))); centers=(np.arange(n)+.5)/hz
            # energy and high-frequency novelty in short centered windows.
            win=max(16,int(round(.10*sr))); half=win//2; rms=np.zeros(n); peak=np.zeros(n); zcr=np.zeros(n); centroid=np.zeros(n)
            for i,t in enumerate(centers):
                j=int(round(t*sr)); lo=max(0,j-half); hi=min(len(x),j+half); y=x[lo:hi]
                if len(y)<8: continue
                y=y-np.mean(y); rms[i]=np.sqrt(np.mean(y*y)+1e-12); peak[i]=np.max(np.abs(y)); zcr[i]=np.mean(y[1:]*y[:-1]<0) if len(y)>1 else 0
                # cheap spectral centroid on a 1024-point FFT
                z=np.abs(np.fft.rfft(y[:min(len(y),2048)],n=1024))**2; centroid[i]=float((np.arange(len(z))*z).sum()/(z.sum()+1e-9))/(len(z)-1)
            lr=np.log1p(1000*rms); d=np.diff(lr,prepend=lr[0]); novelty=np.maximum(d,0)
            # local peak emphasis catches applause/claps while suppressing a loud constant soundtrack.
            local=lr-np.convolve(lr,np.ones(21)/21,mode='same'); score=novelty+0.5*np.maximum(local,0)+0.15*centroid
            rec.update({'present':True,'hz':hz,'times':centers.tolist(),'rms':rms.tolist(),'peak':peak.tolist(),'zcr':zcr.tolist(),'centroid':centroid.tolist(),'novelty':score.astype(np.float32).tolist()})
        except Exception as e: rec['error']=f'{type(e).__name__}:{e}'
        out[vid]=rec
        if len(out)%20==0: print(len(out),vid,rec.get('present'),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(out,separators=(',',':'))+'\n'); print('DONE',len(out),sum(bool(x.get('present')) for x in out.values()))
if __name__=='__main__': main()
