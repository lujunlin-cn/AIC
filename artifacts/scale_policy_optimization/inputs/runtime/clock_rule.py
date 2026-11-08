"""Exact v81 anchor-time rule; no fitted constants."""
import numpy as np

def adjusted(anchors,segments):
    result=[]
    for a in anchors:
        s,e=next((s,e) for s,e in segments if s<=a['time']<e)
        t=float(a['sample_time']);assert np.isfinite(t)
        nt=max(s,min(t,float(np.nextafter(e,s))))
        result.append(dict(a,requested_time=a['time'],time=nt,event_clipped=nt!=t))
    return result
