"""Subject anchors to bounded, smooth target-ratio crops.

Predicted subject boxes are model suggestions, never composition ground truth.
"""
import json, math
import numpy as np
from contracts import crop_box, finite

def parse_subject(text):
    decoder=json.JSONDecoder()
    for i,c in enumerate(text):
        if c!='{':continue
        try:obj,_=decoder.raw_decode(text[i:])
        except json.JSONDecodeError:continue
        if not isinstance(obj,dict) or 'subject_box' not in obj:continue
        box=obj['subject_box']
        if box is None:return None
        if not isinstance(box,list) or len(box)!=4 or not all(finite(v) for v in box):raise ValueError('invalid subject box')
        x1,y1,x2,y2=box
        if not (0<=x1<x2<=1000 and 0<=y1<y2<=1000):raise ValueError('subject outside normalized image')
        if max(box)<=1:raise ValueError('ambiguous 0-1 coordinates; expected 0-1000')
        return [v/1000 for v in box]
    raise ValueError('missing subject_box JSON')

def subject_anchor(model,processor,frame):
    import torch
    from contextlib import nullcontext
    from PIL import Image
    prompt=('Identify the primary subject involved in the main action. Return its tight bounding box as '
            'JSON {"subject_box": [x_min, y_min, x_max, y_max]}. Coordinates MUST be integers normalized '
            'from 0 to 1000 over the complete image, not pixel coordinates. Include the whole subject. '
            'If no clear primary subject exists, return {"subject_box": null}. Return only JSON.')
    messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':prompt}]}]
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    batch=processor(text=[text],images=[Image.fromarray(frame)],do_resize=False,return_tensors='pt').to('cuda:0')
    # Temporal adapters were not trained for spatial framing. Preserve the
    # base spatial model used in the independent crop diagnostic.
    adapter_context=model.disable_adapter() if hasattr(model,'disable_adapter') else nullcontext()
    with adapter_context,torch.inference_mode():result=model.generate(**batch,do_sample=False,max_new_tokens=100,use_cache=True)
    raw=processor.batch_decode(result[:,batch.input_ids.shape[1]:],skip_special_tokens=True)[0]
    error=None
    try:box=parse_subject(raw)
    except ValueError as exc:box=None;error=str(exc)
    return dict(raw=raw,subject_box=box,error=error)

def smooth_crops(pts,segments,anchors,W,H,ratio,tau=.65,max_speed=.15):
    """Full feasible crop size, smoothed centers, <=15% dimension/second.

    Interpolate only inside each event. Missing subjects explicitly fall back to
    center. Filtering restarts at declared cuts and disjoint event boundaries.
    """
    pts=np.asarray(pts);predictions=[];stats=[]
    for start,end in segments:
        ids=np.flatnonzero((pts>=start)&(pts<end))
        if not len(ids):continue
        event=sorted([a for a in anchors if start<=a['time']<end],key=lambda a:a['time'])
        valid=[a for a in event if a.get('subject_box') is not None]
        target=np.full((len(ids),2),.5)
        # Reset at shot boundaries; do not force a pan between unrelated shots.
        cuts=[a['time'] for a in event if a.get('scene_cut')]
        boundaries=[0]+sorted({int(np.searchsorted(pts[ids],t)) for t in cuts if pts[ids][0]<t<=pts[ids][-1]})+[len(ids)]
        smooth=target.copy()
        for left,right in zip(boundaries[:-1],boundaries[1:]):
            shot_start=pts[ids[left]];shot_end=pts[ids[right]] if right<len(ids) else end
            shot_anchors=[a for a in valid if shot_start<=a['time']<shot_end]
            # Account for a subframe interval start before its first selected PTS.
            if left==0:shot_anchors=[a for a in valid if start<=a['time']<shot_end]
            if shot_anchors:
                times=[a['time'] for a in shot_anchors]
                centers=np.array([[(a['subject_box'][0]+a['subject_box'][2])/2,(a['subject_box'][1]+a['subject_box'][3])/2] for a in shot_anchors])
                for axis in range(2):target[left:right,axis]=np.interp(pts[ids[left:right]],times,centers[:,axis])
            # Symmetric low pass avoids the lag of a single causal filter.
            forward=target[left:right].copy();backward=forward.copy()
            for j in range(1,len(forward)):
                dt=pts[ids[left+j]]-pts[ids[left+j-1]];alpha=1-math.exp(-dt/tau)
                forward[j]=forward[j-1]+alpha*(forward[j]-forward[j-1])
            for j in range(len(backward)-2,-1,-1):
                dt=pts[ids[left+j+1]]-pts[ids[left+j]];alpha=1-math.exp(-dt/tau)
                backward[j]=backward[j+1]+alpha*(backward[j]-backward[j+1])
            smooth[left:right]=(forward+backward)/2
            for j in range(left+1,right):
                dt=pts[ids[j]]-pts[ids[j-1]]
                smooth[j]=smooth[j-1]+np.clip(smooth[j]-smooth[j-1],-max_speed*dt,max_speed*dt)
        # Retain subpixel positions; integer rounding would violate a velocity
        # bound on high-fps videos. The contract accepts finite pixel numbers.
        width=crop_box(W,H,ratio)[2];height=width*ratio[1]/ratio[0]
        boxes=[]
        for i,c in zip(ids,smooth):
            box=[float(np.clip(c[0]*W-width/2,0,W-width)),float(np.clip(c[1]*H-height/2,0,H-height)),width]
            predictions.append(dict(frame=int(i),bboxes=box));boxes.append(box)
        stats.append(dict(start=start,end=end,anchors=len(event),valid_subject_anchors=len(valid),center_fallback=not bool(valid),cut_resets=len(boundaries)-2))
    predictions.sort(key=lambda x:x['frame'])
    if len({p['frame'] for p in predictions})!=len(predictions):raise ValueError('overlapping event intervals')
    return predictions,stats

def crop_diagnostics(predictions,pts,anchors,W,H,ratio):
    """Geometry and model-anchor coverage only; no claim of human accuracy."""
    baseline=crop_box(W,H,ratio);coverage={'center':[],'adaptive':[]};speeds=[]
    for anchor in anchors:
        if anchor.get('subject_box') is None or not predictions:continue
        pred=min(predictions,key=lambda p:abs(pts[p['frame']]-anchor['time']))
        x1,y1,x2,y2=anchor['subject_box'];x1*=W;x2*=W;y1*=H;y2*=H
        for name,box in [('center',baseline),('adaptive',pred['bboxes'])]:
            x,y,w=box;h=w*ratio[1]/ratio[0]
            intersection=max(0,min(x+w,x2)-max(x,x1))*max(0,min(y+h,y2)-max(y,y1))
            coverage[name].append(intersection/((x2-x1)*(y2-y1)))
    cuts=[a['time'] for a in anchors if a.get('scene_cut')]
    for previous,current in zip(predictions[:-1],predictions[1:]):
        t0,t1=pts[previous['frame']],pts[current['frame']]
        if current['frame']!=previous['frame']+1 or any(t0<t<=t1 for t in cuts):continue
        delta=np.abs(np.array(current['bboxes'][:2])-np.array(previous['bboxes'][:2]))/np.array([W,H])/(t1-t0)
        speeds.append(float(delta.max()))
    return dict(model_anchor_coverage={k:float(np.mean(v)) if v else None for k,v in coverage.items()},max_normalized_pan_speed=max(speeds,default=0.),p95_normalized_pan_speed=float(np.quantile(speeds,.95)) if speeds else 0.,scope='geometry/self-consistency diagnostic; model subject boxes are not human ground truth')
