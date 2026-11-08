"""Fixed actor-preserving trajectory QP; sparse ADMM, no added model parameters."""
import copy
import numpy as np
from scipy.linalg import cholesky_banded,cho_solve_banded

TAU=.65

def limits(base,x1,x2,width,W):
    overlap=np.maximum(0,np.minimum(base+width,x2)-np.maximum(base,x1))
    lo=np.maximum(0,np.where(overlap>1e-9,np.ceil(x1+overlap-width-1e-8),0.))
    hi=np.minimum(W-width,np.where(overlap>1e-9,np.floor(x2-overlap+1e-8),float(W-width)))
    assert np.all(lo<=base+1e-8) and np.all(base<=hi+1e-8)
    return lo,hi,overlap

def trajectory(base,want,x1,x2,width,W,t,floor=.15):
    b=np.asarray(base,dtype=float);c=np.asarray(want,dtype=float);t=np.asarray(t,dtype=float)
    lo,hi,before=limits(b,x1,x2,width,W);n=len(b)
    if n==1:
        out=np.clip(np.rint(c),lo,hi).astype(int)
        return out,{'solved':True,'iterations':0,'max_constraint_violation_pixels':0.}
    dt=np.diff(t);assert np.all(dt>0)
    lower=lo/W;upper=hi/W;target=c/W
    # Native .15 dimension/second bound, enlarged only to include native integer rounding.
    speed=np.maximum(floor*dt,np.abs(np.diff(b))/W)
    weights=(TAU/dt)**2
    diagonal=np.ones(n);diagonal[:-1]+=weights;diagonal[1:]+=weights
    rho=float(max(1.,np.median(diagonal)))
    def factor(value):
        diag=diagonal+value
        diag=diag.copy();diag[:-1]+=value;diag[1:]+=value
        band=np.zeros((2,n));band[0]=diag;band[1,:-1]=-weights-value
        return cholesky_banded(band,lower=True,check_finite=False)
    def transpose(a,d):
        v=a.copy();v[:-1]-=d;v[1:]+=d;return v
    chol=factor(rho);x=b/W;z=np.clip(x,lower,upper);d=np.clip(np.diff(x),-speed,speed)
    u=np.zeros(n);v=np.zeros(n-1);solved=False;primal=dual=float('inf')
    for iteration in range(1,30001):
        x=cho_solve_banded((chol,True),target+rho*transpose(z-u,d-v),check_finite=False)
        dx=np.diff(x);oldz=z.copy();oldd=d.copy();z=np.clip(x+u,lower,upper);d=np.clip(dx+v,-speed,speed)
        u+=x-z;v+=dx-d
        primal=max(float(np.max(np.abs(x-z))),float(np.max(np.abs(dx-d))))
        dual=float(np.max(np.abs(rho*transpose(z-oldz,d-oldd))))
        # Fractional-image absolute tolerance is <= .00002 px at width1920.
        if primal<1e-8 and dual<1e-7:
            solved=True;break
        if iteration%100==0:
            next_rho=rho
            if primal>10*dual:next_rho=rho*2
            elif dual>10*primal:next_rho=rho/2
            if next_rho!=rho:
                u*=rho/next_rho;v*=rho/next_rho;rho=next_rho;chol=factor(rho)
    assert solved,('Fixed trajectory optimizer did not converge',primal,dual)
    out=np.clip(np.rint(z*W),lo,hi).astype(int)
    after=np.maximum(0,np.minimum(out+width,x2)-np.maximum(out,x1))
    assert np.all(after+1e-7>=before)
    assert np.all(np.abs(np.diff(out))<=speed*W+1.0001)
    objective=lambda q:float(np.sum((q-target)**2)+np.sum(weights*np.diff(q)**2))
    return out,{'solved':True,'iterations':iteration,'max_constraint_violation_pixels':primal*W,'objective':objective(z),'native_objective':objective(b/W),'native_actor_overlap_nonregression':True,'pan_bound_pixels_per_step_rounding_slack':1}

def apply_events(meta,events,anchors,baseline,candidate,qp_floor=.15):
    from clock_rule import adjusted
    pts=np.asarray(meta['pts']);W=meta['width'];out=copy.deepcopy(baseline)
    assert [p['frame'] for p in baseline]==[p['frame'] for p in candidate]
    ids=np.array([p['frame'] for p in baseline],dtype=int)
    assert len(set(ids.tolist()))==len(ids) and np.all(np.diff(ids)>0)
    # The proposal may differ by a rounding pixel on the inactive axis. This projection only reads candidate x and preserves baseline y in its output.
    assert all(a['bboxes'][2]==b['bboxes'][2] for a,b in zip(baseline,candidate))
    nodes=adjusted(anchors,events) if anchors else [];covered=[];audits=[]
    for start,end in events:
        positions=np.flatnonzero((pts[ids]>=start)&(pts[ids]<end));expected=np.flatnonzero((pts>=start)&(pts<end))
        assert ids[positions].tolist()==expected.tolist();covered.extend(positions.tolist())
        if not len(positions):continue
        event_times=pts[ids[positions]];event_nodes=[a for a in nodes if start<=a['time']<end]
        cuts=sorted({int(np.searchsorted(event_times,a['time'])) for a in event_nodes if a.get('scene_cut') and event_times[0]<a['time']<=event_times[-1]})
        bounds=[0]+cuts+[len(positions)]
        for l,r in zip(bounds[:-1],bounds[1:]):
            endshot=event_times[r] if r<len(positions) else end
            valid=[a for a in event_nodes if event_times[l]<=a['time']<endshot and a.get('subject_box') is not None];ix=positions[l:r]
            if not valid:
                audits.append({'solved':True,'reason':'native fallback: no actor in shot','first_frame':int(ids[ix[0]])});continue
            anchor_t=[a['time'] for a in valid];t=event_times[l:r]
            x1=np.interp(t,anchor_t,[a['subject_box'][0]*W for a in valid]);x2=np.interp(t,anchor_t,[a['subject_box'][2]*W for a in valid])
            b=np.array([baseline[i]['bboxes'][0] for i in ix],dtype=float);c=np.array([candidate[i]['bboxes'][0] for i in ix],dtype=float);width=baseline[int(ix[0])]['bboxes'][2]
            assert all(baseline[i]['bboxes'][2]==width for i in ix)
            projected,audit=trajectory(b,c,x1,x2,width,W,t,qp_floor)
            for i,x in zip(ix,projected):out[int(i)]['bboxes'][0]=int(x)
            audits.append(dict(audit,event=[start,end],first_frame=int(ids[ix[0]]),last_frame=int(ids[ix[-1]])))
    assert sorted(covered)==list(range(len(baseline)))
    assert all(a['frame']==b['frame'] and a['bboxes'][1:]==b['bboxes'][1:] for a,b in zip(out,baseline))
    return out,audits
