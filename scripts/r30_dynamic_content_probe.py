"""R30: frozen VideoXum VPB static/dynamic content probe.

Uses only the allowlisted R29 public evidence.  Features are sampled at the
same 12 native PTS as R24; dynamic features are adjacent VPB differences and
cosine change.  TRAIN-only ridge-logistic readouts are evaluated once on the
exposed DEV split at an exact 85% frame budget.
"""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x): Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")
def fscore(mask, order, k):
    m=mask.sum(1); tp=mask[:,order[:k]].sum(1)
    return float(np.divide(2*tp,k+m,out=np.zeros(len(m)),where=(k+m)>0).mean())
def fixed(v): return np.argsort(-v,kind="stable")

def fit(rows, kind, ridge):
    xs=[]; ys=[]
    for r in rows:
        x=r["feat"][kind]; xs.append(x); ys.append(r["sample_y"])
    x=np.concatenate(xs); y=np.concatenate(ys)
    # Source-balanced frame weights.
    w=np.concatenate([np.full(len(r["sample_y"]),1/(len(rows)*len(r["sample_y"]))) for r in rows])
    mu=(x*w[:,None]).sum(0); sd=np.sqrt(((x-mu)**2*w[:,None]).sum(0)).clip(1e-6)
    x=(x-mu)/sd
    reg=np.ones(x.shape[1]); reg[0]=0
    def obj(b):
        z=x@b; loss=(w*(np.logaddexp(0,z)-y*z)).sum()+ridge*(reg*b*b).sum()/2
        g=x.T@(w*(expit(z)-y))+ridge*reg*b
        return loss,g
    q=minimize(obj,np.zeros(x.shape[1]),jac=True,method="L-BFGS-B",options={"maxiter":500,"ftol":1e-12,"gtol":1e-8})
    if not q.success: raise RuntimeError(q.message)
    return {"kind":kind,"beta":q.x.tolist(),"mu":mu.tolist(),"sd":sd.tolist(),"iterations":int(q.nit)}
def predict(r,m):
    x=(r["feat"][m["kind"]]-m["mu"])/m["sd"]
    return expit(x@np.asarray(m["beta"]))
def bootstrap(d,seed=20261016,reps=10000):
    rng=np.random.default_rng(seed); z=rng.choice(d,(reps,len(d)),replace=True).mean(1)
    return {"mean":float(np.mean(d)),"ci95":np.quantile(z,[.025,.975]).tolist(),"sources":len(d)}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); a=ap.parse_args()
    c=json.loads(Path(a.config).read_text()); out=Path(c["output"])
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True); dump(out/"config.json",c); t=time.time()
    ident=json.loads(Path(c["identity_metrics"]).read_text()); report=json.loads(Path(c["axis_report"]).read_text())
    axes={x["video_id"]:x for x in report["axis"]}; rows=[]
    bindings={x['video_id']:x for x in json.loads(Path(c['bindings']).read_text())}
    for split in ("train","dev"):
      for vid in ident[f"{split}_ids"]:
        ar=axes[vid]; axis=Path(c["data_root"])/ar["axis_array_relpath"]
        assert sha(axis)==ar["axis_array_sha256"]
        z=np.load(axis); pts=z["timestamps"]; mask=z["raw_summary_membership"].astype(bool)
        s=bindings[vid]; ix=np.asarray(s['sample_frame_ids'])
        assert s['split']==split and np.array_equal(pts[ix],s["actual_pts"])
        e=np.load(Path(c["feature_root"])/f"{vid}.npz")["vpb"].astype(np.float64)
        assert e.shape[0]==len(ix)==12 and np.isfinite(e).all()
        # Features at 12 observations. Dynamic terms are causal-safe adjacent changes.
        d=np.diff(e,axis=0,prepend=e[:1]); n=np.linalg.norm(e,axis=1); dn=np.linalg.norm(d,axis=1)
        cos=np.ones(12); cos[1:]=1-np.sum(e[1:]*e[:-1],axis=1)/(n[1:]*n[:-1]+1e-8)
        dd=np.diff(d,axis=0,prepend=d[:1])
        tnorm=2*(pts[ix]-pts[0])/max(pts[-1]-pts[0],1e-8)-1
        # Include intercept, scalar motion summaries and optionally full vectors.
        static=np.concatenate([np.ones((12,1)),e],1)
        dynamic=np.concatenate([np.ones((12,1)),dn[:,None],cos[:,None],np.linalg.norm(dd,axis=1)[:,None],d],1)
        both=np.concatenate([static,dynamic[:,1:]],1)
        pos=np.stack([np.ones(12),tnorm,tnorm**2,tnorm**3],1)
        sample_y=mask.mean(0)[ix]
        owner=np.abs(pts[:,None]-pts[ix][None]).argmin(1)
        rows.append({"id":vid,"split":split,"pts":pts,"mask":mask,"ix":ix,"owner":owner,"sample_y":sample_y,
          "feat":{"position":pos,"static":static,"dynamic":dynamic,"static_dynamic":both}})
    train=[r for r in rows if r["split"]=="train"]; dev=[r for r in rows if r["split"]=="dev"]
    models={k:fit(train,k,c["ridge"]) for k in c["arms"]}
    rec=[]
    for r in rows:
      n=len(r["pts"]); k=round(c["keep"]*n); center=np.argsort(abs(np.arange(n)-(n-1)/2),kind="stable")
      rec.append({"video_id":r["id"],"split":r["split"],"arm":"position_center85","f":fscore(r["mask"],center,k),"kept":k,"frames":n})
      for name,m in models.items():
        p12=predict(r,m); p=p12[r["owner"]]; order=fixed(p)
        rec.append({"video_id":r["id"],"split":r["split"],"arm":name+"85","f":fscore(r["mask"],order,k),"kept":k,"frames":n})
    arms={}
    for split in ("train","dev"):
      arms[split]={}
      for arm in sorted(set(x["arm"] for x in rec)):
        rr=[x for x in rec if x["split"]==split and x["arm"]==arm]
        arms[split][arm]={"f":float(np.mean([x["f"] for x in rr])),"videos":rr}
    contrasts={}
    for cand,ctrl in c["contrasts"]:
      x={q["video_id"]:q["f"] for q in arms["dev"][cand]["videos"]}; y={q["video_id"]:q["f"] for q in arms["dev"][ctrl]["videos"]}
      d=np.array([x[v]-y[v] for v in x]); contrasts[cand+"_minus_"+ctrl]=bootstrap(d)
    result={"status":"COMPLETED","experiment_id":c["experiment_id"],"config_sha256":sha(a.config),"script_sha256":sha(__file__),"train_sources":len(train),"dev_sources":len(dev),"protected_opened":False,"models":models,"arms":arms,"contrasts":contrasts,"seconds":time.time()-t,"scope":c["scope"],"limitations":c["limitations"]}
    dump(out/"metrics.json",result); dump(out/"records.json",rec)
    print(json.dumps({"seconds":result["seconds"],"dev":{k:v["f"] for k,v in arms["dev"].items()},"contrasts":contrasts},indent=2))
if __name__=="__main__": main()
