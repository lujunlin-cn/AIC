import json, glob, os, numpy as np
D = "/root/autodl-tmp/vpb/scores_fuse_gifens_a025"
def rankdata(a):
    a=np.asarray(a,float); o=np.argsort(a,kind="mergesort")
    r=np.empty(len(a)); sa=a[o]; i=0
    while i<len(a):
        j=i
        while j+1<len(a) and sa[j+1]==sa[i]: j+=1
        r[o[i:j+1]]=0.5*(i+j)+1.0; i=j+1
    return r
def sp(a,b):
    if np.std(a)<1e-12 or np.std(b)<1e-12: return 0.0
    return float(np.corrcoef(rankdata(a),rankdata(b))[0,1])
uniq=[]; npts=[]; mono=[]; sgn=[]
for p in sorted(glob.glob(os.path.join(D,"*.json"))):
    s=np.asarray(json.load(open(p))["scores"],float)
    if len(s)<8: continue
    npts.append(len(s))
    uniq.append(len(set(np.round(s,6)))/len(s))
    rr=sp(s,np.arange(len(s),dtype=float)); mono.append(abs(rr)); sgn.append(np.sign(rr))
u=np.array(uniq); n=np.array(npts); mo=np.array(mono); sg=np.array(sgn)
print("采样点数分布:", dict(zip(*np.unique(n, return_counts=True))))
print(f"唯一分数比 unique/n: min={u.min():.3f} p05={np.percentile(u,5):.3f} 中位={np.median(u):.3f}")
print(f"  完全常数(unique<=1.05/n): {int((u<=1.05/np.array(n)).sum())}/{len(u)}")
print(f"  近乎唯一(unique<0.5)    : {int((u<0.5).sum())}/{len(u)}")
for t in (0.8,0.9,0.95):
    m=mo>=t
    print(f"|mono_rho|>={t}: {int(m.sum())}/{len(m)}  其中上升 {int(((sg>0)&m).sum())} / 下降 {int(((sg<0)&m).sum())}")
