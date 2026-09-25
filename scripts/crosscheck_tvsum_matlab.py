#!/usr/bin/env python3
"""Numerically compare Python masks with unmodified author MATLAB functions in Octave."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import h5py,numpy as np
from scipy.io import savemat,loadmat
from aic.temporal_metrics import fixed_segments,summary_mask
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--author",required=True);ap.add_argument("--octave",required=True)
    ap.add_argument("--mat",required=True);ap.add_argument("--output",required=True);a=ap.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(82)
    scores=[rng.random(600),np.ones(600),np.zeros(20),np.arange(130,dtype=float)]
    with h5py.File(a.mat) as f:
        refs=f["tvsum50/user_anno"][()].ravel()
        for ref in refs[:2]:
            ann=np.asarray(f[ref])
            scores.extend(ann[:3])
    expected=[];sources=[]
    author=Path(a.author)
    for i,p in enumerate(scores):
        inp=out/f"case_{i}.mat";dest=out/f"case_{i}_octave.mat"
        savemat(inp,{"scores":p[:,None],"segments":fixed_segments(len(p))+np.array([1,0])})
        expression=f"addpath('{author}');addpath('{author}/knapsack');load('{inp}');mask=solve_knapsack(scores,segments,0.15);save('-mat7-binary','{dest}','mask');"
        subprocess.run([a.octave,"--quiet","--no-gui","--eval",expression],check=True,capture_output=True)
        actual=loadmat(dest)["mask"].ravel().astype(bool);py=summary_mask(p)
        expected.append({"case":i,"nframes":len(p),"different_frames":int(np.sum(actual!=py))})
    for p in [author/"solve_knapsack.m",author/"knapsack/knapsack.m"]:
        sources.append({"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
    report={"engine":subprocess.check_output([a.octave,"--version"],text=True).splitlines()[0],
            "author_sources":sources,"cases":expected,"all_exact":all(x["different_frames"]==0 for x in expected)}
    (out/"crosscheck.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    if not report["all_exact"]:raise SystemExit(1)
if __name__=="__main__":main()
