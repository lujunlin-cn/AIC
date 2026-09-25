#!/usr/bin/env python3
"""Compare saved Python predictions to unmodified SumMe author evaluation."""
import argparse,json,subprocess,os
from pathlib import Path
import numpy as np
from scipy.io import savemat

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',required=True);ap.add_argument('--root',required=True)
    ap.add_argument('--octave',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);cases=[]
    for p in sorted(Path(a.results).glob('*/metrics.json')):
        d=json.loads(p.read_text())
        for row in d['per_video']:
            z=np.load(p.parent/(row['video_id']+'.npz'));case=out/(p.parent.name+'_'+row['video_id']+'.mat')
            savemat(case,{'summary_selection':z['summary_mask'].astype(float)[:,None]})
            def q(s):return "'"+str(s).replace("'","''")+"'"
            script=f"addpath({q(Path(a.root)/'matlab')});load({q(case)});[f,s,r,m]=summe_evaluateSummary(summary_selection,{q(row['video_id'])},{q(Path(a.root)/'GT')});fprintf('CHECK %.17g %.17g\\n',f,m);"
            env=os.environ.copy();home=str(Path(a.octave).parent.parent);env.update(OCTAVE_HOME=home,OCTAVE_EXEC_HOME=home)
            r=subprocess.run([a.octave,'--quiet','--no-gui','--eval',script],env=env,text=True,capture_output=True,check=True)
            line=next(x for x in r.stdout.splitlines() if x.startswith('CHECK '));f,m=map(float,line.split()[1:])
            cases.append({'model':d['model'],'video_id':row['video_id'],'mean_error':abs(f-row['native_mean_user_f1']),'max_error':abs(m-row['native_max_user_f1'])})
    report={'cases':cases,'max_error':max(max(c['mean_error'],c['max_error']) for c in cases),'upstream':'unmodified summe_evaluateSummary.m','interpreter':'GNU Octave'}
    if report['max_error']>1e-12:raise ValueError(report)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
