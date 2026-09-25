#!/usr/bin/env python3
"""Aggregate per-fold threshold-sweep JSON artifacts."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
def main(argv=None):
 ap=argparse.ArgumentParser(); ap.add_argument('--prefix',required=True); ap.add_argument('--output',required=True); a=ap.parse_args(argv)
 rows=[]
 for i in range(5):
  p=Path(f'{a.prefix}{i}/thresholds.json'); d=json.loads(p.read_text()); rows.append(d)
 thresholds=d['thresholds']; out=[]
 for j,t in enumerate(thresholds):
  vals=[float(r['thresholds'][j]['video_macro_f1']) for r in rows]
  out.append({'threshold':float(t['threshold']),'fold_values':vals,'mean':float(np.mean(vals)),'std':float(np.std(vals,ddof=1)),'median':float(np.median(vals))})
 result={'prefix':a.prefix,'thresholds':out,'best_by_mean':max(out,key=lambda x:x['mean'])}
 Path(a.output).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
