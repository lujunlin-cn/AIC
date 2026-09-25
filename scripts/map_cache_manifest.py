#!/usr/bin/env python3
"""Rewrite feature-manifest cache paths for an alternate encoder cache root."""
from __future__ import annotations
import argparse, json
from pathlib import Path
def main(argv=None):
 ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--cache-dir',required=True); ap.add_argument('--output',required=True); a=ap.parse_args(argv)
 rows=[]
 for line in Path(a.input).read_text().splitlines():
  if not line.strip(): continue
  row=json.loads(line); row['path']=str(Path(a.cache_dir)/(Path(row['path']).name)); rows.append(row)
 Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text('\n'.join(json.dumps(r,ensure_ascii=False,sort_keys=True) for r in rows)+'\n')
 print(json.dumps({'input':a.input,'output':a.output,'count':len(rows),'cache_dir':a.cache_dir}))
if __name__=='__main__': main()
