#!/usr/bin/env python3
"""Audit RetargetVid annotations and optionally score supplied crop files.

Annotation files contain ``left,top,right,bottom`` per frame.  Without the
original DHF1K frames this produces an annotation-only report; it never
pretends that spatial IoU was measured for our model.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

def read_box_file(path: Path):
    rows=[]
    for line in path.read_text().splitlines():
        vals=[int(x.strip()) for x in line.split(',')]
        if len(vals)!=4: raise ValueError(f"bad box in {path}: {line}")
        rows.append(vals)
    return np.asarray(rows, dtype=np.int64)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--output',required=True)
    ap.add_argument('--full-media',action='store_true',help='real media/Adapter/coordinates/exposure audit; output is a directory')
    ap.add_argument('--media-root',type=Path)
    ap.add_argument('--registry-root',type=Path)
    ap.add_argument('--evidence-root',type=Path,default=Path(__file__).resolve().parents[1]/'ext_data')
    a=ap.parse_args(); root=Path(a.root); ann=root/'annotations_all'
    if a.full_media:
      if a.media_root is None or a.registry_root is None:ap.error('--full-media needs --media-root and --registry-root')
      sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ext_data'))
      from aicext.retargetvid_audit import run_full_audit
      return run_full_audit(a.media_root,a.registry_root,root,a.evidence_root,Path(a.output))
    report={'dataset':'RetargetVid','annotation_root':str(ann),'spatial_iou':None,'video_count':0,'annotator_count':0,'ratios':{},'files':[]}
    users=sorted(p for p in ann.glob('annotator_*') if p.is_dir()); report['annotator_count']=len(users)
    by={}
    for u in users:
      for p in sorted(u.glob('*.txt')):
        stem=p.stem; parts=stem.rsplit('_',1)
        if len(parts)!=2 or parts[1] not in ('1-3','3-1'): continue
        vid, ratio=parts; boxes=read_box_file(p)
        by.setdefault((vid,ratio),[]).append((u.name,boxes))
    report['video_count']=len({v for v,r in by})
    for (vid,ratio), entries in sorted(by.items()):
      lengths=[len(x[1]) for x in entries]; coords=np.concatenate([x[1] for x in entries]); mins=coords.min(axis=0); maxs=coords.max(axis=0)
      report['ratios'][ratio]=report['ratios'].get(ratio,0)+1
      report['files'].append({'video_id':vid,'target_ratio':ratio,'annotators':len(entries),'frame_counts':lengths,'consistent_frame_count':len(set(lengths))==1,'coordinate_min':mins.tolist(),'coordinate_max':maxs.tolist()})
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('video_count','annotator_count','ratios','spatial_iou')},indent=2))
if __name__=='__main__': main()
