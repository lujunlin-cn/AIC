#!/usr/bin/env python3
"""Rebuild PHD2 union/soft labels without re-running DINOv3."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from dinov3_extract_phd2_frame_cache import load_annotations, labels

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cache',type=Path,required=True); ap.add_argument('--annotations',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args(); ann=load_annotations(a.annotations); a.out.mkdir(parents=True,exist_ok=True); rows=[]; max_soft=0.; old_max=0.; invalid_frames=0; changed_rows=0; union_changes=0
    for line in (a.cache/'manifest.jsonl').read_text().splitlines():
        r=json.loads(line); z=np.load(r['path']); u,s,n=labels(z['timestamps'],ann.get(r['src'],[])); max_soft=max(max_soft,float(s.max(initial=0)))
        old_max=max(old_max,float(z['soft'].max(initial=0))); invalid_frames+=int((z['soft']>1).sum()); changed_rows+=int(not np.array_equal(s,z['soft'])); union_changes+=int(not np.array_equal(u,z['union']))
        target=a.out/Path(r['path']).name
        payload={k:z[k] for k in z.files if k not in ('union','soft','user_count','metadata_json')}
        payload.update(union=u,soft=s,user_count=np.asarray(n,dtype=np.int32),metadata_json=np.asarray(json.dumps({'label_semantics':'merged per-user intervals; absence remains unobserved','source':r['src']},sort_keys=True)))
        np.savez_compressed(target,**payload); rows.append({**r,'path':str(target),'users':n})
    report={'rows':len(rows),'max_soft':max_soft,'old_max_soft':old_max,'old_out_of_range_frames':invalid_frames,'changed_soft_rows':changed_rows,'changed_union_rows':union_changes,'label_fix':'merge overlapping intervals within each user before vote density','source_cache_unchanged':str(a.cache),'warning':'unselected times are observed nonselections, not confirmed semantic negatives; existing all-true mask is not validity supervision'}
    (a.out/'manifest.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n'); (a.out/'relabel_meta.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report))
if __name__=='__main__': main()
