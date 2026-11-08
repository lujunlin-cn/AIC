"""Join public labels with a real P1 deployed-center cache; no missing-row fallback."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from aic.scale_data import read_jsonl, attach_frozen_positions
from aic.scale_geometry import deployed_center, checked_scale_action


def prepare(labels, positions, parent_sha):
    allowed={'LIVE_YT_VC','GNMC','GAICD','SACD'}
    for row in labels+positions:
        if row.get('dataset_source') not in allowed or row.get('split') not in {'train','val'}:
            raise ValueError('only named public training/validation sources are accepted')
    converted=[]
    for row in positions:
        if row.get('parent_identity_sha256')!=parent_sha or row.get('position_kind')!='P1_final_deployed':
            raise ValueError('P1 center cache lacks deployed-path identity/provenance')
        center=list(deployed_center(row['p1_aic_bbox_xyw'],row['target_ratio']))
        if row.get('center_xy')!=center:
            raise ValueError('P1 cache center does not match actual deployed bbox')
        baseline=checked_scale_action(row['image_width'],row['image_height'],row['target_ratio'],center,1.)
        if not baseline['valid'] or baseline['aic_bbox_xyw']!=row['p1_aic_bbox_xyw']:
            raise ValueError('Scale=1 must exactly reproduce actual P1 deployed baseline')
        for field in ('selected_frames_sha256','path_cache_sha256','position_protocol_sha256','spatial_entrypoint_sha256'):
            if not row.get(field):raise ValueError('P1 cache is missing '+field)
        if row.get('qp_floor')!=.15:raise ValueError('P1 QP floor must be .15')
        converted.append(dict(row,center_xy=center))
    joined=attach_frozen_positions(labels,converted)
    source_sets={s:{(r['dataset_source'],r['source_id']) for r in joined if r['split']==s} for s in ['train','val']}
    if source_sets['train']&source_sets['val']:raise ValueError('source leakage')
    hashes={s:{r['image_sha256'] for r in joined if r['split']==s} for s in ['train','val']}
    if hashes['train']&hashes['val']:raise ValueError('identical input media crosses train/validation')
    return joined


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--positions',type=Path,required=True);p.add_argument('--parent-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    joined=prepare(read_jsonl(a.labels),read_jsonl(a.positions),a.parent_sha256)
    a.output.mkdir(parents=True,exist_ok=False);manifest={}
    for split in ['train','val']:
        path=a.output/(split+'.jsonl')
        path.write_text(''.join(json.dumps(r)+'\n' for r in joined if r['split']==split))
        manifest[split]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                         'rows':sum(r['split']==split for r in joined)}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
