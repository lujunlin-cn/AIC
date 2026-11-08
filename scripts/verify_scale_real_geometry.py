"""Audit real public human boxes and historical P1-center outputs without training."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
from aic.scale_geometry import checked_scale_action, deployed_center, transpose_xywh


def verify(snapshot: Path, bins):
    root=snapshot/'repro/liveytvc_full_retrain_20261023'
    labels={r['id']:r for r in json.loads((root/'data/dev_a.json').read_text())}
    results={}
    for arm,key,file in [('a','a:new_a','arm_a_gate.raw.json'),('b','b:new_b','arm_b_gate.raw.json')]:
        raw=json.loads((root/'eval'/file).read_text())[key]
        ratio=[9,16] if arm=='a' else [16,9]
        statuses=Counter();max_drift=0.;raw_shifts=0;frames=0;centers=[]
        for source,rows in raw.items():
            for row in rows:
                label=labels[f"{source}-frame{row['frame_id']}"]
                W,H=label['original_size']
                if arm=='b':W,H=H,W
                response=json.loads(row['raw']);cx,cy=response['crop_center']
                raw_center=[cx*W/1000,cy*H/1000]
                maximum=checked_scale_action(W,H,ratio,raw_center,1.)
                raw_shifts+=not maximum['valid']
                # The existing P1 parser projects its maximum window. This is
                # the center of that deployed P1 output, not a new Scale clamp.
                center=deployed_center(maximum['aic_bbox_xyw'],ratio);centers.append(center)
                for scale in bins:
                    action=checked_scale_action(W,H,ratio,center,scale)
                    statuses[action['status']]+=1
                    if action['valid']:max_drift=max(max_drift,action['position_drift_px'])
                    box=action['crop_xywh'];flipped,dims=transpose_xywh(box,W,H)
                    restored,_=transpose_xywh(flipped,*dims)
                    assert max(abs(a-b) for a,b in zip(box,restored))<1e-6
                    assert 0<=box[0] and box[0]+box[2]<=W+1e-6
                    assert 0<=box[1] and box[1]+box[3]<=H+1e-6
                frames+=1
        results[arm]={'historical_public_dev_outputs':frames,'action_status_counts':dict(statuses),
                      'maximum_valid_center_drift_px':max_drift,'raw_requested_center_shift_count':raw_shifts,
                      'p1_deployed_center_range':{'x':[min(c[0] for c in centers),max(c[0] for c in centers)],
                                                  'y':[min(c[1] for c in centers),max(c[1] for c in centers)]},
                      'source':'historical raw public-dev center output; not a new real-model run',
                      'training_labels_used':False}
    return {'status':'historical_dev_geometry_verified','arms':results,
            'training_geometry_1000_per_arm_requirement_passed':False,
            'remaining':'Need hash-bound actual training P1 deployed centers; dev audit does not replace this gate.'}


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--bins',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=verify(a.snapshot,json.loads(a.bins.read_text())['values'])
    a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
