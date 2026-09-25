#!/usr/bin/env python3
"""Reproduce native RetargetVid negative-coordinate clamp and inclusive IoU."""
import argparse,ast,json,hashlib
from pathlib import Path
import numpy as np
from scripts.benchmark_spatial import iou

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--previous',required=True);ap.add_argument('--annotations',required=True)
    ap.add_argument('--upstream',required=True);ap.add_argument('--reference',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();root=Path(a.previous);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    report=json.loads((root/'metrics.json').read_text());manifest={r['video_id']:r for r in json.loads((root/'manifest.json').read_text())}
    tree=ast.parse(Path(a.upstream).read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='bb_intersection_over_union')
    ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),'upstream','exec'),ns)
    truth={};negative=0
    for vid in manifest:
        for ratio in ['1-3','3-1']:
            raw=np.stack([np.loadtxt(Path(a.annotations)/f'annotator_{i}'/(vid+'_'+ratio+'.txt'),delimiter=',') for i in range(1,7)])
            negative+=int((raw<0).sum());truth[(vid,ratio)]=np.maximum(raw,0)
    rows=[];errors=[]
    for rec in report['per_video']:
        vid,ratio,mode=rec['video_id'],rec['ratio'],rec['mode'];pred=np.loadtxt(root/mode/(vid+'_'+ratio+'.txt'),delimiter=',');gt=truth[(vid,ratio)]
        pred=np.maximum(pred,0);overlap=iou(pred[None],gt);cx=(pred[:,:2]+pred[:,2:])/2;gc=(gt[:,:,:2]+gt[:,:,2:])/2
        shape=manifest[vid];error=np.linalg.norm(gc-cx[None],axis=-1)/np.hypot(shape['width'],shape['height'])
        for user in range(6):
            for f in range(len(pred)):
                errors.append(abs(overlap[user,f]-ns['bb_intersection_over_union'](gt[user,f],pred[f])))
        rows.append({**rec,'iou_before_native_clamp':rec['iou'],'iou':float(overlap.mean()),'center_error_normalized':float(error.mean())})
    references=[]
    for (vid,ratio),gt in truth.items():
        p=Path(a.reference)/(vid+'_'+ratio+'.txt');pred=np.maximum(np.loadtxt(p,delimiter=','),0)
        if pred.shape!=gt.shape[1:]:raise ValueError('reference frame mismatch')
        references.append({'video_id':vid,'ratio':ratio,'mean_iou':float(iou(pred[None],gt).mean()),'prediction_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    report.update(protocol='RETARGETVID_DENSE_NATIVE_IOU_V2',supersedes=str(root/'metrics.json'),
        correction='Exact upstream max(coord,0) applied independently to GT and predictions before inclusive-pixel IoU',
        gt_negative_coordinates=negative,per_video=rows,upstream_iou_max_error=max(errors),upstream_crosscheck_count=len(errors),
        evaluator_sha256=hashlib.sha256(Path(a.upstream).read_bytes()).hexdigest(),
        methods={m:{r:float(np.mean([x['iou'] for x in rows if x['mode']==m and x['ratio']==r])) for r in ['1-3','3-1']} for m in report['methods']},
        upstream_reference={'source':'released SmartVidCrop predictions, not our execution or runtime','per_video':references,
                            'means':{r:float(np.mean([x['mean_iou'] for x in references if x['ratio']==r])) for r in ['1-3','3-1']}})
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n');(out/'manifest.json').write_text((root/'manifest.json').read_text())
    print(json.dumps({k:report[k] for k in ['methods','gt_negative_coordinates','upstream_iou_max_error','upstream_crosscheck_count']},indent=2))
if __name__=='__main__':main()
