"""Independently inspect submitted ZIP bytes and original cached P2 geometry.

Optional full media decode checks original IDs/PTS once per source for all four
packages, using CPU only. No labels, learned evaluators or submission API.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import importlib.util
import json
import math
from pathlib import Path
import time
import zipfile

from aic.scale_policy import POLICIES, atomic_json, canonical_sha, sha
from scripts.build_scale_policy_candidates import load_protocol, rows


def media_one(task):
    root, cache_file, video_root, runtime_dir = map(Path,task)
    record=json.loads(cache_file.read_text()); c=record['cache']
    receipt=root/'media_validation'/cache_file.name
    name=c['row'].get('video_file') or str(record['video_id'])+'.mp4'
    source=(video_root/name).resolve()
    if not source.is_relative_to(video_root.resolve()):raise ValueError('unsafe media path')
    expected=record['video_sha256']
    if sha(source) != expected:raise ValueError('original video SHA256 differs: '+name)
    if receipt.exists():
        previous=json.loads(receipt.read_text())
        if previous['video_sha256'] != expected or previous['cache_sha256'] != sha(cache_file):
            raise ValueError('media receipt identity changed')
        return previous
    spec=importlib.util.spec_from_file_location('p2_original_media',runtime_dir/'video.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    tick=time.monotonic(); meta=module.scan(source)
    if (meta['width'],meta['height'],meta['num_frames']) != (c['W'],c['H'],c['num_frames']):
        raise ValueError('original decoded dimensions/frame count changed: '+name)
    if meta['pts'] != c['pts']:raise ValueError('original decoded PTS changed: '+name)
    result={'passed':True,'video_sha256':expected,'cache_sha256':sha(cache_file),
            'decoded_frames':meta['num_frames'],'PTS_exact':True,'GPU_used':False,
            'elapsed_CPU_wall_seconds':time.monotonic()-tick}
    atomic_json(receipt,result)
    return result


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--video-root',type=Path)
    p.add_argument('--workers',type=int,default=8)
    a=p.parse_args(); root=a.root.resolve(); protocol=load_protocol(root)
    manifest=json.loads((root/'candidate_manifest.json').read_text())
    parent=rows(protocol['parent_predictions']); expected_index=rows(protocol['index'])
    source_files=sorted((root/'inputs/p1p2_records').glob('*.json'))
    if len(parent) != len(expected_index) or len(parent) != len(source_files):
        raise ValueError('parent/index/cache coverage mismatch')
    result={}; hashes={}
    for name,policy in POLICIES.items():
        m=manifest['candidates'][name]; pred=Path(m['predictions']); packed=Path(m['zip'])
        # Output locations are root-relative by identity, so remote copies can
        # be audited without changing a frozen protocol or generation manifest.
        pred=root/'candidates'/name/pred.name; packed=root/'candidates'/name/packed.name
        if sha(pred) != m['predictions_sha256'] or sha(packed) != m['zip_sha256']:
            raise ValueError('packaged output SHA256 changed')
        with zipfile.ZipFile(packed) as archive:
            if archive.namelist()!=['predictions.jsonl'] or archive.testzip():raise ValueError('bad ZIP')
            data=archive.read('predictions.jsonl')
        if data!=pred.read_bytes():raise ValueError('ZIP does not contain audited predictions')
        values=[json.loads(s) for s in data.decode('utf-8-sig').splitlines() if s.strip()]
        if len(values)!=len(parent) or len({str(v['video_id']) for v in values})!=len(parent):
            raise ValueError('ZIP index coverage or unique IDs changed')
        stats=Counter(); routes={rt:Counter() for rt in ['a','b','native']}; max_drift=0.;max_qp_drift=0.
        min_scale=1.;max_scale=0.;sum_scale=0.
        for f,baseline,value,index in zip(source_files,parent,values,expected_index):
            record=json.loads(f.read_text()); c=record['cache']; rt=c['route']; ratio=c['ratio']
            if baseline!=record['results']['P2']:raise ValueError('cached P2 is not submitted P2')
            if c['row']!=index or any(value[k]!=v for k,v in index.items()):raise ValueError('source index mismatch')
            if record['parameter_elements']!=8999771458 or value['model_params_b']!=8.999771458:
                raise ValueError('actual inference parameter contract changed')
            if [x['frame'] for x in baseline['predictions']] != [x['frame'] for x in value['predictions']]:
                raise ValueError('temporal membership/order changed')
            receipt=json.loads((root/'completed'/f.name).read_text())
            if receipt['source_sha256']!=sha(f) or receipt['protocol_sha256']!=protocol['protocol_sha256']:
                raise ValueError('source receipt drifted')
            audit_path=root/receipt['metrics'][name]['frame_audit']
            if sha(audit_path)!=receipt['output_hashes'][str(audit_path.relative_to(root))]:raise ValueError('audit changed')
            with gzip.open(audit_path,'rt') as handle: audits=[json.loads(s) for s in handle]
            if len(audits)!=len(value['predictions']):raise ValueError('missing per-frame audit')
            last=-1; routes[rt]['videos']+=1; stats['videos']+=1
            for old,new,audit in zip(baseline['predictions'],value['predictions'],audits):
                frame=new['frame']; b=new['bboxes']; previous=old['bboxes']; rw,rh=ratio
                if type(frame) is not int or not last<frame<c['num_frames']:raise ValueError('invalid original frame')
                last=frame
                if len(b)!=3 or any(type(v) is not int for v in b):raise ValueError('not integer xyw')
                x,y,w=b; h=w*rh/rw
                if not (x>=0 and y>=0 and w>0 and x+w<=c['W'] and y+h<=c['H']+1e-9):
                    raise ValueError('illegal deployed geometry')
                oldcenter=[previous[0]+previous[2]/2,previous[1]+previous[2]*rh/rw/2]
                newcenter=[x+w/2,y+h/2]; shift=max(abs(a-b) for a,b in zip(oldcenter,newcenter))
                max_drift=max(max_drift,shift)
                if shift>1:raise ValueError('fixed P2 Position drifted')
                s=policy[rt]; width=math.floor(previous[2]*s+1e-9)
                expected=[round(oldcenter[0]-width/2),round(oldcenter[1]-width*rh/rw/2),width]
                reason=audit['fallback_reason']
                if reason:
                    if b!=previous:raise ValueError('fallback did not retain original P2')
                    stats['fallback_frames']+=1
                elif b!=expected:raise ValueError('candidate differs from predeclared fixed-center Scale')
                if s==1 and b!=previous:raise ValueError('unmodified route changed')
                if (audit['frame']!=frame or audit['parent_aic_bbox_xyw']!=previous or
                    audit['final_aic_bbox_xyw']!=b or abs(audit['position_drift_px']-shift)>1e-9):
                    raise ValueError('per-frame evidence differs from actual ZIP')
                before=audit['pre_QP_aic_bbox_xyw']; after=audit['post_QP_aic_bbox_xyw']
                qc=max(abs((before[0]+before[2]/2)-(after[0]+after[2]/2)),
                       abs((before[1]+before[2]*rh/rw/2)-(after[1]+after[2]*rh/rw/2)))
                max_qp_drift=max(max_qp_drift,qc)
                stats['QP_width_changes']+=before[2]!=after[2]
                stats['frames']+=1;routes[rt]['frames']+=1
                routes[rt]['changed_frames']+=b!=previous
                actual=w/previous[2];min_scale=min(min_scale,actual);max_scale=max(max_scale,actual);sum_scale+=actual
        if name=='C0' and (sha(pred)!=sha(protocol['parent_predictions']) or sha(packed)!=sha(protocol['parent_package'])):
            raise ValueError('C0 lost original P2 byte identity')
        result[name]={'passed':True,'counts':dict(stats),'by_route':{k:dict(v) for k,v in routes.items()},
                      'max_position_drift_px':max_drift,'max_QP_center_change_px':max_qp_drift,
                      'actual_width_scale':{'min':min_scale,'max':max_scale,'frame_mean':sum_scale/stats['frames']}}
        hashes[name]=sha(packed)
    result['geometry_is_not_official_quality_evaluation']=True
    result['parent']='P2';result['protocol_sha256']=protocol['protocol_sha256'];result['zip_hashes']=hashes
    controls=[root/'stability_controls'/k/'predictions.jsonl' for k in ['C1-frame','C1-segment']]
    if not sha(controls[0])==sha(controls[1])==sha(root/'candidates/C1/predictions.jsonl'):
        raise ValueError('fixed-action stability controls changed output')
    atomic_json(root/'independent_audit.json',result)
    if a.video_root:
        done=[];tick=time.monotonic()
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures=[pool.submit(media_one,(str(root),str(f),str(a.video_root),protocol['runtime_dir'])) for f in source_files]
            for future in as_completed(futures):
                done.append(future.result())
                atomic_json(root/'media_validation_progress.json',{'completed':len(done),'total':len(source_files),
                            'elapsed_CPU_wall_seconds':time.monotonic()-tick,'GPU_used':False})
        atomic_json(root/'media_validation.json',{'passed':all(r['passed'] for r in done),'videos':len(done),
                    'decoded_frames':sum(r['decoded_frames'] for r in done),'PTS_exact':True,
                    'video_SHA256_exact':True,'GPU_used':False,'labels_accessed':False,
                    'all_four_packages_validated':True,'protocol_sha256':protocol['protocol_sha256'],
                    'elapsed_CPU_wall_seconds':time.monotonic()-tick})
    print(json.dumps({'passed':True,'candidates':list(POLICIES),'parent':'P2','GPU_used':False}))


if __name__=='__main__':main()
