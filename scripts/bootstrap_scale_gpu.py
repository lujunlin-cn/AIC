"""Real P1 deployment plus zero-update GPU smoke on one public training source."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--video-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    torch.set_num_threads(8);torch.manual_seed(20261108)
    c=json.loads(a.config.read_text())
    label=next(json.loads(line) for line in a.labels.open() if json.loads(line)['dataset_source']=='LIVE_YT_VC'
        and json.loads(line)['split']=='train' and json.loads(line)['target_ratio']==[9,16]
        and (a.video_root/(json.loads(line)['source_id']+'.mp4')).is_file())
    video=a.video_root/(label['source_id']+'.mp4')
    digest=hashlib.sha256(video.read_bytes()).hexdigest()
    if digest!=label['source_video_sha256']:raise ValueError('original LIVE source hash mismatch')
    started=time.monotonic();started_at=datetime.now(timezone.utc).isoformat()
    receipt={'started_at':started_at,'purpose':'real_model_preflight_no_optimizer_steps','formal_training':False,
             'gpu_name':torch.cuda.get_device_name(0),'source_id':label['source_id']}
    try:
        from aic.scale_videoora import create_backend
        from aic.scale_geometry import deployed_center
        from scripts.smoke_scale_videoora import run_smoke
        backend=create_backend(c,'cuda:0')
        print('REAL_MODEL_LOADED',backend.total_inference_elements,flush=True)
        source=Path(c['runtime']['runtime_dir']).parent/'infer_candidates.py'
        spec=importlib.util.spec_from_file_location('verified_original_p1_inference',source)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        from video import scan
        meta=scan(video)
        backend.model.gradient_checkpointing_disable();backend._activate(scale=False,ratio=[9,16])
        with torch.inference_mode():
            outputs,cache,audit=module.spatial(backend.model,backend.processor,str(video),
                {'video_id':label['source_id'],'targetRatioWH':[9,16]},meta,[[0.,meta['duration']]],0)
        crop=next(r['bboxes'] for r in outputs['P1']['predictions'] if r['frame']==label['frame_id'])
        row=dict(label,center_xy=list(deployed_center(crop,[9,16])),position_source='P1_frozen',
                 p1_aic_bbox_xyw=crop,position_kind='P1_final_deployed',
                 parent_identity_sha256=c['parent']['identity_sha256'])
        (a.output/'sample.json').write_text(json.dumps(row,indent=2)+'\n')
        (a.output/'p1_deployment.json').write_text(json.dumps({'predictions':outputs['P1'],'cache':cache,'audit':audit,
            'source_sha256':digest,'conditioning':'entire public clip; no temporal model/GT-based event selection',
            'original_inference_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()},default=str)+'\n')
        backend._activate(scale=True,ratio=[9,16])
        backend.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        report=run_smoke(backend,c,row,a.output/'smoke')
        receipt['status']='passed';receipt['smoke']=report
        print('REAL_SMOKE_PASSED',json.dumps(report),flush=True)
    except Exception as exc:
        receipt.update(status='failed',error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        receipt.update(ended_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started,
                       gpu_hours=(time.monotonic()-started)/3600)
        (a.output/'preflight_usage.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
