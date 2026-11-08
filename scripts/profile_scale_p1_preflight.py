"""Measure exact frozen P1 preprocessing vs generation; zero optimizer updates."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import time

from aic.scale_execution import fingerprint
from aic.scale_parent import OriginalP1Projector, atomic_json, file_sha


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--video-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',default='video2')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    from aic.scale_videoora import create_backend
    c=json.loads(a.config.read_text())
    rows=[json.loads(line) for line in a.labels.open() if line.strip()]
    selected=[r for r in rows if r['dataset_source']=='LIVE_YT_VC' and r['source_id']==a.source]
    if len(selected)!=60 or any(r['split']!='train' for r in selected):
        raise ValueError('one complete public Train source required')
    started=time.monotonic();started_at=datetime.now(timezone.utc).isoformat()
    receipt=dict(purpose='CPU_thread_and_P1_stage_profile',optimizer_steps=0,formal_training=False,
                 started_at=started_at,gpu_name=torch.cuda.get_device_name(0),
                 source=a.source,source_sha256=file_sha(a.video_root/(a.source+'.mp4')),
                 labels_sha256=file_sha(a.labels),runs=[])
    torch.manual_seed(20261108);torch.set_num_threads(8)
    backend=create_backend(c,'cuda:0')
    from frame_bank import FrameBank
    # Exclude one-time CUDA kernel warming from the timed comparison.
    bank=FrameBank(a.video_root/(a.source+'.mp4'),side=640,rate=4)
    projector=OriginalP1Projector(backend,c,1)
    projector.generate([bank.frames[0]],projector.subject_prompt)
    original_generate=backend.model.generate
    measurements=[]
    def measured_generate(*args,**kwargs):
        torch.cuda.synchronize();wall=time.monotonic();cpu=time.process_time()
        result=original_generate(*args,**kwargs)
        torch.cuda.synchronize()
        measurements.append(dict(seconds=time.monotonic()-wall,cpu_seconds=time.process_time()-cpu,
                                 input_tokens=int(kwargs['input_ids'].shape[1]),
                                 output_tokens=int(result.shape[1]-kwargs['input_ids'].shape[1])))
        return result
    backend.model.generate=measured_generate
    try:
        for threads in [8,20]:
            torch.set_num_threads(threads)
            measurements.clear();wall=time.monotonic();cpu=time.process_time()
            projector=OriginalP1Projector(backend,c,1)
            records=projector.video(a.video_root/(a.source+'.mp4'),selected,
                                    a.output/f'threads_{threads}'/'videos'/(a.source+'.json'))
            elapsed=time.monotonic()-wall;cpu_elapsed=time.process_time()-cpu
            run=dict(torch_threads=threads,decoder_threads_per_video=2,wall_seconds=elapsed,
                     generation_calls=len(measurements),generation_seconds=sum(r['seconds'] for r in measurements),
                     non_generation_seconds=elapsed-sum(r['seconds'] for r in measurements),
                     cpu_seconds=cpu_elapsed,average_CPU_cores=cpu_elapsed/elapsed,
                     generated_tokens=sum(r['output_tokens'] for r in measurements),
                     positions_sha256=fingerprint([(r['image_id'],r['target_ratio'],r['p1_aic_bbox_xyw']) for r in records]),
                     generation_measurements=list(measurements))
            receipt['runs'].append(run)
            atomic_json(a.output/'profile.json',receipt)
            print(json.dumps({k:v for k,v in run.items() if k!='generation_measurements'}),flush=True)
    finally:
        backend.model.generate=original_generate
        receipt.update(ended_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started,
                       gpu_hours=(time.monotonic()-started)/3600,
                       parent_immutable=backend._parent_digest()==backend.parent_state_sha256)
        if len(receipt['runs'])==2:
            one,two=receipt['runs']
            receipt.update(thread20_speedup=one['wall_seconds']/two['wall_seconds'],
                           final_P1_boxes_identical=one['positions_sha256']==two['positions_sha256'])
        atomic_json(a.output/'profile.json',receipt)


if __name__=='__main__':main()
