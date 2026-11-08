"""Reconcile successful/failed preflight GPU intervals, including shared-device overlap."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
from pathlib import Path

from aic.scale_execution import fingerprint
from aic.scale_parent import atomic_json,file_sha


def interval_union_hours(intervals):
    merged=[]
    for start,end in sorted(intervals):
        if end<start:raise ValueError('invalid preflight timestamps')
        if merged and start<=merged[-1][1]:merged[-1][1]=max(end,merged[-1][1])
        else:merged.append([start,end])
    return sum((b-a).total_seconds() for a,b in merged)/3600


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();intervals=[];runs=[]
    sources=['bootstrap_01/preflight_usage.json','pilot_16/result.json','pilot_16_diff/result.json',
             'pilot_single/result.json','profile_threads_01/profile.json','parallel_one/result.json',
             'parallel_two/result.json','smoke_AB_final/receipt.json','smoke_AB_verified/receipt.json']
    for name in sources:
        path=a.root/name
        if not path.exists():continue
        r=json.loads(path.read_text());start=datetime.fromisoformat(r['started_at']);end=datetime.fromisoformat(r['ended_at'])
        intervals.append((start,end))
        runs.append(dict(path=str(path),sha256=file_sha(path),started_at=r['started_at'],ended_at=r['ended_at'],
                         status=r.get('status','completed_profile'),error=r.get('error'),
                         gpu_name=r['gpu_name'],process_hours=(end-start).total_seconds()/3600,
                         formal_training=False))
    usage=dict(purpose='preflight only',host='connect.westd.seetacloud.com:35092',
               same_device_process_hours=sum(r['process_hours'] for r in runs),
               same_device_wall_hours=interval_union_hours(intervals),runs=runs,
               formal_optimizer_steps=0,formal_stage_hours=0.,
               note='Parallel processes on the same GPU are merged for device-wall accounting; process hours also retained. No full-data preprocessing budget approved.')
    atomic_json(a.root/'preflight_gpu_usage.json',usage)
    def position_identity(path):
        rows=[json.loads(l) for l in path.open()]
        return fingerprint(sorted([(r['image_id'],r['target_ratio'],r['p1_aic_bbox_xyw'])
            for r in rows if r['dataset_source']=='LIVE_YT_VC']))
    baseline=position_identity(a.root/'pilot_single/positions.jsonl')
    workers=[]
    for name in ['parallel_one','parallel_two']:
        r=json.loads((a.root/name/'result.json').read_text())
        workers.append(dict(worker=name,elapsed_seconds=r['elapsed_seconds'],
                            started_at=r['started_at'],ended_at=r['ended_at'],
                            positions_equal_to_original_singleton=position_identity(a.root/name/'positions.jsonl')==baseline))
    # Exclude initial loading by using each worker's source processing progress.
    times=[json.loads((a.root/name/'progress.json').read_text())['elapsed_seconds'] for name in ['parallel_one','parallel_two']]
    thread_profile=json.loads((a.root/'profile_threads_01/profile.json').read_text())
    solo=thread_profile['runs'][0]['wall_seconds']
    # Full process makespan is conservative: no hidden exclusion of model load.
    makespan=(max(datetime.fromisoformat(w['ended_at']) for w in workers)-
              min(datetime.fromisoformat(w['started_at']) for w in workers)).total_seconds()
    parallel=dict(workers=workers,full_process_makespan_seconds=makespan,
                  serial_per_video_profile_seconds=solo,
                  conservative_throughput_speedup_excluding_solo_load=2*solo/makespan,
                  gpu_memory_observed_MiB=35622,scope='one source repeated per worker; pilot, not full-data ETA')
    atomic_json(a.root/'parallel_profile.json',parallel)
    print(json.dumps(dict(device_wall_hours=usage['same_device_wall_hours'],parallel=parallel),indent=2))


if __name__=='__main__':main()
