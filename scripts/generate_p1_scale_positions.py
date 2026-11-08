"""Resumable real frozen-P1 preflight. Never starts SFT/DPO optimization.

LIVE requires original source videos and verifies all annotated decoded frames.
Still images use the explicitly recorded single-frame P1 projection protocol.
Limits identify a pilot; only an unrestricted, fully covered run is complete.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import time

from aic.scale_parent import OriginalP1Projector, atomic_json, file_sha
from aic.scale_execution import fingerprint


def choose_rows(rows, datasets, video_limit=None, still_limit=None):
    videos = defaultdict(list)
    stills = defaultdict(list)
    for row in rows:
        if row['dataset_source'] not in datasets:
            continue
        if row['split'] not in {'train', 'val'}:
            raise ValueError('public Test rows forbidden')
        group = videos if row['dataset_source'] == 'LIVE_YT_VC' else stills
        group[(row['dataset_source'], str(row['source_id']))].append(row)
    # Labels contain train before val. Preserve their source ordering.
    return (list(videos.items())[:video_limit], list(stills.items())[:still_limit])


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--video-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--datasets', nargs='+', default=['LIVE_YT_VC', 'GNMC'])
    p.add_argument('--limit-videos', type=int)
    p.add_argument('--limit-still-sources', type=int)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--max-hours', type=float, default=8.)
    a = p.parse_args()
    def terminate(signum, frame):
        raise TimeoutError('preflight terminated; recoverable per-source caches retained')
    signal.signal(signal.SIGTERM, terminate)
    if not 0 < a.max_hours <= 12:
        raise ValueError('bounded preflight must be <=12 hours')
    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
        os.environ[name] = '1'
    import torch
    torch.set_num_threads(8)
    torch.manual_seed(20261108)
    c = json.loads(a.config.read_text())
    rows = [json.loads(line) for line in a.labels.open() if line.strip()]
    videos, stills = choose_rows(rows, set(a.datasets), a.limit_videos, a.limit_still_sources)
    required = [r for _, group in videos + stills for r in group]
    # Fail before GPU load if any required source media is absent.
    missing = [s for (_, s), _ in videos if not (a.video_root / (s + '.mp4')).is_file()]
    if missing:
        raise ValueError(f'{len(missing)} original source videos missing: {missing[:10]}')
    a.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    receipt = dict(started_at=datetime.now(timezone.utc).isoformat(),
                   purpose='frozen_P1_public_data_preflight', formal_training=False,
                   optimizer_steps=0, gpu_name=torch.cuda.get_device_name(0),
                   labels_sha256=file_sha(a.labels), max_hours=a.max_hours,
                   requested_views=len(required), batch_size=a.batch_size)
    records = []
    try:
        from aic.scale_videoora import create_backend
        from scripts.smoke_scale_videoora import run_smoke
        backend = create_backend(c, 'cuda:0')
        projector = OriginalP1Projector(backend, c, a.batch_size)
        signature = fingerprint({'labels': receipt['labels_sha256'], 'protocol': projector.identity,
                                 'keys': [(k, len(v)) for k, v in videos + stills]})
        signature_path = a.output / 'signature.json'
        if signature_path.exists() and json.loads(signature_path.read_text())['sha256'] != signature:
            raise ValueError('resume inputs/protocol differ')
        atomic_json(signature_path, {'sha256': signature, 'protocol_sha256': projector.identity,
                                    'config_sha256': file_sha(a.config), 'arguments': vars(a)})

        def check_time():
            if time.monotonic() - started > a.max_hours * 3600:
                raise TimeoutError('preflight time allowance exhausted; cached work is recoverable')

        def take_smoke(group):
            if not a.smoke:
                return
            for ratio in ([9, 16], [16, 9]):
                output = a.output / ('smoke_' + '_'.join(map(str, ratio)))
                if (output / 'real_smoke.json').exists():
                    continue
                position = next((r for r in group if r['target_ratio'] == ratio and r['split'] == 'train'), None)
                if position is None:
                    continue
                raw = next(r for r in required if all(r[f] == position[f] for f in
                           ('dataset_source', 'source_id', 'image_id', 'target_ratio')))
                backend._activate(scale=True, ratio=ratio)
                backend.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
                run_smoke(backend, c, dict(raw, **position), output)
                backend.model.gradient_checkpointing_disable()
                print('SMOKE_PASSED', ratio, flush=True)

        for i, ((dataset, source), group) in enumerate(videos):
            check_time()
            cache = a.output / 'videos' / (source + '.json')
            tick = time.monotonic()
            if cache.exists():
                value = json.loads(cache.read_text())
                if value['position_protocol_sha256'] != projector.identity:
                    raise ValueError('video cache protocol differs')
                if value['source_video_sha256'] != file_sha(a.video_root / (source + '.mp4')):
                    raise ValueError('resumed video source hash differs')
                current = value['records']
            else:
                current = projector.video(a.video_root / (source + '.mp4'), group, cache)
            records.extend(current)
            take_smoke(current)
            print(json.dumps({'video_sources_done': i+1, 'video_sources_total': len(videos),
                              'source': source, 'rows': len(records),
                              'source_seconds': time.monotonic()-tick,
                              'elapsed_seconds': time.monotonic()-started}), flush=True)
            atomic_json(a.output / 'progress.json', dict(receipt, status='running', rows=len(records),
                        video_sources_done=i+1, elapsed_seconds=time.monotonic()-started))
        # Keep complete sources together in each resumable batch.
        for offset in range(0, len(stills), max(1, a.batch_size // 2)):
            check_time()
            groups = stills[offset:offset+max(1, a.batch_size // 2)]
            group = [r for _, values in groups for r in values]
            cache = a.output / 'stills' / (fingerprint([k for k, _ in groups]) + '.json')
            if cache.exists():
                value = json.loads(cache.read_text())
                if value['position_protocol_sha256'] != projector.identity:
                    raise ValueError('still cache protocol differs')
                current = value['records']
            else:
                current = projector.still_batch(group)
                atomic_json(cache, {'position_protocol_sha256': projector.identity, 'records': current})
            records.extend(current)
            take_smoke(current)
            print(json.dumps({'still_sources_done': min(len(stills), offset+len(groups)),
                              'still_sources_total': len(stills), 'rows': len(records),
                              'elapsed_seconds': time.monotonic()-started}), flush=True)
            atomic_json(a.output / 'progress.json', dict(receipt, status='running', rows=len(records),
                        still_sources_done=min(len(stills), offset+len(groups)),
                        elapsed_seconds=time.monotonic()-started))
        if len(records) != len(required):
            raise ValueError('position coverage mismatch')
        path = a.output / 'positions.jsonl'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in records))
        temporary.replace(path)
        receipt.update(status='passed', complete=a.limit_videos is None and a.limit_still_sources is None,
                       rows=len(records), positions_sha256=file_sha(path),
                       batch_parity=projector.batch_parity, protocol_sha256=projector.identity,
                       parent_immutable=backend._parent_digest()==backend.parent_state_sha256)
        if not receipt['parent_immutable']:
            raise ValueError('parent adapter changed during preflight')
    except BaseException as exc:
        receipt.update(status='failed', error=type(exc).__name__+': '+str(exc), rows=len(records))
        raise
    finally:
        receipt.update(ended_at=datetime.now(timezone.utc).isoformat(),
                       elapsed_seconds=time.monotonic()-started,
                       gpu_hours=(time.monotonic()-started)/3600)
        atomic_json(a.output / ('usage_' + receipt['started_at'].replace(':', '-') + '.json'), receipt)
        atomic_json(a.output / 'result.json', receipt)


if __name__ == '__main__':
    main()
