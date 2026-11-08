"""One cumulative, hard-bounded frozen-P1 diagnostic run; never trains a model.

Prepare is CPU-only. Run supervises independent singleton P1 workers sharing
one physical GPU and counts device wall time once, including both model loads.
The 100-source throughput cohort is subject to the same 3,600-second cap.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

SEED = 20261108
CAP_SECONDS = 3600.0


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    tmp.replace(path)


@contextmanager
def locked(path, nonblocking=False):
    with Path(path).open('a+') as f:
        flags = fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0)
        fcntl.flock(f, flags)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def read_rows(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def select_frames(rows, count=6):
    """One frame per disjoint temporal stratum, with alternating GT-scale targets.

    Public GT controls only this declared diagnostic sampling. It is never
    passed to the Position model or used to fit actions on validation data.
    """
    by_frame = {r['frame_id']: r for r in rows if not r.get('transposed_geometry_augmentation')}
    ordered = sorted(by_frame)
    if not ordered or count < 1:
        raise ValueError('nonempty source and positive frame count required')
    count = min(count, len(ordered))
    scale = {f: by_frame[f]['gt_crop_xywh'][2] / min(
        by_frame[f]['image_width'], by_frame[f]['image_height'] * 9 / 16) for f in ordered}
    lo, hi = min(scale.values()), max(scale.values())
    targets = (0., .8, .2, 1., .4, .6)
    selected = []
    for j in range(count):
        part = ordered[j * len(ordered) // count:(j + 1) * len(ordered) // count]
        middle = (part[0] + part[-1]) / 2
        target = lo + targets[j % len(targets)] * (hi - lo)
        # Temporal strata guarantee broad coverage; within-stratum tie prefers
        # target scale, then the stratum midpoint, then the original frame ID.
        selected.append(min(part, key=lambda f: (abs(scale[f] - target), abs(f - middle), f)))
    return selected


def schedule_sources(groups):
    ordered = {}
    for split in ('train', 'val'):
        ordered[split] = sorted([s for s, rows in groups.items() if rows[0]['split'] == split],
            key=lambda s: hashlib.sha256(f'final-preflight:{SEED}:{s}'.encode()).hexdigest())
    result = []
    for i in range(max(map(len, ordered.values()))):
        for split in ('train', 'val'):
            if i < len(ordered[split]):
                result.append(ordered[split][i])
    return result


def prepare(a):
    if (a.output / 'p1_cache_manifest.json').exists():
        raise FileExistsError('do not overwrite an existing diagnostic plan')
    labels = read_rows(a.labels)
    groups = defaultdict(list)
    for r in labels:
        if r['dataset_source'] == 'LIVE_YT_VC':
            if r['split'] not in ('train', 'val'):
                raise ValueError('Test data forbidden')
            groups[r['source_id']].append(r)
    schedule = schedule_sources(groups)
    selected, source_plan = [], []
    for source in schedule:
        rows = groups[source]
        ids = select_frames(rows, a.frames)
        chosen = [r for r in rows if r['frame_id'] in ids]
        if len(chosen) != 2 * len(ids):
            raise ValueError('both geometry views required for every sampled frame')
        selected.extend(chosen)
        source_plan.append({'source_id': source, 'split': rows[0]['split'], 'frame_ids': ids,
                            'physical_frames': len(ids), 'label_views': len(chosen)})
    target = a.output / 'inputs' / 'cohort.jsonl'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in selected))
    write(a.output / 'source_schedule.json', source_plan)
    c = json.loads(a.config.read_text())
    smoke = {}
    for name in ('model_load', 'sft_forward_backward', 'dpo_forward_backward', 'geometry_validation'):
        path = a.config.parent.parent / 'smoke_test' / (name + '.json')
        value = json.loads(path.read_text())
        if name != 'geometry_validation' and not (value.get('real_model') and value.get('passed')):
            raise ValueError('real smoke missing: ' + name)
        if name == 'geometry_validation' and not value.get('raw_video_pixel_roundtrip_passed'):
            raise ValueError('real geometry smoke missing')
        smoke[name] = {'path': str(path), 'sha256': sha(path), 'reused': True}
    write(a.output / 'p1_cache_manifest.json', {
        'schema_version': 'aic.final_preflight.v1', 'created_at': datetime.now(timezone.utc).isoformat(),
        'cache_runner_sha256': sha(__file__),
        'seed': SEED, 'source_labels_sha256': sha(a.labels), 'cohort_sha256': sha(target),
        'config_sha256': sha(a.config), 'source_schedule_sha256': sha(a.output / 'source_schedule.json'),
        'p1_base_sha256': c['parent']['weights_sha256'],
        'p1_adapter_manifest_sha256': c['parent']['adapter_sha256'],
        'p1_adapter_sha256_by_route': c['runtime']['parent_adapter_hashes'],
        'tokenizer_sha256': c['parent']['tokenizer_sha256'],
        'parent_identity_sha256': c['parent']['identity_sha256'],
        'postprocessing_config_sha256': c['parent']['inference_config_sha256'],
        'runtime_source_hashes': c['runtime']['runtime_source_hashes'],
        'spatial_entrypoint_sha256': c['runtime']['spatial_entrypoint_sha256'],
        'position_kind': 'P1_final_deployed', 'qp_floor': .15,
        'original_LIVE_sources': {'train': 1600, 'val': 200},
        'clean_LIVE_sources': {s: sum(g[0]['split'] == s for g in groups.values()) for s in ('train', 'val')},
        'quarantined_train_sources': 5, 'requested_frames_per_source': a.frames,
        'planned_physical_frames': len(selected) // 2, 'planned_view_frames': len(selected),
        'sampling': 'six disjoint temporal strata; alternating public-GT-scale targets within strata',
        'source_order': 'label-blind SHA256 order; alternate Train/Val for error comparison',
        'benchmark_requested_sources': 100, 'benchmark_counts_toward_same_budget': True,
        'GPU_seconds_cap': CAP_SECONDS, 'formal_training_authorized': False,
        'B_scope': 'transposed LIVE augmentation; not native B validation',
        'original_temporal_set': 'entire public clip, identical for all policies; no Temporal inference',
        'evaluation_frames': 'sampled annotated original frame IDs; full P1 QP still uses original full clip',
        'scale_insertion': 'after frozen original P1 QP and integer output; fixed final center',
        'smoke_evidence': smoke, 'license_review': 'completed by project owner per user instruction; no repeated review',
        'cache_records': 'positions.jsonl; generated from retained per-source complete caches',
        'p1_validation_training_exposure': None,
    })


def elapsed_budget(ledger):
    if any(r.get('ended_at') is None for r in ledger.get('runs', [])):
        raise RuntimeError('unclosed GPU attempt; reconcile it before any restart')
    return sum(float(r['elapsed_seconds']) for r in ledger.get('runs', []))


def claim(output):
    with locked(output / 'queue.lock'):
        path = output / 'queue.json'
        q = json.loads(path.read_text())
        if q['next'] >= len(q['sources']):
            return None
        source = q['sources'][q['next']]
        q['next'] += 1
        write(path, q)
        return source


def worker(a):
    # The supervisor starts the budget BEFORE this subprocess imports/loads torch.
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    import torch
    from aic.scale_parent import OriginalP1Projector, file_sha
    from aic.scale_videoora import create_backend
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    def stop(signum, frame):
        raise TimeoutError('supervisor stopped diagnostic worker at shared deadline')
    signal.signal(signal.SIGTERM, stop)
    c = json.loads(a.config.read_text())
    manifest = json.loads((a.output / 'p1_cache_manifest.json').read_text())
    if sha(a.config) != manifest['config_sha256'] or sha(a.output / 'inputs/cohort.jsonl') != manifest['cohort_sha256']:
        raise ValueError('prepared inputs changed')
    groups = defaultdict(list)
    for row in read_rows(a.output / 'inputs/cohort.jsonl'):
        groups[row['source_id']].append(row)
    started = time.monotonic()
    backend = create_backend(c, 'cuda:0')
    backend.model.requires_grad_(False)
    backend.model.eval()
    projector = OriginalP1Projector(backend, c, 1)
    write(a.output / f'worker_{a.worker_id}_load.json', {
        'load_seconds': time.monotonic() - started, 'parent_identity_sha256': c['parent']['identity_sha256'],
        'position_protocol_sha256': projector.identity, 'GPU': torch.cuda.get_device_name(0),
        'optimizer_steps': 0, 'formal_training': False})
    failures = 0
    try:
        while (source := claim(a.output)) is not None:
            t = time.monotonic()
            # The original projector writes batch-parity receipts beside its
            # videos directory. Isolate those receipts per process as well.
            path = a.output / 'workers' / str(a.worker_id) / 'videos' / (source + '.json')
            if path.exists():
                cache = json.loads(path.read_text())
                if cache['position_protocol_sha256'] != projector.identity:
                    raise ValueError('resumed original P1 protocol differs')
                if cache['source_video_sha256'] != file_sha(a.video_root / (source + '.mp4')):
                    raise ValueError('resumed source bytes differ')
                current = cache['records']
            else:
                try:
                    current = projector.video(a.video_root / (source + '.mp4'), groups[source], path)
                except Exception as exc:
                    if isinstance(exc, TimeoutError):
                        raise
                    write(a.output / 'failures' / (source + '.json'), {
                        'source_id': source, 'error': type(exc).__name__ + ': ' + str(exc),
                        'seconds': time.monotonic() - t, 'worker': a.worker_id})
                    failures += 1
                    if failures >= 3:
                        raise RuntimeError('three consecutive source failures; stop for investigation') from exc
                    continue
            failures = 0
            required = {(r['image_id'], tuple(r['target_ratio'])) for r in groups[source]}
            current = [r for r in current if (r['image_id'], tuple(r['target_ratio'])) in required]
            if len(current) != len(required):
                raise ValueError('incomplete per-source geometry cache')
            for r in current:
                r.update(p1_base_sha256=c['parent']['weights_sha256'],
                    p1_adapter_manifest_sha256=c['parent']['adapter_sha256'],
                    p1_adapter_sha256=c['runtime']['parent_adapter_hashes'][
                        'portrait' if r['target_ratio'] == [9, 16] else 'vertical'],
                    tokenizer_sha256=c['parent']['tokenizer_sha256'],
                    postprocessing_config_sha256=c['parent']['inference_config_sha256'],
                    p1_validation_source_disjoint_verified=None)
            write(a.output / 'records' / (source + '.json'), current)
            write(a.output / 'timing' / (source + '.json'), {
                'source_id': source, 'split': groups[source][0]['split'],
                'seconds': time.monotonic() - t, 'view_frames': len(current), 'worker': a.worker_id})
            print(json.dumps({'source': source, 'split': groups[source][0]['split'],
                'seconds': time.monotonic() - t, 'view_frames': len(current)}), flush=True)
    finally:
        write(a.output / f'worker_{a.worker_id}_final.json', {
            'parent_immutable': backend._parent_digest() == backend.parent_state_sha256,
            'elapsed_seconds': time.monotonic() - started, 'optimizer_steps': 0})


def supervise(a):
    a.output.mkdir(parents=True, exist_ok=True)
    with locked(a.output / 'gpu_budget.lock', nonblocking=True):
        path = a.output / 'gpu_budget_ledger.json'
        ledger = json.loads(path.read_text()) if path.exists() else {'cap_seconds': CAP_SECONDS, 'runs': []}
        if ledger['cap_seconds'] != CAP_SECONDS:
            raise ValueError('budget cap changed')
        spent = elapsed_budget(ledger)
        remaining = CAP_SECONDS - spent
        if remaining <= 60:
            raise RuntimeError('cumulative GPU budget exhausted; no restart')
        schedule = json.loads((a.output / 'source_schedule.json').read_text())
        # A restart uses only its remaining allowance and retains completed data.
        pending = [r['source_id'] for r in schedule if not (a.output / 'records' / (r['source_id'] + '.json')).exists()]
        write(a.output / 'queue.json', {'sources': pending, 'next': 0})
        start = time.monotonic()
        record = {'started_at': datetime.now(timezone.utc).isoformat(), 'ended_at': None,
                  'elapsed_seconds': None, 'budget_remaining_at_start_seconds': remaining,
                  'includes_model_load_warmup_and_inference': True, 'formal_training': False,
                  'physical_GPU_count': 1, 'workers': a.workers, 'optimizer_steps': 0,
                  'config_sha256': sha(a.config), 'cohort_sha256': sha(a.output / 'inputs/cohort.jsonl')}
        ledger['runs'].append(record)
        write(path, ledger)
        children, logs = [], []
        stopping = False
        def stop(signum, frame):
            nonlocal stopping
            stopping = True
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            for i in range(a.workers):
                log = (a.output / f'worker_{i}.log').open('a')
                logs.append(log)
                command = [sys.executable, '-m', 'scripts.run_final_p1_cache', 'worker',
                    '--config', str(a.config), '--output', str(a.output), '--video-root', str(a.video_root),
                    '--worker-id', str(i)]
                children.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True, env=dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')))
            while any(p.poll() is None for p in children):
                elapsed = time.monotonic() - start
                done = len(list((a.output / 'records').glob('*.json')))
                write(a.output / 'progress.json', {'elapsed_seconds': elapsed,
                    'cumulative_GPU_seconds': spent + elapsed, 'source_records_completed': done,
                    'source_records_requested': len(schedule), 'benchmark_requested_sources': 100,
                    'benchmark_100_complete': done >= 100, 'formal_training': False})
                if stopping or elapsed >= remaining - 40:
                    record['stop_reason'] = 'shared_hard_budget' if not stopping else 'signal'
                    break
                if any(p.poll() not in (None, 0) for p in children):
                    record['stop_reason'] = 'worker_failure_preserve_completed_sources'
                    break
                time.sleep(2)
        finally:
            for p in children:
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGTERM)
            deadline = time.monotonic() + 5
            while any(p.poll() is None for p in children) and time.monotonic() < deadline:
                time.sleep(.1)
            for p in children:
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGKILL)
                p.wait(timeout=5)
            for log in logs:
                log.close()
            record.update(ended_at=datetime.now(timezone.utc).isoformat(),
                elapsed_seconds=time.monotonic() - start,
                worker_exit_codes=[p.returncode for p in children])
            ledger['cumulative_GPU_seconds'] = spent + record['elapsed_seconds']
            ledger['within_budget'] = ledger['cumulative_GPU_seconds'] <= CAP_SECONDS
            write(path, ledger)
            records = []
            for p in sorted((a.output / 'records').glob('*.json')):
                records.extend(json.loads(p.read_text()))
            (a.output / 'positions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
            print(json.dumps({'cumulative_GPU_seconds': ledger['cumulative_GPU_seconds'],
                'within_budget': ledger['within_budget'], 'view_frames': len(records),
                'complete_sources': len({r['source_id'] for r in records})}), flush=True)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('mode', choices=['prepare', 'run', 'worker'])
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--labels', type=Path)
    p.add_argument('--video-root', type=Path)
    p.add_argument('--frames', type=int, default=6)
    p.add_argument('--workers', type=int, choices=[1, 2], default=2)
    p.add_argument('--worker-id', type=int, default=0)
    a = p.parse_args()
    if a.mode == 'prepare':
        prepare(a)
    elif a.mode == 'run':
        supervise(a)
    else:
        worker(a)


if __name__ == '__main__':
    main()
