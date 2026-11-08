"""After the existing bounded GPU run stops, run CPU diagnostics once and stop.

This command cannot start/resume a GPU task or SFT/DPO. It only observes the
existing ledger, recovers complete receipts and invokes CPU-only commands.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.run_final_p1_cache import locked, sha, write


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--image-root', type=Path, required=True)
    p.add_argument('--wait-seconds', type=float, default=4000)
    a = p.parse_args(); started = time.monotonic()
    with locked(a.root/'cpu_finalize.lock', nonblocking=True):
        previous_path = a.root/'postprocess_receipt.json'
        if previous_path.exists():
            previous = json.loads(previous_path.read_text())
            if previous.get('status') == 'passed':
                raise FileExistsError('CPU diagnostics already passed; do not silently overwrite that run')
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            write(a.root/('postprocess_receipt_previous_'+stamp+'.json'), previous)
        while True:
            path = a.root/'gpu_budget_ledger.json'
            ledger = json.loads(path.read_text())
            if ledger.get('runs') and all(r.get('ended_at') is not None for r in ledger['runs']):
                break
            if time.monotonic()-started > a.wait_seconds:
                raise TimeoutError('GPU ledger not closed; no new GPU work or budget reset attempted')
            time.sleep(5)
        receipt = {'started_at': datetime.now(timezone.utc).isoformat(), 'status': 'running',
            'waited_seconds': time.monotonic()-started, 'GPU_task_started': False,
            'formal_training_started': False, 'commands': [], 'GPU_ledger_sha256': sha(path)}
        write(a.root/'postprocess_receipt.json', receipt)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4')
        commands = [
            ['scripts.finalize_scale_p1_cache', '--root', str(a.root), '--recover-only'],
            ['scripts.extract_final_scale_features', '--labels', str(a.root/'inputs/cohort.jsonl'),
             '--records', str(a.root/'records'), '--image-root', str(a.image_root),
             '--output', str(a.root/'visual_features.json')],
            ['scripts.replay_final_p1_geometry', '--config', str(a.config), '--root', str(a.root),
             '--output', str(a.root/'geometry_replay.json')],
            ['scripts.finalize_scale_p1_cache', '--root', str(a.root)],
            ['scripts.analyze_scale_final_preflight', '--root', str(a.root)],
        ]
        for args in commands:
            tick = time.monotonic(); cmd = [sys.executable, '-m', *args]
            result = subprocess.run(cmd, env=env, text=True, capture_output=True)
            receipt['commands'].append({'command': cmd, 'exit_code': result.returncode,
                'stdout': result.stdout, 'stderr': result.stderr, 'elapsed_seconds': time.monotonic()-tick})
            if result.returncode:
                receipt.update(status='failed', ended_at=datetime.now(timezone.utc).isoformat())
                write(a.root/'postprocess_receipt.json', receipt)
                raise RuntimeError('CPU postprocessing failed; evidence preserved: ' + args[0])
            write(a.root/'postprocess_receipt.json', receipt)
        receipt.update(status='passed', ended_at=datetime.now(timezone.utc).isoformat())
        write(a.root/'postprocess_receipt.json', receipt)
        print(json.dumps({'status': 'passed', 'commands': len(commands), 'GPU_task_started': False}), flush=True)


if __name__ == '__main__':
    main()
