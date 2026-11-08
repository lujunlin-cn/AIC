"""CPU-only, route-specific Scale policies on immutable final P2 crops.

No model loader, training objective, test label or quality estimator belongs here.
The original active-axis QP is reused with an explicitly registered equality
constraint: its position must equal the rounded, fixed-parent-center window.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path

import numpy as np

POLICIES = {
    'C0': {'a': 1., 'b': 1., 'native': 1.},
    'C1': {'a': .90, 'b': .95, 'native': 1.},
    'C2': {'a': .93, 'b': .97, 'native': 1.},
    'C3': {'a': .90, 'b': 1., 'native': 1.},
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n')
    tmp.replace(path)


def center(box, ratio):
    x, y, w = box
    return [x+w/2, y+w*ratio[1]/ratio[0]/2]


def drift(a, b):
    return max(abs(x-y) for x, y in zip(a, b))


def route(width, height, ratio):
    if ratio[0]*16 == ratio[1]*9 and width > height:
        return 'a'
    if ratio[0]*9 == ratio[1]*16 and height-width*9/16 > 2:
        return 'b'
    return 'native'


def legal(box, width, height, ratio):
    if len(box) != 3 or any(type(x) is not int for x in box):
        return False
    x, y, w = box
    return (x >= 0 and y >= 0 and w > 0 and x+w <= width
            and y+w*ratio[1]/ratio[0] <= height+1e-9)


def scale_box(parent, scale, width, height, ratio):
    """Shrink the *actual parent width*, with no clamp or center rescue."""
    if not legal(parent, width, height, ratio):
        raise ValueError('invalid original P2 crop')
    if not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError('illegal Scale action')
    if scale == 1:
        return list(parent), None
    w = math.floor(parent[2]*scale+1e-9)
    if w < 1:
        return list(parent), 'nonpositive_integer_width'
    cx, cy = center(parent, ratio)
    out = [round(cx-w/2), round(cy-w*ratio[1]/ratio[0]/2), w]
    if not legal(out, width, height, ratio):
        return list(parent), 'fixed_center_window_out_of_bounds'
    if drift(center(parent, ratio), center(out, ratio)) > 1:
        return list(parent), 'position_shift_invalid'
    return out, None


def shot_slices(cache, predictions, adjusted):
    """Use original events and original adjusted scene-cut anchors, never bridge cuts."""
    ids = np.asarray([p['frame'] for p in predictions], dtype=int)
    pts = np.asarray(cache['pts'])
    nodes = adjusted(cache['anchors'], cache['events']) if cache['anchors'] else []
    shots = []
    for event_id, (start, end) in enumerate(cache['events']):
        positions = np.flatnonzero((pts[ids] >= start) & (pts[ids] < end))
        if not len(positions):
            continue
        times = pts[ids[positions]]
        cuts = sorted({int(np.searchsorted(times, a['time'])) for a in nodes
                       if a.get('scene_cut') and start <= a['time'] < end
                       and times[0] < a['time'] <= times[-1]})
        for left, right in zip([0]+cuts, cuts+[len(positions)]):
            stop = float(times[right]) if right < len(positions) else end
            begin = start if left == 0 else float(times[left])
            anchors = [a for a in nodes if begin <= a['time'] < stop]
            shots.append({'event_id': event_id, 'positions': positions[left:right].tolist(),
                          'begin': begin, 'end': stop, 'anchor_count': len(anchors)})
    covered = [i for s in shots for i in s['positions']]
    if sorted(covered) != list(range(len(predictions))) or len(set(covered)) != len(covered):
        raise ValueError('event/shot coverage differs from frozen temporal set')
    return shots


def scale_schedule(shots, size, requested, mode):
    """Actual anchor interpolation / within-shot median; constant policies coincide."""
    if mode not in ('frame', 'segment'):
        raise ValueError('unknown stability control')
    out = np.full(size, np.nan)
    for s in shots:
        # This experiment has one fixed action per route. Every anchor in the
        # shot gets that action; no GT, saliency, scene or score chooses it.
        anchors = np.full(max(1, s['anchor_count']), requested)
        value = float(np.median(anchors)) if mode == 'segment' else float(anchors[0])
        out[s['positions']] = value
    if not np.isfinite(out).all():
        raise ValueError('Scale schedule has missing frames')
    return out


@contextmanager
def frozen_qp_limits(module):
    """Retain original overlap bounds, adding an exact fixed-position equality.

    The original solver, objective, speed and cut logic still run. No parent
    file is modified. The singleton legal interval cannot translate a window.
    """
    original = module.limits
    def fixed(base, x1, x2, width, image_width):
        lo, hi, overlap = original(base, x1, x2, width, image_width)
        base = np.asarray(base, dtype=float)
        if np.any(base < lo-1e-8) or np.any(base > hi+1e-8):
            raise ValueError('fixed P2 center violates original QP overlap bounds')
        return base.copy(), base.copy(), overlap
    module.limits = fixed
    try:
        yield
    finally:
        module.limits = original


def budget_ledger(path):
    """Persistent shared 10h ledger; rerunning CPU jobs cannot reset GPU use."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            value = json.loads(path.read_text())
            if value['cap_GPU_seconds'] != 36000:
                raise ValueError('GPU budget cap changed')
            if any(r.get('ended_at') is None for r in value['runs']):
                raise RuntimeError('unclosed GPU run; do not reset or restart budget')
            durations = [r['elapsed_GPU_seconds'] for r in value['runs']]
            if any(not math.isfinite(s) or s < 0 for s in durations):
                raise RuntimeError('invalid GPU duration')
            spent = sum(durations)
            if spent != value['cumulative_GPU_seconds'] or spent > 36000:
                raise RuntimeError('GPU budget inconsistent or exceeded')
            return value
        value = {'schema': 'aic.scale_policy.GPU_budget.v1',
                 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                 'cap_GPU_seconds': 36000, 'cumulative_GPU_seconds': 0.,
                 'runs': [], 'GPU_tasks_started': False,
                 'formal_training_allowed': False,
                 'scope': 'this Route-specific P2 policy task; prior completed diagnostic budget separate',
                 'reason_no_GPU_use': 'reuse verified P2 inference; geometry/QP/packages/audits are CPU only'}
        atomic_json(path, value)
        return value
