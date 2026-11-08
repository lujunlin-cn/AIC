"""CPU semantic replay of original P1 smoothing/QP from retained real anchors."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys

from scripts.run_final_p1_cache import sha, write


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    c = json.loads(a.config.read_text())
    runtime = Path(c['runtime']['runtime_dir'])
    for name, digest in c['runtime']['runtime_source_hashes'].items():
        if sha(runtime/name) != digest:
            raise ValueError('original runtime source changed: ' + name)
    if sha(runtime.parent/'infer_candidates.py') != c['runtime']['spatial_entrypoint_sha256']:
        raise ValueError('original spatial entrypoint changed')
    sys.path.insert(0, str(runtime))
    from adaptive_crop import smooth_crops
    from clock_rule import adjusted
    from contracts import integer_predictions
    from recovery_path_projection import apply_events
    from generation import apply_vertical
    spec = importlib.util.spec_from_file_location('original_P1_replay', runtime.parent/'infer_candidates.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    checks = []
    for path in sorted(a.root.glob('workers/*/videos/*.json')):
        cache = json.loads(path.read_text())
        for detail in cache['details']:
            value = detail['original_runtime_cache']
            W, H, ratio = value['W'], value['H'], value['ratio']
            meta = {'width': W, 'height': H, 'pts': value['pts'],
                    'duration': value['duration'], 'num_frames': value['num_frames']}
            events, anchors, proposed = value['events'], value['anchors'], value['proposed']
            native, _ = smooth_crops(meta['pts'], events, adjusted(anchors, events), W, H, ratio, max_speed=.15)
            native = integer_predictions(native, W, H, ratio)
            if value['route'] == 'native':
                final = native
            else:
                candidate, _ = smooth_crops(meta['pts'], events, adjusted(proposed, events), W, H, ratio, max_speed=.15)
                candidate = integer_predictions(candidate, W, H, ratio)
                function = apply_events if value['route'] == 'a' else apply_vertical
                final, _ = function(meta, events, anchors, native, candidate, qp_floor=.15)
            predictions = module.reconcile(final, value['target_frames'], W, H, ratio)
            lookup = {r['frame']: r['bboxes'] for r in predictions}
            rows = [r for r in cache['records'] if r['target_ratio'] == ratio]
            mismatch = [r['frame_id'] for r in rows if lookup.get(r['frame_id']) != r['p1_aic_bbox_xyw']]
            checks.append({'source_id': value['video_id'], 'target_ratio': ratio, 'sample_frames': len(rows),
                'mismatched_frame_ids': mismatch, 'passed': not mismatch,
                'full_original_output_frames': len(predictions), 'raw_cache_sha256': sha(path)})
    cuda_initialized = bool('torch' in sys.modules and sys.modules['torch'].cuda.is_initialized())
    if cuda_initialized:
        raise RuntimeError('geometry replay unexpectedly initialized CUDA')
    write(a.output, {'status': 'passed' if checks and all(v['passed'] for v in checks) else 'failed',
        'CUDA_initialized': cuda_initialized, 'no_model_load_or_inference': True,
        'original_QP_and_smoothing_source_hashes_verified': True, 'checks': checks,
        'checked_view_frames': sum(v['sample_frames'] for v in checks),
        'scope': 'exact CPU replay of final P1 path from real retained subject/candidate anchors',
        'raw_IoU_used_as_deploy_IoU': False})
    if not checks or any(not v['passed'] for v in checks):
        raise ValueError('final P1 CPU replay mismatch')


if __name__ == '__main__':
    main()
