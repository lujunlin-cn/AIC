"""Build four P2-only Scale candidates from cached inference, entirely on CPU.

Phase receipts preserve resumability. Parent results, model files and temporal
predictions are never overwritten. This command has no evaluation/upload code.
"""
from __future__ import annotations

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
import bisect
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import importlib
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

import numpy as np

from aic.scale_policy import (POLICIES, atomic_json, budget_ledger, canonical_sha, center, drift,
                              frozen_qp_limits, legal, route, scale_box,
                              scale_schedule, sha, shot_slices)


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def load_protocol(root):
    """Fail closed on input/code drift, including resumptions and remote copies."""
    root = Path(root).resolve()
    raw = json.loads((root/'protocol.json').read_text())
    digest = raw.pop('protocol_sha256')
    if canonical_sha(raw) != digest:
        raise ValueError('protocol canonical SHA256 mismatch')
    if raw['parent'] != 'P2' or raw['policies'] != POLICIES:
        raise ValueError('only frozen P2 policies allowed')
    repo = Path(__file__).resolve().parents[1]
    for name, expected in raw['code_hashes'].items():
        if sha(repo/name) != expected:
            raise ValueError('policy code SHA256 changed: '+name)
    for name, expected in raw['input_hashes'].items():
        if sha(root/name) != expected:
            raise ValueError('frozen input SHA256 changed: '+name)
    sources = json.loads((root/'inputs/cache_manifest.json').read_text())
    for name, expected in sources['files'].items():
        if sha(root/'inputs/p1p2_records'/name) != expected:
            raise ValueError('P2 cache input SHA256 changed: '+name)
    if {p.name for p in (root/'inputs/p1p2_records').glob('*.json')} != set(sources['files']):
        raise ValueError('P2 cache coverage changed')
    if len(sources['files']) != len(rows(root/raw['index'])):
        raise ValueError('P2 cache/index count differs')
    raw['protocol_sha256'] = digest
    for key in ['runtime_dir', 'index', 'parent_predictions', 'parent_package']:
        raw[key] = str(root/raw[key])
    return raw


def runtime(path, hashes):
    path = Path(path).resolve()
    for name, digest in hashes.items():
        if sha(path/name) != digest:
            raise ValueError('original CPU runtime hash changed: '+name)
    sys.path.insert(0, str(path))
    mods = {n: importlib.import_module(n) for n in
            ['contracts', 'adaptive_crop', 'clock_rule', 'recovery_path_projection', 'generation']}
    for name, module in mods.items():
        if Path(module.__file__).resolve() != path/(name+'.py'):
            raise ValueError('wrong original runtime imported: '+name)
    return mods


def reconcile(predictions, frames, width, height, ratio, contracts):
    boxes = {p['frame']: p['bboxes'] for p in predictions}; keys = sorted(boxes)
    default = contracts.integer_predictions(
        [{'frame': 0, 'bboxes': contracts.crop_box(width, height, ratio)}], width, height, ratio)[0]['bboxes']
    out = []
    for frame in frames:
        value = boxes.get(frame)
        if value is None and keys:
            i = bisect.bisect_left(keys, frame)
            nearby = keys[max(0, i-1):min(len(keys), i+1)]
            value = boxes[min(nearby, key=lambda k: (abs(k-frame), k))]
        out.append({'frame': frame, 'bboxes': list(value or default)})
    return out


def replay_p2(cache, mods):
    """Original CPU postprocessing only, no P1 replay and no model inference."""
    W, H, ratio = cache['W'], cache['H'], cache['ratio']
    contracts = mods['contracts']; adjusted = mods['clock_rule'].adjusted
    if cache['branch'] == 'subject':
        native, _ = mods['adaptive_crop'].smooth_crops(np.asarray(cache['pts']), cache['events'],
            adjusted(cache['anchors'], cache['events']), W, H, ratio, max_speed=.15)
        native = contracts.integer_predictions(native, W, H, ratio)
        result = native
        if cache['route'] != 'native':
            proposal, _ = mods['adaptive_crop'].smooth_crops(np.asarray(cache['pts']), cache['events'],
                adjusted(cache['proposed'], cache['events']), W, H, ratio, max_speed=.15)
            proposal = contracts.integer_predictions(proposal, W, H, ratio)
            function = mods['recovery_path_projection'].apply_events if cache['route'] == 'a' else mods['generation'].apply_vertical
            result, _ = function({'pts': cache['pts'], 'width': W, 'height': H}, cache['events'],
                                 cache['anchors'], native, proposal, qp_floor=.6)
    else:
        result = contracts.integer_predictions(contracts.frame_predictions(cache['pts'], cache['events'],
                    contracts.crop_box(W, H, ratio)), W, H, ratio)
    return result


def transform(cache, full_parent, requested, mods):
    W, H, ratio = cache['W'], cache['H'], cache['ratio']
    shots = shot_slices(cache, full_parent, mods['clock_rule'].adjusted)
    frame_scales = scale_schedule(shots, len(full_parent), requested, 'frame')
    segment_scales = scale_schedule(shots, len(full_parent), requested, 'segment')
    if not np.array_equal(frame_scales, segment_scales):
        raise ValueError('constant action changed under within-shot median')
    scaled = []; reasons = {}; invalid_triggers = 0
    for p, value in zip(full_parent, frame_scales):
        b, reason = scale_box(p['bboxes'], float(value), W, H, ratio)
        scaled.append({'frame': p['frame'], 'bboxes': b})
        if reason:
            reasons[p['frame']] = reason; invalid_triggers += 1
    # Preserve the original QP's constant-width-per-shot contract if any frame
    # cannot be shrunk: fall back atomically within that existing shot only.
    for shot in shots:
        if any(full_parent[i]['frame'] in reasons for i in shot['positions']):
            for i in shot['positions']:
                frame = full_parent[i]['frame']
                scaled[i] = {'frame': frame, 'bboxes': list(full_parent[i]['bboxes'])}
                reasons.setdefault(frame, 'same_shot_fallback_for_infeasible_scale')
    qp_audits = []; out = scaled
    if requested != 1 and cache['route'] != 'native' and scaled:
        function = mods['recovery_path_projection'].apply_events if cache['route'] == 'a' else mods['generation'].apply_vertical
        try:
            with frozen_qp_limits(mods['recovery_path_projection']):
                out, qp_audits = function({'pts': cache['pts'], 'width': W, 'height': H},
                    cache['events'], cache['anchors'], scaled, scaled, qp_floor=.6)
        except (AssertionError, ValueError) as exc:
            # Preserve input/failed-attempt evidence in the receipt. Never use
            # an unconstrained or nonconvergent QP result as a final prediction.
            reasons = {p['frame']: 'QP_failed_fallback_to_original_P2: '+str(exc) for p in full_parent}
            out = [{'frame': p['frame'], 'bboxes': list(p['bboxes'])} for p in full_parent]
            qp_audits = [{'solved': False, 'fallback': True, 'error': str(exc)}]
    for before, after, parent in zip(scaled, out, full_parent):
        if before['frame'] != after['frame'] or not legal(after['bboxes'], W, H, ratio):
            raise ValueError('QP changed temporal identity or produced illegal geometry')
        if drift(center(after['bboxes'], ratio), center(parent['bboxes'], ratio)) > 1:
            after['bboxes'] = list(parent['bboxes'])
            reasons[after['frame']] = 'QP_position_shift_invalid_fallback_to_original_P2'
    return scaled, out, reasons, qp_audits, shots, invalid_triggers


def process_one(task):
    root, protocol, source = Path(task[0]), task[1], Path(task[2])
    receipt_path = root/'completed'/(source.stem+'.json')
    source_hash = sha(source)
    if receipt_path.exists():
        old = json.loads(receipt_path.read_text())
        if old['source_sha256'] != source_hash or old['protocol_sha256'] != protocol['protocol_sha256']:
            raise ValueError('resume receipt/input mismatch')
        for name, digest in old['output_hashes'].items():
            if sha(root/name) != digest:
                raise ValueError('resume output changed: '+name)
        return old
    tick = time.monotonic(); record = json.loads(source.read_text()); cache = record['cache']
    if cache['route'] != route(cache['W'], cache['H'], cache['ratio']):
        raise ValueError('cached route differs from original route logic')
    mods = runtime(protocol['runtime_dir'], protocol['runtime_hashes'])
    full_parent = replay_p2(cache, mods)
    contract = mods['contracts']; parent = record['results']['P2']
    original = reconcile(full_parent, cache['target_frames'], cache['W'], cache['H'], cache['ratio'], contract)
    if original != parent['predictions']:
        raise ValueError('original P2 QP replay does not reproduce frozen P2: '+str(record['video_id']))
    folder = root/'work'/source.stem
    folder.mkdir(parents=True, exist_ok=True)
    attempt = folder/('attempt_'+str(len(list(folder.glob('attempt_*')))+1).zfill(4))
    attempt.mkdir()
    outputs = {}; metrics = {}; output_hashes = {}
    parent_lookup = {p['frame']: p['bboxes'] for p in original}
    for name, scales in POLICIES.items():
        requested = scales[cache['route']]
        pre, post, reasons, audits, shots, invalid = transform(cache, full_parent, requested, mods)
        pre_lookup = {p['frame']: p['bboxes'] for p in reconcile(pre, cache['target_frames'], cache['W'], cache['H'], cache['ratio'], contract)}
        post_lookup = {p['frame']: p['bboxes'] for p in reconcile(post, cache['target_frames'], cache['W'], cache['H'], cache['ratio'], contract)}
        final = reconcile(post, cache['target_frames'], cache['W'], cache['H'], cache['ratio'], contract)
        result = dict(parent, predictions=final)
        contract.validate_record(result, cache['row'], cache['num_frames'], cache['W'], cache['H'], True)
        if [p['frame'] for p in final] != [p['frame'] for p in original]:
            raise ValueError('frozen P2 temporal membership changed')
        audit_path = attempt/(name+'_frames.jsonl.gz')
        maximum = 0.; changed = 0; fallback_count = 0; qp_width_changed = 0
        with gzip.open(audit_path, 'wt', encoding='utf-8', compresslevel=3) as handle:
            for value in final:
                frame = value['frame']; baseline = parent_lookup[frame]
                before = pre_lookup.get(frame, value['bboxes']); after = post_lookup.get(frame, value['bboxes'])
                centers = [center(b, cache['ratio']) for b in [baseline, before, after, value['bboxes']]]
                error = drift(centers[0], centers[-1]); maximum = max(maximum, error)
                changed += baseline != value['bboxes']; fallback_count += frame in reasons
                qp_width_changed += before[2] != after[2]
                item = {'video_id': record['video_id'], 'frame': frame, 'route': cache['route'],
                    'parent': 'P2', 'candidate': name, 'target_ratio': cache['ratio'],
                    'parent_aic_bbox_xyw': baseline, 'pre_QP_aic_bbox_xyw': before,
                    'post_QP_aic_bbox_xyw': after, 'final_aic_bbox_xyw': value['bboxes'],
                    'parent_center_xy': centers[0], 'post_scale_center_xy': centers[1],
                    'post_QP_center_xy': centers[2], 'final_center_xy': centers[3],
                    'requested_scale': requested, 'parent_width': baseline[2],
                    'pre_QP_width': before[2], 'post_QP_width': after[2], 'final_width': value['bboxes'][2],
                    'scale_before_QP': before[2]/baseline[2], 'scale_after_QP': after[2]/baseline[2],
                    'final_actual_scale': value['bboxes'][2]/baseline[2],
                    'final_ratio': value['bboxes'][2]/(value['bboxes'][2]*cache['ratio'][1]/cache['ratio'][0]),
                    'position_drift_px': error, 'fallback_reason': reasons.get(frame),
                    'QP_center_change_px': drift(centers[1], centers[2])}
                handle.write(json.dumps(item, separators=(',', ':'), allow_nan=False)+'\n')
        output_path = attempt/(name+'.json'); atomic_json(output_path, result)
        outputs[name] = str(output_path.relative_to(root))
        for path in [output_path, audit_path]:
            output_hashes[str(path.relative_to(root))] = sha(path)
        metrics[name] = {'changed_frames': changed, 'max_position_drift_px': maximum,
            'invalid_window_triggers': invalid, 'fallback_output_frames': fallback_count,
            'QP_width_changed_frames': qp_width_changed, 'QP_audits': audits,
            'frame_audit': str(audit_path.relative_to(root)), 'shots': len(shots),
            'C1_frame_segment_schedule_equal': True if name == 'C1' else None}
    receipt = {'source_sha256': source_hash, 'protocol_sha256': protocol['protocol_sha256'],
        'ordinal': record['ordinal'], 'video_id': record['video_id'], 'route': cache['route'],
        'branch': cache['branch'], 'output_frames': len(original), 'parent': 'P2',
        'P2_CPU_replay_exact': True, 'source_video_sha256': record['video_sha256'],
        'outputs': outputs, 'metrics': metrics, 'output_hashes': output_hashes,
        'elapsed_CPU_wall_seconds': time.monotonic()-tick, 'GPU_used': False}
    atomic_json(receipt_path, receipt)
    return receipt


def package(root, name, data, protocol):
    folder = root/'candidates'/name; folder.mkdir(parents=True, exist_ok=True)
    pred = folder/'predictions.jsonl'
    expected_bytes = (Path(protocol['parent_predictions']).read_bytes() if name == 'C0' else
                      ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in data).encode())
    if pred.exists() and pred.read_bytes() != expected_bytes:
        raise ValueError('existing candidate conflicts with frozen inputs: '+name)
    if rows(protocol['parent_predictions']) != data and name == 'C0':
        raise ValueError('C0 source receipts differ from original P2')
    if not pred.exists():
        tmp = folder/'predictions.jsonl.tmp'; tmp.write_bytes(expected_bytes); tmp.replace(pred)
    packed = folder/(name+'_P2_Scale.zip')
    if not packed.exists():
        tmp = folder/(packed.name+'.tmp')
        if name == 'C0':
            shutil.copyfile(protocol['parent_package'], tmp)
        else:
            with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as archive:
                info = zipfile.ZipInfo('predictions.jsonl', date_time=(2026, 11, 8, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, pred.read_bytes(), compresslevel=9)
        tmp.replace(packed)
    if name == 'C0' and sha(packed) != sha(protocol['parent_package']):
        raise ValueError('C0 package lost byte identity')
    with zipfile.ZipFile(packed) as archive:
        if archive.namelist() != ['predictions.jsonl'] or archive.read('predictions.jsonl') != pred.read_bytes() or archive.testzip():
            raise ValueError('submission ZIP roundtrip failed')
    item = {'candidate': name, 'parent': 'P2', 'scales': POLICIES[name],
        'predictions': str(pred), 'predictions_sha256': sha(pred), 'zip': str(packed),
        'zip_sha256': sha(packed), 'zip_bytes': packed.stat().st_size,
        'videos': len(data), 'output_frames': sum(len(r['predictions']) for r in data),
        'model_params_b': 8.999771458, 'official_f1': None, 'promoted': False,
        'status': 'PACKAGED_AWAITING_INDEPENDENT_AUDIT_AND_OFFICIAL_EVALUATION'}
    atomic_json(folder/'manifest.json', item)
    (folder/'SHA256SUMS').write_text(f"{sha(pred)}  predictions.jsonl\n{sha(packed)}  {packed.name}\n")
    return item


def finalize(root, receipts, protocol):
    index = rows(protocol['index']); parent = rows(protocol['parent_predictions'])
    receipts.sort(key=lambda r: r['ordinal'])
    if [r['ordinal'] for r in receipts] != list(range(len(index))):
        raise ValueError('missing/repeated source receipt')
    route_counts = Counter(r['route'] for r in receipts)
    route_frames = Counter()
    for r in receipts: route_frames[r['route']] += r['output_frames']
    manifests = {}; geometry = {}; temporal = {}; qp = {}; changed = {}
    for name in POLICIES:
        data = [json.loads((root/r['outputs'][name]).read_text()) for r in receipts]
        for expected, baseline, value in zip(index, parent, data):
            if any(value[k] != v for k, v in expected.items()):
                raise ValueError('index identity changed')
            if [p['frame'] for p in baseline['predictions']] != [p['frame'] for p in value['predictions']]:
                raise ValueError('temporal frame set changed')
        manifests[name] = package(root, name, data, protocol)
        temporal[name] = {'videos': len(data), 'frame_order_and_membership_identical_videos': len(data),
            'identity_rate': 1., 'output_frames': sum(r['output_frames'] for r in receipts),
            'frame_additions': 0, 'frame_deletions': 0, 'parent': 'P2'}
        geometry[name] = {'maximum_position_drift_px': max(r['metrics'][name]['max_position_drift_px'] for r in receipts),
            'invalid_window_triggers': sum(r['metrics'][name]['invalid_window_triggers'] for r in receipts),
            'fallback_output_frames': sum(r['metrics'][name]['fallback_output_frames'] for r in receipts),
            'three_integer_xyw_and_exact_ratio_height_valid': True,
            'all_original_P2_CPU_replays_exact': all(r['P2_CPU_replay_exact'] for r in receipts)}
        audits = [a for r in receipts for a in r['metrics'][name]['QP_audits']]
        qp[name] = {'QP_floor': .6, 'QP_width_changed_frames': sum(r['metrics'][name]['QP_width_changed_frames'] for r in receipts),
            'QP_shot_audits': len(audits), 'QP_failed_audits': sum(not a['solved'] for a in audits),
            'mode': 'original_active_axis_QP_with_registered_exact_position_equality' if name != 'C0' else 'original_P2_replay_and_byte_exact_reuse',
            'maximum_window_integer_exporter_called_after_scale': False,
            'per_frame_width_center_audits': 'work/*/attempt_*/'+name+'_frames.jsonl.gz'}
        changed[name] = {route: {'changed_output_frames': sum(r['metrics'][name]['changed_frames'] for r in receipts if r['route'] == route),
                               'output_frames': route_frames[route]} for route in ('a','b','native')}
    c1 = root/'candidates/C1/predictions.jsonl'
    for control in ['C1-frame','C1-segment']:
        folder = root/'stability_controls'/control; folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(c1, folder/'predictions.jsonl')
    atomic_json(root/'candidate_manifest.json', {'parent': 'P2', 'protocol_sha256': protocol['protocol_sha256'],
        'candidates': manifests, 'official_test_labels_accessed': False, 'training_started': False})
    atomic_json(root/'geometry_audit.json', {'by_candidate': geometry, 'geometry_is_not_quality_evaluation': True})
    atomic_json(root/'route_coverage.json', {'parent': 'P2', 'videos': dict(route_counts), 'output_frames': dict(route_frames),
        'changes_by_candidate': changed, 'B_risk': 'only transposed LIVE evidence; native B has not been validated on public labels',
        'constant_evidence_parent': 'P1 diagnostic only; P2 benefit requires official evaluation'})
    atomic_json(root/'qp_scale_audit.json', {'by_candidate': qp,
        'registered_postprocessing': protocol['registered_postprocessing'],
        'stability_controls': {'all_video_anchor_and_shot_schedules_equal': all(r['metrics']['C1']['C1_frame_segment_schedule_equal'] for r in receipts),
            'frame_sha256': sha(root/'stability_controls/C1-frame/predictions.jsonl'),
            'segment_sha256': sha(root/'stability_controls/C1-segment/predictions.jsonl'),
            'identical_to_C1': True, 'cross_shot_median_used': False,
            'conclusion': 'constant actions have the same within-shot median; no additional candidate or benefit claimed'}})
    atomic_json(root/'temporal_identity_check.json', {'parent': 'P2', 'by_candidate': temporal})
    atomic_json(root/'generation_complete.json', {'status': 'passed', 'sources': len(receipts), 'GPU_used': False,
        'protocol_sha256': protocol['protocol_sha256'], 'candidate_zip_hashes': {k:v['zip_sha256'] for k,v in manifests.items()},
        'final_artifact_hashes': {str(f.relative_to(root)):sha(f) for f in
            [root/n for n in ['candidate_manifest.json','geometry_audit.json','route_coverage.json',
                             'qp_scale_audit.json','temporal_identity_check.json']]+[
                root/'candidates'/k/n for k in POLICIES for n in ['predictions.jsonl', k+'_P2_Scale.zip']]}})
    return manifests


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit', type=int, help='CPU smoke in a separate root only; cannot finalize a partial submission')
    a = p.parse_args(); a.root = a.root.resolve(); protocol = load_protocol(a.root)
    budget_ledger(a.root/'gpu_budget_ledger.json')
    if (a.root/'generation_complete.json').exists():
        done = json.loads((a.root/'generation_complete.json').read_text())
        if done['protocol_sha256'] != protocol['protocol_sha256']: raise ValueError('completed protocol changed')
        for name, digest in done['final_artifact_hashes'].items():
            if sha(a.root/name) != digest: raise ValueError('completed output changed: '+name)
        print(json.dumps({'status':'already_complete','sources':done['sources']})); return
    files = sorted((a.root/'inputs/p1p2_records').glob('*.json'))
    if a.limit: files = files[:a.limit]
    receipts = []; started = time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(process_one, (str(a.root), protocol, str(f))) for f in files]
        for future in as_completed(futures):
            r = future.result(); receipts.append(r)
            atomic_json(a.root/'progress.json', {'completed_sources':len(receipts),'requested_sources':len(files),
                'elapsed_CPU_wall_seconds':time.monotonic()-started,'GPU_used':False,'parent':'P2'})
    if a.limit:
        print(json.dumps({'status':'CPU_smoke_only','sources':len(receipts),'no_submission_finalized':True})); return
    result = finalize(a.root, receipts, protocol)
    print(json.dumps({'status':'passed','candidates':list(result),'sources':len(receipts),'GPU_used':False}))


if __name__ == '__main__':
    main()
