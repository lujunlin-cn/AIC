"""Frozen B3, bounded temporal-window controls around R26's official best.

No decoding, training, GT access or automatic competition upload.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import time
import zipfile
import zlib
from pathlib import Path

from aic.contract import load_index, load_jsonl, validate_submission
from independent_submission_check import check


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    a = ap.parse_args()
    started = time.monotonic()
    c = json.loads(Path(a.config).read_text())
    report = Path(c['report'])
    report.mkdir(parents=True, exist_ok=True)
    output = Path(c['output'])
    output.mkdir(parents=True, exist_ok=True)
    assert sha(c['index']) == c['index_sha256']
    assert sha(c['spatial']) == c['spatial_sha256']
    baseline = Path(c['baseline'])
    assert sha(baseline / (baseline.name + '.zip')) == c['baseline_zip_sha256']
    metas = load_jsonl(c['index'])
    spatial_rows = load_jsonl(c['spatial'])
    baseline_rows = load_jsonl(baseline / 'predictions.jsonl')
    spatial = {r['video_id']: r for r in spatial_rows}
    best = {r['video_id']: r for r in baseline_rows}
    assert len(metas) == len(spatial_rows) == len(baseline_rows) == len(spatial) == len(best) == 426
    assert set(best) == set(spatial) == {m['video_id'] for m in metas}
    allrows = {arm['name']: [] for arm in c['arms']}
    stats = {name: [] for name in allrows}
    replay_count = 0
    for m in metas:
        vid, n = m['video_id'], m['frame_count']
        preds = spatial[vid]['predictions']
        byframe = {p['frame']: p['bboxes'] for p in preds}
        assert len(preds) == len(byframe) == n and set(byframe) == set(range(n))
        k0 = round(.85 * n)
        start0 = (n - k0) // 2
        replay = [dict(frame=f, bboxes=byframe[f]) for f in range(start0, start0+k0)]
        assert replay == best[vid]['predictions'], vid
        replay_count += len(replay)
        for arm in c['arms']:
            k = round(arm['keep'] * n)
            start = (n-k)//2 + round(arm['shift']*n)
            assert k > 0 and 0 <= start <= n-k
            chosen = [dict(frame=f, bboxes=byframe[f]) for f in range(start, start+k)]
            allrows[arm['name']].append(dict(video_id=vid, targetRatioWH=m['targetRatioWH'],
                model_size_mb=c['parameters']/1e6, model_params_b=c['parameters']/1e9, predictions=chosen))
            overlap = max(0, min(start+k, start0+k0)-max(start, start0))
            stats[arm['name']].append(dict(video_id=vid, frames=n, kept=k, start=start,
                stop_exclusive=start+k, added_vs_best=k-overlap, removed_vs_best=k0-overlap))
    packages = {}
    for arm in c['arms']:
        name = arm['name']
        directory = output/name
        directory.mkdir(exist_ok=True)
        pred = directory/'predictions.jsonl'
        pred.write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in allrows[name]))
        independent = check(c['index'], pred)
        assert independent['valid'], independent
        contract = validate_submission(allrows[name], load_index(c['index']), stage='semifinal',
            actual_model_params_b=c['parameters']/1e9, require_sorted=True)
        assert contract.valid, contract.to_dict()
        zp = directory/(name+'.zip')
        with zipfile.ZipFile(zp, 'w') as z:
            zi = zipfile.ZipInfo('predictions.jsonl', date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, pred.read_bytes(), compresslevel=9)
        with zipfile.ZipFile(zp) as z:
            assert z.namelist() == ['predictions.jsonl']
            assert z.read('predictions.jsonl') == pred.read_bytes()
        ss = stats[name]
        manifest = dict(submission_id=name, status='PACKAGED_NOT_UPLOADED', official_score=None,
            uploaded=False, arm=arm, total_parameters=c['parameters'], parameter_limit=c['parameter_limit'],
            parameter_components=dict(LFM=448718848, spatial_head=1511425, YuNet=53104),
            spatial_weight_bytes_reported=903762857, precision='cached frozen B3 boxes, no new neural inference',
            model_size_mb_semantics='total inference parameters / 1e6, NOT file bytes',
            index_sha256=sha(c['index']), spatial_sha256=sha(c['spatial']),
            baseline_zip_sha256=c['baseline_zip_sha256'], config_sha256=sha(a.config),
            source_code_sha256=sha(__file__), predictions_sha256=sha(pred), zip_sha256=sha(zp),
            zip_bytes=zp.stat().st_size, video_count=len(ss), prediction_count=sum(s['kept'] for s in ss),
            added_vs_best=sum(s['added_vs_best'] for s in ss), removed_vs_best=sum(s['removed_vs_best'] for s in ss),
            same_boxes_as_b3=True, validation=independent, contract=contract.to_dict())
        dump(directory/'manifest.json', manifest)
        (directory/(name+'.zip.sha256')).write_text(sha(zp)+'  '+zp.name+'\n')
        dump(report/(name+'.json'), manifest)
        packages[name] = manifest
    dump(report/'run.json', dict(config=c, packages=packages, baseline_replayed_frames=replay_count,
        wall_seconds=time.monotonic()-started, python=platform.python_version(), zlib=zlib.ZLIB_VERSION,
        command=['PYTHONPATH=.', 'python', 'scripts/r27_center_windows.py', '--config', a.config],
        git_parent_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        protected_sources_opened=False, training=False, official_upload_performed=False))
    print(json.dumps({name: {k: m[k] for k in ['prediction_count', 'added_vs_best', 'removed_vs_best', 'zip_sha256']}
        for name, m in packages.items()}, indent=2))


if __name__ == '__main__':
    main()
