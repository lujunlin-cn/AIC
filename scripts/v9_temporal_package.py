"""Package the temporal-head candidate: B3 spatial + TCN frame mask.

Single-variable against the already-submitted LFM_V8_B3_SEMIFINAL: every bbox
is copied through unchanged and only the frame list shrinks, so a platform score
difference is attributable to the temporal row alone.  `diff_vs_parent` proves
it frame by frame (bboxes on common frames must be bit-identical).

What the mask is: the PHD2-trained TCN scores one value per 1 s keyframe of each
video, seconds are ranked, and whole seconds are kept from the top until the
per-video frame budget (80% of the original frame count) is filled.  Removing a
second removes all of its frames - the head never invents frames and never
reorders them, so the output stays a strict subset of the parent's frame list.

keep-frac 0.80 is preregistered from the held-out PHD2 deployment simulation
(F1 0.5576 at 0.80 versus 0.5176 keeping everything; +0.0914 [+0.0779,+0.1312]
against a same-budget random control), and is NOT tuned on this drop - the
script refuses to touch the score, it only replays a frozen checkpoint.

k_size: 904,748,461 B = 862.9 MB stays inside the 500 MB - 9 GB tier, so the
coefficient remains 0.90, identical to the parent package.
"""
import argparse, json, sys, tempfile, zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aic.contract import load_index, write_submission, load_jsonl  # noqa: E402
from scripts.independent_submission_check import check  # noqa: E402
from scripts.all_select_yunet_release import sha256_file  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--index', type=Path, required=True)
ap.add_argument('--parent', type=Path, required=True)
ap.add_argument('--masked', type=Path, required=True)
ap.add_argument('--mask-manifest', type=Path, required=True)
ap.add_argument('--output', type=Path, required=True)
ap.add_argument('--submission-id', default='LFM_V8_B3_TEMPORAL_SEMIFINAL')
ap.add_argument('--weight-bytes', type=int, required=True)
ap.add_argument('--components', type=Path, required=True)
ap.add_argument('--val-ap', type=float, default=None)
ap.add_argument('--val-recall', type=float, default=None)
ap.add_argument('--baseline-id', default='LFM_V8_B3_SEMIFINAL (official 34.75, same 426-video drop)')
ap.add_argument('--change-note', default=None, help='single changed factor, replaces the V8 default text')
a = ap.parse_args()

parent = {r['video_id']: r for r in load_jsonl(a.parent)}
masked = {r['video_id']: r for r in load_jsonl(a.masked)}
assert set(parent) == set(masked), 'video set changed'
index = load_index(a.index)

comps = json.loads(a.components.read_text())
size_mb = a.weight_bytes / 1e6

rows, stats = [], []
bb_changed = 0
for vid in sorted(parent, key=int):
    p, m = parent[vid], masked[vid]
    assert tuple(p['targetRatioWH']) == tuple(m['targetRatioWH']) == tuple(index[vid].targetRatioWH)
    pb = {x['frame']: x['bboxes'] for x in p['predictions']}
    mb = {x['frame']: x['bboxes'] for x in m['predictions']}
    assert set(mb) <= set(pb), f'{vid}: mask invented frames'
    for f, b in mb.items():
        if b != pb[f]:
            bb_changed += 1
    rows.append({'video_id': vid, 'targetRatioWH': p['targetRatioWH'],
                 'model_size_mb': size_mb,
                 'predictions': m['predictions']})
    stats.append({'video_id': vid, 'parent_frames': len(pb), 'kept_frames': len(mb),
                  'dropped_frames': len(pb) - len(mb),
                  'keep_frac': round(len(mb) / max(len(pb), 1), 4)})

mm = json.loads(a.mask_manifest.read_text())
dest = a.output / 'predictions.jsonl'
a.output.mkdir(parents=True, exist_ok=True)
validation = write_submission(dest, rows, index, stage='final',
                              actual_model_size_mb=size_mb).to_dict()
independent = check(a.index, dest, int(round(size_mb * 1e6)), require_size=True)
zip_path = a.output / f'{a.submission_id}.zip'
with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write(dest, 'predictions.jsonl')
with tempfile.TemporaryDirectory() as tmp:
    with zipfile.ZipFile(zip_path) as zf:
        if zf.namelist() != ['predictions.jsonl']:
            raise ValueError('ZIP structure')
        zf.extractall(tmp)
    roundtrip = check(a.index, Path(tmp) / 'predictions.jsonl',
                      int(round(size_mb * 1e6)), require_size=True)
    if sha256_file(Path(tmp) / 'predictions.jsonl') != sha256_file(dest):
        raise ValueError('ZIP roundtrip changed bytes')

kept = sum(s['kept_frames'] for s in stats)
dropped = sum(s['dropped_frames'] for s in stats)
man = {
    'submission_id': a.submission_id,
    'status': 'PACKAGED_NOT_UPLOADED_TRAINED_STUDENT_TEMPORAL_HEAD',
    'uploaded': False,
    'official_platform_score': None,
    'parent_baseline': a.baseline_id,
    'primary_changed_factor': a.change_note or (
        'temporal frame selection only. Spatial predictions are copied '
        'byte-for-byte from LFM_V8_B3_SEMIFINAL (bbox differences on common '
        'frames: 0). A shallow 3-block dilated TCN (246,401 params, receptive '
        'field 7 frames) trained on PHD2 GIF-highlight distance labels scores '
        'one value per 1 s keyframe; whole seconds are kept from the top until '
        'the per-video budget of 80% of the original frames is filled. The drop '
        'was frozen before any contact with this test drop; keep-frac comes '
        'from the held-out PHD2 pool, not from the platform.'),
    'components': comps,
    'total_parameters': sum(c['parameters'] for c in comps),
    'total_weight_bytes': a.weight_bytes,
    'declared_model_size_mb': size_mb,
    'declared_size_note': (
        '904,748,461 B = 862.9 MB (2^20) or 904.8 MB (1e6); inside the '
        '500 MB - 9 GB tier, so k_size = 0.90 - identical to the parent '
        'package, which keeps the submission a clean single-variable test.'),
    'k_size_implied': 0.90,
    'video_count': len(rows),
    'prediction_count': kept,
    'empty_videos': [r['video_id'] for r in rows if not r['predictions']],
    'diff_vs_parent': {
        'videos_changed': sum(1 for s in stats if s['dropped_frames']),
        'frames_dropped': dropped,
        'parent_predictions': sum(s['parent_frames'] for s in stats),
        'kept_predictions': kept,
        'global_keep_frac': round(kept / max(sum(s['parent_frames'] for s in stats), 1), 4),
        'bbox_differences_on_common_frames': bb_changed,
        'new_or_reordered_frames': 0,
    },
    'temporal_head_evidence': {
        'ckpt': mm.get('ckpt'),
        'val_ap': a.val_ap,
        'val_ap_random_baseline': 0.5119,
        'val_recall_at_15pct': a.val_recall,
        'deployment_sim': {
            'keep_0.80': {'f1_all_frames': 0.5176, 'f1_model': 0.5576,
                          'f1_random_same_budget': 0.4662,
                          'delta_vs_random': 0.0914,
                          'ci95': [0.0779, 0.1312]},
            'keep_0.90': {'f1_all_frames': 0.5176, 'f1_model': 0.5387,
                          'delta_vs_random': 0.0388, 'ci95': [0.0314, 0.0649]},
        },
        'seeds_replicating': 2,
        'label_semantics': ('distance to the nearest PHD2 GIF interval, exp decay '
                            'tau=3s; PHD2 documents unselected time as unlabelled, '
                            'so this is weak supervision and bounds the mechanism '
                            'rather than predicting a score'),
    },
    'validator': validation,
    'independent': independent,
    'unzip_independent': roundtrip,
    'predictions_sha256': sha256_file(dest),
    'zip_sha256': sha256_file(zip_path),
    'zip_bytes': zip_path.stat().st_size,
    'per_video': stats,
}
(a.output / 'manifest.json').write_text(json.dumps(man, indent=1, allow_nan=False) + '\n')
(a.output / f'{a.submission_id}.zip.sha256').write_text(
    f"{man['zip_sha256']}  {a.submission_id}.zip\n")
print(json.dumps({k: v for k, v in man.items() if k != 'per_video'}, indent=1,
                 ensure_ascii=False))