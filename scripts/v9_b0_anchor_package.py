"""P0-1: B0 semifinal anchor package.

Purpose is measurement, not score.  B0 is the 53,104-parameter YuNet observer
whose preliminary score (raw 34.42, 174 videos) is the only k_size-free
baseline in the project.  On the semifinal drop it answers the question the
three 10-01 submissions cannot separate:

    is the 48.67 -> 34.84 drop the k_size coefficient, or a harder drop?

Because B0's declared weight is 232,589 B, it lands in the <=100 MB tier with
k_size = 1.00, so its semifinal score IS its raw F_video.  Comparing that with
the preliminary 34.42 gives the drop-difficulty ratio directly:

    ratio = B0_semifinal / 34.42
    teacher_raw_implied = 34.84 / ratio / k_teacher

This package also carries the first `model_size_mb` declaration in the project
(0.2218 MB for YuNet).  Rule 01 s6.1 makes the field mandatory for the
semifinal and says a missing field can void a submission; the three 10-01
packages omit it and the platform scored them anyway, so B0 is the cheapest
possible probe of whether the field is accepted.
"""
import argparse, json, sys, tempfile, zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aic.contract import load_index, write_submission, load_jsonl  # noqa: E402
from scripts.independent_submission_check import check  # noqa: E402
from scripts.all_select_yunet_release import sha256_file  # noqa: E402

YU = {'name': 'YuNet 2023mar', 'parameters': 53104, 'bytes': 232589}
# 232589 B under either MB convention (1e6 or 2^20) is far below the 100 MB
# boundary, so k_size = 1.00 either way and the anchor stays unambiguous.
YU_MB = 232589 / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', type=Path, required=True)
    ap.add_argument('--b0', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--submission-id', default='LFM_B0_SEMIFINAL')
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)

    index = load_index(a.index)
    src = {r['video_id']: r for r in load_jsonl(a.b0)}
    rows = []
    for vid, r in src.items():
        assert vid in index, vid
        assert tuple(r['targetRatioWH']) == tuple(index[vid].targetRatioWH), vid
        rows.append({'video_id': vid, 'targetRatioWH': r['targetRatioWH'],
                     'model_size_mb': YU_MB, 'predictions': r['predictions']})
    rows.sort(key=lambda r: r['video_id'])

    dest = a.output / 'predictions.jsonl'
    validation = write_submission(dest, rows, index, stage='final',
                                  actual_model_size_mb=YU_MB).to_dict()
    independent = check(a.index, dest, int(round(YU_MB * 1e6)), require_size=True)

    zip_path = a.output / f'{a.submission_id}.zip'
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(dest, 'predictions.jsonl')
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            if zf.namelist() != ['predictions.jsonl']:
                raise ValueError('ZIP structure')
            zf.extractall(tmp)
        roundtrip = check(a.index, Path(tmp) / 'predictions.jsonl',
                          int(round(YU_MB * 1e6)), require_size=True)
        if sha256_file(Path(tmp) / 'predictions.jsonl') != sha256_file(dest):
            raise ValueError('ZIP roundtrip changed bytes')

    n_pred = sum(len(r['predictions']) for r in rows)
    man = {
        'submission_id': a.submission_id,
        'status': 'PACKAGED_NOT_UPLOADED_MEASUREMENT_ANCHOR',
        'uploaded': False,
        'official_platform_score': None,
        'parent_baseline': 'B0 preliminary raw 34.42 (174 videos, k_size not applied)',
        'primary_changed_factor': ('none - this is the unmodified B0 YuNet-face observer '
                                   'prediction set for the 426-video semifinal drop, '
                                   'repackaged with the mandatory model_size_mb field. '
                                   'Purpose is to separate the k_size coefficient from '
                                   'drop difficulty, not to chase score.'),
        'components': [YU],
        'total_parameters': YU['parameters'],
        'total_weight_bytes': YU['bytes'],
        'declared_model_size_mb': YU_MB,
        'declared_size_note': ('232,589 B = 0.232589 MB (1e6 convention) or 0.2218 MB '
                               '(2^20 convention); either way <=100 MB so k_size = 1.00 '
                               'and the platform score equals raw F_video. First project '
                               'package to declare model_size_mb (rules 01 s6.1).'),
        'k_size_implied': 1.0,
        'video_count': len(rows),
        'prediction_count': n_pred,
        'empty_videos': [r['video_id'] for r in rows if not r['predictions']],
        'validator': validation,
        'independent': independent,
        'unzip_independent': roundtrip,
        'predictions_sha256': sha256_file(dest),
        'zip_sha256': sha256_file(zip_path),
        'zip_bytes': zip_path.stat().st_size,
        'measurement_plan': {
            'why': ('teacher 48.67 -> 34.84 with an unchanged model is explained by '
                    'k_size (unknown, >9GB tier) and/or a harder 426-video drop; B0 '
                    'separates them because its k_size is pinned at 1.00'),
            'ratio': 'B0_semifinal / 34.42 = drop difficulty ratio',
            'then': ('teacher_raw_implied = 34.84 / ratio / k_teacher; if the implied raw '
                     'lands near 48.67 the drop is coefficient-only and the student '
                     'distillation gap is the whole story'),
        },
    }
    (a.output / 'manifest.json').write_text(json.dumps(man, indent=1, allow_nan=False) + '\n')
    (a.output / f'{a.submission_id}.zip.sha256').write_text(
        f"{man['zip_sha256']}  {a.submission_id}.zip\n")
    print(json.dumps({k: v for k, v in man.items()
                      if k not in ('measurement_plan',)}, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()