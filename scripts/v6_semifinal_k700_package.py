"""V6: rebuild the best scored <=9B system as a semifinal submission package.

Semifinal contract (V6 directive): total parameters <= 9B declared per row via
model_params_b, reproducible from original videos, no teacher-cache dependency.
Base: SUB_E_INTERNVIDEO2_NATIVE_V1 (InternVideo2 Stage1-1B native QVH head +
true_face_smooth YuNet spatial) — official platform score 34.43 (K700), the
best scored <=9B system.  Predictions are carried over byte-for-byte (the
platform already scored this exact file); the only change is the metadata row
field model_params_b required by aic.contract V6 (size_coefficient_params).

Identity chain asserted, not assumed: source predictions sha256 must equal the
scored run_manifest value; content equality (all fields except the added
model_params_b) is re-checked after the rewrite; independent checker and ZIP
roundtrip re-run here.
"""
import argparse, hashlib, json, sys, tempfile, zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aic.contract import load_jsonl, load_index, write_submission, size_coefficient_params
from scripts.independent_submission_check import check
from scripts.all_select_yunet_release import sha256_file

PARAMS_B = 1.022373889  # SUB_E run_manifest parameter_count 1,022,373,889 (encoder+head+YuNet)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='/data/aic/official_test_20260926/submissions/SUB_E_INTERNVIDEO2_NATIVE_V1_FINAL')
    ap.add_argument('--out', default='/data/aic/official_test_20260926/submissions/SUB_SF_K700_SEMIFINAL_V1')
    ap.add_argument('--index', default='/data/aic/official_test_20260926/intake/index.enriched.jsonl')
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rm = json.loads((src / 'run_manifest.json').read_text().strip().removesuffix('\\n'))
    assert rm['submission_id'] == 'SUB_E_INTERNVIDEO2_NATIVE_V1'
    src_pred = src / 'predictions.jsonl'
    got = sha256_file(src_pred)
    assert got == rm['predictions_sha256'], f'source predictions sha mismatch {got}'
    frozen = json.loads((Path('/data/aic/experiments/NATIVE_CANDIDATE_V1') / 'SUB_E_frozen_manifest.json').read_text())
    assert frozen['index_sha256'] == sha256_file(a.index), 'index changed since SUB_E freeze'
    rows = load_jsonl(src_pred)
    assert len(rows) == rm['video_count'] == 174
    report = write_submission(out / 'predictions.jsonl', rows, load_index(a.index),
                              stage='final', actual_model_params_b=PARAMS_B)
    # content equality: identical modulo the added model_params_b field
    new_rows = load_jsonl(out / 'predictions.jsonl')
    for old, new in zip(rows, new_rows):
        strip = {k: v for k, v in new.items() if k != 'model_params_b'}
        assert strip == old, f'content drift at {old.get("video_id")}'
        assert new['model_params_b'] == PARAMS_B
    independent = check(a.index, out / 'predictions.jsonl', None, require_size=False)
    with tempfile.TemporaryDirectory() as tmp:
        zp = out / 'upload.zip'
        with zipfile.ZipFile(zp, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(out / 'predictions.jsonl', 'predictions.jsonl')
        with zipfile.ZipFile(zp) as zf:
            assert zf.namelist() == ['predictions.jsonl']
            zf.extractall(tmp)
        rt = check(a.index, Path(tmp) / 'predictions.jsonl', None, require_size=False)
        assert sha256_file(Path(tmp) / 'predictions.jsonl') == sha256_file(out / 'predictions.jsonl')
    man = {
        'submission_id': 'SUB_SF_K700_SEMIFINAL_V1',
        'status': 'READY_TO_UPLOAD_SEMIFINAL',
        'uploaded': False,
        'official_platform_score': None,
        'semifinal': {'model_params_b': PARAMS_B, 'tier': '<=9B', 'k_size': size_coefficient_params(PARAMS_B),
                      'declaration': 'per-row model_params_b; total inference components (InternVideo2 Stage1-1B encoder + native QVH temporal head + YuNet detector), no teacher-cache dependency at inference'},
        'parent': {'submission_id': 'SUB_E_INTERNVIDEO2_NATIVE_V1 (=K700)',
                   'official_platform_score': 34.43,
                   'zip_sha256': rm['zip_sha256'], 'predictions_sha256': rm['predictions_sha256'],
                   'note': 'predictions carried over byte-for-byte; platform scored this exact content 34.43 (K700)'},
        'identity': {'index_sha256': frozen['index_sha256'],
                     'predictions_sha256': sha256_file(out / 'predictions.jsonl'),
                     'zip_sha256': None, 'zip_bytes': None,
                     'changed_factor_vs_parent': 'none in prediction space; metadata-only (model_params_b rows) per semifinal contract',
                     'weights': frozen['weights'], 'code_files': frozen['code_files'],
                     'repro': {'entry': 'scripts/foundation_candidate_release.py (code sha256 in code_files)',
                               'weights_shas': [w['sha256'][:16] for w in frozen['weights']],
                               'precision': frozen['precision'], 'sample_fps': frozen['sample_fps'],
                               'threshold': frozen['threshold'], 'spatial_mode': frozen['spatial_mode'],
                               'temporal_postprocess': frozen['temporal_postprocess']}},
        'video_count': len(new_rows), 'prediction_count': sum(len(r['predictions']) for r in new_rows),
        'validator': report.to_dict(), 'independent': independent, 'unzip_independent': rt,
    }
    (out / 'run_manifest.json').write_text(json.dumps(man, indent=1, ensure_ascii=False) + '\n')
    man['identity']['zip_sha256'] = sha256_file(zp); man['identity']['zip_bytes'] = zp.stat().st_size
    (out / 'run_manifest.json').write_text(json.dumps(man, indent=1, ensure_ascii=False) + '\n')
    (out / 'upload.zip.sha256').write_text(man['identity']['zip_sha256'] + '  upload.zip\n')
    print(json.dumps({'submission_id': man['submission_id'], 'videos': man['video_count'],
                      'predictions': man['prediction_count'], 'model_params_b': PARAMS_B,
                      'k_size': man['semifinal']['k_size'], 'zip_sha256': man['identity']['zip_sha256'],
                      'zip_bytes': man['identity']['zip_bytes'], 'parent_score': 34.43,
                      'independent_valid': independent.get('valid'), 'roundtrip_valid': rt.get('valid')}))


if __name__ == '__main__':
    main()
