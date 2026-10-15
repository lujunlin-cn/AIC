"""Verify the downloaded R25 package with two validators and exact hashes."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from independent_submission_check import check
from aic.contract import load_jsonl, load_index, validate_submission, size_coefficient_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', required=True)
    ap.add_argument('--package', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    package = Path(args.package)
    manifest = json.loads((package / 'manifest.json').read_text())
    zpath = package / 'R25_AHA_BF16_K085.zip'
    digest = hashlib.sha256(zpath.read_bytes()).hexdigest()
    assert digest == manifest['zip_sha256']
    with zipfile.ZipFile(zpath) as z:
        assert z.namelist() == ['predictions.jsonl']
        payload = z.read('predictions.jsonl')
    assert hashlib.sha256(payload).hexdigest() == manifest['predictions_sha256']
    # Avoid extracting arbitrary archive names: the sole member is fixed above.
    pred = package / 'predictions.jsonl'
    pred.write_bytes(payload)
    independent = check(args.index, pred, require_size=True)
    rows = load_jsonl(pred)
    parameters = manifest['run']['total_parameters']
    assert parameters == 8480646545
    assert all(r['model_size_mb'] == parameters / 1e6 for r in rows)
    # Validate the actual parameter total independently of the legacy field name.
    size_coefficient_params(parameters / 1e9)
    contract = validate_submission(rows, load_index(args.index), stage='semifinal', require_sorted=True)
    assert contract.valid, contract.to_dict()
    pervideo = manifest['per_video']
    assert len(pervideo) == independent['videos'] == 426
    assert max(x['native_forward_max_abs_difference'] for x in pervideo) < 1e-5
    assert manifest['bbox_changes_on_common_frames'] == 0
    result = dict(valid=True, independent=independent, contract=contract.to_dict(),
                  zip_sha256=digest, total_parameters=parameters,
                  parameter_limit=9000000000,
                  native_forward_max_abs_difference=max(x['native_forward_max_abs_difference'] for x in pervideo),
                  keep_fraction=sum(x['kept'] for x in pervideo)/sum(x['frame_count'] for x in pervideo),
                  constant_score_videos=sum(x['score_std'] == 0 for x in pervideo),
                  zip_root=['predictions.jsonl'], official_score=None)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != 'contract'}, indent=2))


if __name__ == '__main__':
    main()
