"""Apply trained Scale actions to an immutable P1 deployed-center manifest.

Only the official three-coordinate rectangle is emitted. Position drift, invalid
JSON and a missing P1/Scale identity fail closed, without hidden fallbacks.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from aic.scale_data import read_jsonl
from aic.scale_geometry import checked_scale_action, deployed_center


def validate_position_provenance(rows, parent_identity):
    for row in rows:
        if (row.get('parent_identity_sha256') != parent_identity
                or row.get('position_kind') != 'P1_final_deployed'
                or row.get('position_source') != 'P1_frozen'):
            raise ValueError('inference P1 deployed-position identity mismatch')
        center = deployed_center(row['p1_aic_bbox_xyw'], row['target_ratio'])
        if list(center) != row.get('center_xy'):
            raise ValueError('inference center differs from the original deployed P1 bbox')


def validate_checkpoint_promotion(path, expected_sha):
    from scripts.check_scale_gate import assess_dpo_metrics, assess_sft_metrics
    report = json.loads((path / 'evaluation.json').read_text())
    if report.get('checkpoint_sha256') != expected_sha:
        raise ValueError('inference checkpoint differs from the measured deployment evidence')
    if report.get('reference_frozen_verified'):
        if not assess_dpo_metrics(report):
            raise ValueError('DPO did not pass deployment promotion; retain the frozen SFT checkpoint')
    elif not assess_sft_metrics(report):
        raise ValueError('SFT did not pass deployment promotion')


def validate_submission_positions(p1_records, inputs, index):
    from aic.contract import validate_submission
    validate_submission(p1_records,index).raise_for_errors()
    expected={(record['video_id'],p['frame']):p['bboxes'] for record in p1_records for p in record['predictions']}
    observed=set()
    for row in inputs:
        key=(row['source_id'],row.get('frame_id'))
        if key in observed or key not in expected or row['p1_aic_bbox_xyw'] != expected[key]:
            raise ValueError('input manifest differs from the original P1 selected frames/bboxes')
        meta=index[row['source_id']]
        if (row['image_width'],row['image_height'],tuple(row['target_ratio'])) != (meta.width,meta.height,meta.target_ratio):
            raise ValueError('input manifest differs from official deployment metadata')
        observed.add(key)
    if observed != expected.keys():
        raise ValueError('input manifest does not cover every original P1 selected frame')


def infer_rows(backend, rows, bins):
    outputs=[]
    for row in rows:
        if row.get('position_source')!='P1_frozen':
            raise ValueError('inference requires the original frozen P1 deployed position')
        action=backend.predict([row])[0]
        if type(action) is not int or not 0<=action<len(bins):
            raise ValueError('invalid model Scale action')
        crop=checked_scale_action(row['image_width'],row['image_height'],row['target_ratio'],row['center_xy'],bins[action])
        if not crop['valid']:raise ValueError('position_shift_invalid')
        outputs.append({'source_id':row['source_id'],'image_id':row['image_id'],'frame':row.get('frame_id'),
                        'scale_id':action,'bboxes':crop['aic_bbox_xyw'],'position_drift_px':crop['position_drift_px']})
    return outputs


def write_scaled_submission(p1_records, action_rows, index, output, bins, parameter_count):
    """Apply actions to the original selected frames, then use the native validator."""
    from aic.contract import write_submission
    from aic.scale_runtime import apply_scale_to_p1_predictions
    actions = {}
    for row in action_rows:
        key = (row['source_id'], row['frame'])
        if key in actions or type(row['frame']) is not int:
            raise ValueError('Scale actions require unique original decoded frame IDs')
        actions[key] = row['scale_id']
    result, used = [], set()
    for record in p1_records:
        video = record['video_id']; metadata = index[video]
        ratio = tuple(int(v) for v in metadata.target_ratio)
        if ratio != metadata.target_ratio:
            raise ValueError('Scale export requires the registered integer target ratio')
        frames = [item['frame'] for item in record['predictions']]
        requested = {(video, frame) for frame in frames}
        if not requested <= actions.keys():
            raise ValueError('Scale actions do not cover every original P1 selected frame')
        updated = apply_scale_to_p1_predictions(record, {frame:actions[(video,frame)] for frame in frames},
            metadata.width, metadata.height, ratio, bins)
        updated['model_params_b'] = parameter_count / 1_000_000_000
        # Current eligibility is by complete inference parameters. The shared
        # validator's historical "final" branch still requires an obsolete
        # MB-only field. Use its full coverage/geometry/parameter validation
        # without enabling that legacy byte-size tier. Preserve measurements
        # separately; do not emit a misleading legacy eligibility field.
        updated.pop('model_size_mb',None)
        result.append(updated); used |= requested
    if used != actions.keys():
        raise ValueError('Scale actions contain frames outside the original P1 selection')
    return write_submission(output, result, index, stage='preliminary',
                            actual_model_params_b=parameter_count / 1_000_000_000)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--checkpoint-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu',choices=['cpu'])
    p.add_argument('--p1-predictions',type=Path)
    p.add_argument('--index',type=Path)
    p.add_argument('--submission-output',type=Path)
    a=p.parse_args();c=json.loads(a.config.read_text())
    submission_args=[a.p1_predictions,a.index,a.submission_output]
    if any(submission_args) and not all(submission_args):
        p.error('--p1-predictions, --index and --submission-output must be supplied together')
    if not c.get('model_smoke',{}).get('real_model_forward_backward_passed'):
        raise ValueError('real model smoke is required before final Scale inference')
    inputs=read_jsonl(a.input)
    validate_position_provenance(inputs,c['parent']['identity_sha256'])
    validate_checkpoint_promotion(a.checkpoint,a.checkpoint_sha256)
    if a.submission_output:
        from aic.contract import load_index, load_jsonl
        originals=load_jsonl(a.p1_predictions);index=load_index(a.index)
        validate_submission_positions(originals,inputs,index)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    from aic.scale_videoora import create_backend
    from scripts.run_scale_stage import load_initialization
    b=create_backend(c,a.device);load_initialization(b,a.checkpoint,a.checkpoint_sha256)
    rows=infer_rows(b,inputs,c['data']['scale_bins']['values'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    if a.submission_output:
        write_scaled_submission(originals,rows,index,
            a.submission_output,c['data']['scale_bins']['values'],b.total_inference_elements)
if __name__=='__main__':main()
