"""Audit complete real-P1 Train coverage and propose a train-only locked recipe.

No GPU is loaded. A pilot can produce diagnostic evidence but cannot lock the
formal config. Stage gates remain responsible for model/execution readiness.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from aic.scale_audit import position_conditioned_coverage
from aic.scale_calibration import calibrate
from aic.scale_data import read_jsonl
from aic.scale_parent import atomic_json, file_sha
from scripts.prepare_scale_manifests import prepare


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--positions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pilot',action='store_true')
    p.add_argument('--lock-run-config',action='store_true')
    a=p.parse_args()
    if a.pilot and a.lock_run_config:raise ValueError('pilot cannot lock formal training config')
    c=json.loads(a.config.read_text())
    labels=read_jsonl(a.labels);positions=read_jsonl(a.positions)
    # prepare() requires every selected public label and rejects unmatched cache
    # entries, source leakage, hash mismatch and Train/Val identical image bytes.
    joined=prepare(labels,positions,c['parent']['identity_sha256'])
    train=[r for r in joined if r['split']=='train']
    required=set(c['data']['training_sources'])
    if {r['dataset_source'] for r in train} != required:
        raise ValueError('calibration training datasets differ from registered recipe')
    result=calibrate(train,c['data']['scale_bins']['values'])
    result.update(pilot=a.pilot,labels_sha256=file_sha(a.labels),
                  positions_sha256=file_sha(a.positions),parent_identity_sha256=c['parent']['identity_sha256'])
    a.output.mkdir(parents=True,exist_ok=False)
    atomic_json(a.output/'calibration.json',result)
    evidence=file_sha(a.output/'calibration.json')
    coverage=position_conditioned_coverage(train,result['values'],result['minimum_position_conditioned_iou'])
    coverage.update(pilot=a.pilot,calibration_evidence_sha256=evidence,
                    valid_scale_sft_labels=sum(v['valid_scale_sft_labels'] for v in coverage['by_dataset_ratio'].values()),
                    validation_used_for_train_labels=False)
    atomic_json(a.output/'position_conditioned_scale_coverage.json',coverage)
    lines=['# Position-conditioned Scale coverage','',f'Pilot: {a.pilot}',
           f"Effective labels: {coverage['valid_scale_sft_labels']}",'',
           '| Dataset/ratio | Input views | Effective labels | Independent label sources | Position mismatch rate |',
           '|---|---:|---:|---:|---:|']
    for key,value in coverage['by_dataset_ratio'].items():
        lines.append(f"| {key} | {value['original_samples']} | {value['valid_scale_sft_labels']} | {value['valid_label_sources']} | {value['position_mismatch_rate']:.5f} |")
    (a.output/'position_conditioned_scale_coverage.md').write_text('\n'.join(lines)+'\n')
    manifest={}
    for split in ('train','val'):
        path=a.output/(split+'.jsonl')
        path.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in joined if r['split']==split))
        manifest[split]=dict(path=str(path.resolve()),sha256=file_sha(path),
                             rows=sum(r['split']==split for r in joined))
    atomic_json(a.output/'manifest.json',manifest)
    if a.lock_run_config:
        if result['scale_collapse_in_supervision'] or not result['minimum_samples_gate_passed']:
            raise ValueError('train statistics fail stability/collapse criteria; config unchanged')
        old=c['data']['scale_bins']
        c['data']['scale_bins']=dict(old,values=result['values'],count=result['count'],
            status='fitted_from_train_only_and_P1_coverage_locked',fit_method='source_balanced_train_greedy_deploy_IoU',
            fit_source_manifest_sha256=file_sha(a.labels),calibration_evidence_sha256=evidence)
        c['data']['minimum_position_conditioned_iou']=result['minimum_position_conditioned_iou']
        c['data']['geometry_preference_delta_iou']=result['geometry_preference_delta_iou']
        for split,field in [('train','train'),('val','validation')]:
            c['data'][field+'_manifest']=manifest[split]['path']
            c['data'][field+'_manifest_sha256']=manifest[split]['sha256']
        c['training']['majority_scale_upper_bound']=result['majority_scale_upper_bound']
        c['action_protocol']['grid_scale_ids']=[round(i*(result['count']-1)/3) for i in range(4)]
        c['position_coverage']=dict(status='audited',coverage_requirement_locked=True,
                                   gate_passed=True,evidence_sha256=evidence,
                                   coverage_requirement_basis='train source-level stability and noncollapsed label supervision')
        c['promotion']['sft']=dict(minimum_label_coverage=result['train_valid_label_source_dataset_macro'],
                                   coverage_evidence_sha256=evidence)
        c['promotion']['preference']=dict(minimum_pairs=sum(x['minimum_pairs'] for x in result['source_stability_requirements'].values()),
            minimum_sources=sum(x['minimum_independent_sources'] for x in result['source_stability_requirements'].values()),
            evidence_sha256=evidence,method='per-dataset independent-source effect precision requirements',
            by_dataset=result['source_stability_requirements'])
        # A changed mapping requires a fresh real-weight action smoke. Existing
        # checks remain evidence for the previous bins, not a free pass here.
        c['model_smoke']['real_model_forward_backward_passed']=False
        c['formal_gpu_training_authorized']=False
        atomic_json(a.config,c)
        atomic_json(a.output/'scale_bins.json',c['data']['scale_bins'])
    print(json.dumps(dict(status=result['status'],pilot=a.pilot,config_locked=a.lock_run_config,
                          valid_labels=coverage['valid_scale_sft_labels'],bins=result['values']),indent=2))


if __name__=='__main__':main()
