"""CPU-only last diagnostic round. Never launches a model, optimizer or GPU job."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from aic.scale_final_preflight import center_errors, center_jitter, constants_oracle, content_probe, validate_join
from scripts.run_final_p1_cache import read_rows, sha, write


def analyze(a):
    root = a.root
    target = root / 'preview' if a.preview else root
    m = json.loads((root / 'p1_cache_manifest.json').read_text())
    labels = read_rows(root / 'inputs/cohort.jsonl')
    positions = []
    for path in sorted((root / 'records').glob('*.json')):
        positions.extend(json.loads(path.read_text()))
    joined = validate_join(labels, positions, m)
    bins_path = root.parent / 'config/scale_bins.json'
    bins = json.loads(bins_path.read_text())['values']
    ledger = json.loads((root / 'gpu_budget_ledger.json').read_text())
    ongoing = any(r['ended_at'] is None for r in ledger['runs'])
    if ongoing and not a.preview:
        raise ValueError('GPU task still running: final reports must await actual stop receipt')
    exposure_path = root / 'parent_exposure_audit.json'
    exposure = json.loads(exposure_path.read_text()) if exposure_path.exists() else {}
    feature_path = root / 'visual_features.json'
    features = json.loads(feature_path.read_text())['features_by_image_sha256'] if feature_path.exists() else {}
    outputs = {n: {'by_dataset_and_route': {}} for n in (
        'p1_center_error', 'center_jitter_analysis', 'constant_scale_results', 'scale_oracle_results')}
    probes = {}
    exclusion = {}
    for dataset in ('LIVE_YT_VC', 'GNMC'):
        for branch, ratio in [('A', [9, 16]), ('B', [16, 9])]:
            key = dataset + '/' + branch
            rows = [r for r in joined if r['dataset_source'] == dataset and r['target_ratio'] == ratio]
            proof = exposure.get('branches', {}).get(branch, {})
            allowed = proof.get('independent_validation_sources') if dataset == 'LIVE_YT_VC' else None
            eligible = [r for r in rows if r['split'] == 'train' or
                        (allowed is not None and r['source_id'] in allowed)]
            excluded = sorted({r['source_id'] for r in rows if r['split'] == 'val' and r not in eligible})
            exclusion[key] = {'excluded_validation_sources': excluded,
                'known_P1_finetuning_exposure_verified': allowed is not None,
                'parent_training_provenance': str(exposure_path) if exposure_path.exists() else None,
                'reason': None if allowed is not None else 'known P1 training/validation separation unverified'}
            if not rows:
                for out in outputs.values():
                    out['by_dataset_and_route'][key] = {'result': None,
                        'reason': 'no complete actual final-P1 Train/Val cache for this dataset in the 1h LIVE run'}
                probes[key] = {'learned_minus_best_constant': None, 'reason': 'actual final-P1 inputs unavailable'}
                continue
            error = center_errors(eligible, a.bootstrap)
            error['descriptive_all_sampled_sources'] = center_errors(rows, a.bootstrap)
            error['independence'] = exclusion[key]
            outputs['p1_center_error']['by_dataset_and_route'][key] = error
            outputs['center_jitter_analysis']['by_dataset_and_route'][key] = center_jitter(eligible, bins, a.bootstrap)
            constant, oracle = constants_oracle(eligible, bins, a.bootstrap)
            outputs['constant_scale_results']['by_dataset_and_route'][key] = constant
            outputs['scale_oracle_results']['by_dataset_and_route'][key] = oracle
            probes[key] = content_probe(eligible, bins, features, a.bootstrap)
    physical = {(r['dataset_source'], r['source_id'], r.get('frame_id', r['image_id'])) for r in joined}
    reused = [r for r in joined if r.get('reused_prior_cache_sha256')]
    new = [r for r in joined if not r.get('reused_prior_cache_sha256')]
    coverage = {'completed_physical_frames': len(physical), 'completed_view_frames': len(joined),
        'new_in_this_GPU_run_physical_frames': len({(r['source_id'], r['frame_id']) for r in new}),
        'new_in_this_GPU_run_view_frames': len(new),
        'reused_prior_physical_frames': len({(r['source_id'], r['frame_id']) for r in reused}),
        'reused_prior_view_frames': len(reused), 'original_expected_physical_frames': 10800,
        'clean_planned_physical_frames': m['planned_physical_frames'],
        'physical_frame_coverage_of_original_request': len(physical)/10800,
        'by_split': {}, 'by_route': {}, 'GPU_budget_ledger': ledger,
        'sampling': m['sampling'], 'quarantined_train_sources': 5,
        'scope': 'completed sources only; interrupted partial source outputs excluded; not full-data coverage'}
    for split, original_total, clean_total in [('train', 1600, 1595), ('val', 200, 200)]:
        selected = [r for r in joined if r['split'] == split and r['dataset_source'] == 'LIVE_YT_VC']
        source_count = len({r['source_id'] for r in selected})
        counts = Counter(r['source_id'] for r in selected if r['target_ratio'] == [9, 16])
        coverage['by_split'][split] = {'completed_sources': source_count, 'original_sources': original_total,
            'clean_sources': clean_total, 'original_source_coverage': source_count/original_total,
            'clean_source_coverage': source_count/clean_total,
            'frames_per_completed_source': dict(Counter(counts.values())),
            'source_ids': sorted(counts)}
    for branch, ratio in [('A', [9, 16]), ('B', [16, 9])]:
        rr = [r for r in joined if r['target_ratio'] == ratio]
        coverage['by_route'][branch] = {'view_frames': len(rr), 'sources': len({r['source_id'] for r in rr})}
    new_sources = len({r['source_id'] for r in new})
    seconds = ledger.get('cumulative_GPU_seconds')
    coverage['throughput'] = {'requested_benchmark_sources': 100, 'completed_new_sources': new_sources,
        '100_source_benchmark_completed': new_sources >= 100,
        'measured_100_source_runtime_seconds': seconds if new_sources == 100 else None,
        'new_sources_per_GPU_wall_second': new_sources/seconds if seconds else None,
        'partial_cohort_extrapolated_all_clean_sources_seconds': seconds/new_sources*1795 if seconds and new_sources else None,
        'estimate_warning': 'partial-cohort extrapolation including loading; not a completed 100-source benchmark or full-job timing',
        'frame_reduction': 'not used: all completed sources retain 6; original 24-anchor runtime cost does not scale with evaluation frame count'}
    positives = []
    for key, probe in probes.items():
        delta = probe.get('results', {}).get('geometry_plus_RGB', {}).get('learned_minus_best_constant', {})
        grid_delta = probe.get('results', {}).get('geometry_plus_RGB', {}).get('learned_minus_train_selected_grid_constant', {})
        increment = probe.get('incremental_RGB_over_geometry', {})
        intervals = [item.get('bootstrap_ci95') for item in (delta, grid_delta, increment)]
        if all(x is not None and x[0] > 0 for x in intervals):
            positives.append(key)
    readiness = {'Scale_SFT_started': False, 'DPO_started': False, 'formal_training_authorized': False,
        'actual_videoora_smoke_reused': m['smoke_evidence'], 'smoke_reexecuted': False,
        'content_probe': probes, 'independence_exclusions': exclusion,
        'reliable_positive_RGB_probe_groups': positives,
        'recommend_start_Scale_SFT': False,
        'reason': 'partial diagnostic cohort; no automatic training authorization; native B/GNMC deployment and full label calibration remain incomplete',
        'learning_merit_on_measured_cohorts': 'preliminary_positive_probe_requires_confirmation' if positives else 'no_reliable_positive_content_increment_established',
        'GNMC_historical_human_scale_probe': {
            'evidence': '../diagnostics/scale_predictability.json',
            'scope': 'human-scale regression, not final-P1 deployed policy gain; cannot substitute for this round\'s null GNMC results'},
        'DPO_necessity': None, 'soft_target_SFT_sufficiency': None,
        'next_if_learning_merit_established': 'compare Soft-Target SFT with best constant first; DPO needs separately verified independent MOS/ranking evidence',
        'SACD_used_for_preferences': False, 'LIVE_GNMC_IoU_relabelled_as_independent_human_preference': False,
        'license_review': 'project owner confirmed completed; not repeated',
        'official_hidden_data_used': False}
    revision_path = root/'code_revision.json'
    if revision_path.exists():
        revision = json.loads(revision_path.read_text())
        for path, expected_sha in revision['files'].items():
            if sha(path) != expected_sha:
                raise ValueError('diagnostic code revision changed: ' + path)
        git_commit = revision['git_commit']
    elif a.preview:
        git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    else:
        raise ValueError('final diagnostic code revision receipt missing')
    common = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'seed': 20261108,
        'metric_scope': 'source-macro final deployed spatial IoU; fixed P1/temporal set, QP and frame count; not official competition Score',
        'cohort_sha256': sha(root/'inputs/cohort.jsonl'), 'manifest_sha256': sha(root/'p1_cache_manifest.json'),
        'scale_bins_sha256': sha(bins_path), 'Scale_bins_refitted': False,
        'code_sha256': {'analysis_script': sha(__file__),
                        'analysis_module': sha(Path(__file__).resolve().parents[1]/'aic/scale_final_preflight.py')},
        'position_record_count': len(positions), 'git_commit': git_commit,
        'A_scope': 'LIVE native portrait crop', 'B_scope': 'transposed LIVE landscape crop; not native upright B validation',
        'source_selection': 'predeclared label-blind source hash; alternating splits; budget-censored completed cohort',
        'interval_scope': 'conditional on observed cohort and Train-selected configuration; no multiple-comparison or full-population guarantee'}
    for name, data in outputs.items():
        data['provenance'] = common; write(target/(name+'.json'), data)
    coverage['provenance'] = common; write(target/'p1_cache_coverage.json', coverage)
    readiness['provenance'] = common; write(target/'scale_learning_readiness.json', readiness)
    write_report(target, coverage, outputs, readiness)
    return coverage, readiness


def metric(x):
    if not x or x.get('mean') is None:
        return 'null'
    ci = x.get('bootstrap_ci95')
    return f"{x['mean']:.6f}" + (f" [{ci[0]:.6f}, {ci[1]:.6f}]" if ci else ' [CI=null]')


def write_report(out, coverage, outputs, readiness):
    lines = ['# Scale-SFT 启动前最后一轮诊断', '',
        '**本轮没有启动 SFT/DPO。当前不建议据此自动进入正式训练。**', '',
        '所有指标使用已冻结 P1 最终 QP 后中心、整数 `[x,y,w]` 和目标比例推导高度。原始时间集合、空间后处理和比较用帧集合在各 Scale 策略间完全相同；这是公开数据的空间诊断，不是官方总分。', '',
        '## 一小时预算与覆盖', '',
        f"GPU 累计设备墙钟：{coverage['GPU_budget_ledger'].get('cumulative_GPU_seconds')} 秒，包含两个进程的加载和推理，同一设备不重复计时。",
        f"本轮新完成 {coverage['new_in_this_GPU_run_physical_frames']} 个物理帧、{coverage['new_in_this_GPU_run_view_frames']} 个 A/B view；复用历史 {coverage['reused_prior_physical_frames']} 个物理帧。", '',
        '| Split | 完成源 | 原始源覆盖率 | 无泄漏源覆盖率 |', '|---|---:|---:|---:|']
    for split, v in coverage['by_split'].items():
        lines.append(f"| {split} | {v['completed_sources']} | {v['original_source_coverage']:.2%} | {v['clean_source_coverage']:.2%} |")
    lines += ['', '原始 Train 1,600 源中 5 源已有跨 split 图像重复，继续隔离，未将其混回训练诊断。每个完成源保留 6 个跨时间段、兼顾 GT 尺度的标注帧，两条路线分别运行原始整段 P1 路径。',
        '100 源吞吐测试受同一个一小时硬预算约束；若未达到 100，完整测试耗时为 null，仅报告实际部分的吞吐外推。降低评价帧数不能消除原始每源 24 个锚点的生成成本。', '',
        '## 最终部署常数与 GT 可见 Oracle', '',
        '| 数据/路线 | Train 最佳常数 | Val IoU [95% CI] | 相对 1.0 增量 [95% CI] | Oracle@8−最佳常数 [95% CI] |',
        '|---|---:|---|---|---|']
    for group, c in outputs['constant_scale_results']['by_dataset_and_route'].items():
        oracle = outputs['scale_oracle_results']['by_dataset_and_route'][group]
        gain = oracle.get('oracle_at_k', {}).get('8', {}).get('gain_over_best_constant') if oracle.get('oracle_at_k') else None
        lines.append(f"| {group} | {c.get('best_constant_scale')} | {metric(c.get('val_source_macro_iou'))} | {metric(c.get('val_gain_over_scale_1'))} | {metric(gain)} |")
    lines += ['', '常数仅在 Train 的 source-macro 上从 0.87、0.90、0.93、0.95、0.97、1.0 选择。Oracle@2/4/8 的八动作前缀同样仅由 Train 排序，Val 不选择常数、不重拟合 bins。GT 可见 Oracle 是理论上界；学习型模型 Greedy 为 null。LIVE B 是转置增强，原生 B 与 GNMC 缺最终 P1 Train/Val 缓存时均为 null。', '',
        '| 数据/路线 | P1 Greedy（Scale=1）| Oracle@2 | Oracle@4 | Oracle@8 |',
        '|---|---|---|---|---|']
    for group, o in outputs['scale_oracle_results']['by_dataset_and_route'].items():
        curve = o.get('oracle_at_k') or {}
        values = [metric(curve.get(str(k), {}).get('val_source_macro_iou')) for k in (2, 4, 8)]
        lines.append(f"| {group} | {metric(o.get('P1_greedy_scale_1'))} | {' | '.join(values)} |")
    lines += ['',
        '## 中心误差与样本内差异', '',
        '| 数据/路线 | Train 活动轴 MAE 像素 | Val 活动轴 MAE 像素 | Train MAE/尺寸 | Val MAE/尺寸 | Val−Train [95% CI] |',
        '|---|---|---|---|---|---|']
    for group, e in outputs['p1_center_error']['by_dataset_and_route'].items():
        tr = (e.get('train') or {}).get('absolute_active_axis_fraction')
        va = (e.get('val') or {}).get('absolute_active_axis_fraction')
        gap = e.get('train_validation_gap')
        gd = {'mean': gap['val_minus_train_normalized_active_axis_MAE'],
              'bootstrap_ci95': gap['independent_source_bootstrap_ci95']} if gap else None
        train_px = (e.get('train') or {}).get('absolute_active_axis_px')
        val_px = (e.get('val') or {}).get('absolute_active_axis_px')
        lines.append(f"| {group} | {metric(train_px)} | {metric(val_px)} | {metric(tr)} | {metric(va)} | {metric(gd)} |")
    lines += ['', '中心误差 JSON 同时给出 x/y 绝对像素误差、占宽/高比例、均值、中位数、P5/P95 和 source bootstrap。独立 Val 排除任何已知 P1 训练图像重复所在的整个视频源。训练内与 Val 的差异不等于已证明的因果乐观偏差，内容构成与历史模型选择仍可能影响比较。', '',
        '## 中心抖动与可预测性', '',
        '±1%、±3%、±5% 作用于活动轴；误差匹配扰动按独立 Val 源等权采样有符号归一化残差，保留偏置和幅度，并排除当前被评价的 Val 源。所有扰动只用于敏感性诊断，不修改 P1、不回写训练标签。位置投影漂移相对扰动目标计量，故意施加的中心位移单列。无合法窗口和非法动作均单独统计，没有通过 clamp 修复中心。', '']
    for group, j in outputs['center_jitter_analysis']['by_dataset_and_route'].items():
        if not j.get('conditions'):
            continue
        lines += [f'### {group}', '', '| 扰动 | Val 最优动作改变率 [95% CI] | Val Oracle IoU 变化 [95% CI] | 沿用原动作的 regret | 无合法窗比例 |', '|---|---|---|---|---|']
        for condition, value in j['conditions'].items():
            v = value.get('by_split', {}).get('val', {})
            lines.append(f"| {condition} | {metric(v.get('optimal_scale_changed_rate'))} | {metric(v.get('oracle_iou_change_common_legal'))} | {metric(v.get('holding_original_common_best_scale_regret'))} | {metric(v.get('no_legal_window_rate'))} |")
        lines.append('')
    lines += ['CPU 视觉探针固定 ridge alpha=10，仅用 Train 拟合/标准化八动作真实部署 IoU 向量；Val 上比较 `Learned−BestConstant`、`Learned−GridConstant` 和视觉相对几何特征增量。其目标不是人工原始尺度回归；它也不是 Video-ORA 训练结果。', '',
        '| 数据/路线 | 视觉探针−最佳常数 [95% CI] | 视觉增量（相对几何探针）[95% CI] |', '|---|---|---|']
    for group, probe in readiness['content_probe'].items():
        delta = probe.get('results', {}).get('geometry_plus_RGB', {}).get('learned_minus_best_constant')
        lines.append(f"| {group} | {metric(delta)} | {metric(probe.get('incremental_RGB_over_geometry'))} |")
    lines += ['', '## 决策与剩余限制', '',
        f"独立验证源上同时超过最佳常数、八动作最佳常数且具有正视觉增量的组：{readiness['reliable_positive_RGB_probe_groups']}。",
        'Oracle 空间不能作为启动大模型 SFT 的充分条件。小成本探针失败也不证明所有视觉语义都不可预测。当前为受预算限制的部分样本诊断，不替代完整数据覆盖、原生 B/GNMC 验证与标签校准。',
        '**是否值得立即启动 Scale-SFT：本轮不启动；由负责人结合这些源外差值及区间决定后续预算。**',
        '**是否需要 DPO，或 Soft-Target SFT 已足够：均为 null，尚未进行合法的比较实验。若后续确认学习价值，先用 Soft-Target SFT 与最佳常数公平比较；GAICD MOS 是独立监督，但固定中心有效覆盖仍需实际确认。SACD best_box 未被转换为排序；LIVE/GNMC 的 IoU 排序没有包装成人工偏好。**', '',
        '真实 Video-ORA Base/P1/Scale forward/backward、checkpoint 保存恢复和几何 Smoke Test 直接复用既有证据，本轮未重复。许可由负责人确认，未重复开展许可研究。所有未知结果保留 null。', '',
        '复现：`python -m scripts.analyze_scale_final_preflight --root artifacts/scale_sft_dpo/final_preflight`。详细输入、代码、P1、QP、frame-set、图像 SHA 与设备累计账本见相应 JSON；raw 原始锚点和整段轨迹缓存保留在 `workers/*/videos/`，最终逐帧输出在 `positions.jsonl`。', '',
        '方法说明：资源配额检测与源级数据审计使用 get-available-resources / exploratory-data-analysis 工作流；方法出处 Kassis, Agarwal, He, Patel & Brueckner (2026), Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents, DOI 10.48550/arXiv.2609.00065。该引用不是 Scale 收益证据；核验记录见 methodology_reference.json。']
    (out/'final_report.md').write_text('\n'.join(lines)+'\n')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', type=Path, default=Path('artifacts/scale_sft_dpo/final_preflight'))
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--preview', action='store_true')
    a = p.parse_args()
    coverage, readiness = analyze(a)
    print(json.dumps({'view_frames': coverage['completed_view_frames'],
        'new_physical_frames': coverage['new_in_this_GPU_run_physical_frames'],
        'recommend_SFT': readiness['recommend_start_Scale_SFT']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
