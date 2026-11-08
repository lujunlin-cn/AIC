"""Create evidence-backed P2 policy report; optional official-feedback import.

This command does not submit anything or calculate a proxy official score.
Feedback must bind each result to its exact ZIP and original evaluation index.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from aic.scale_policy import POLICIES, atomic_json, budget_ledger, sha
from scripts.build_scale_policy_candidates import load_protocol


def official_results(protocol, manifest, feedback=None, feedback_root=None):
    scores={name:None for name in POLICIES}; evidence={}
    if feedback is not None:
        if (feedback['stage']!='semifinal' or feedback['metric']!='official_f1_percent' or
                feedback['index_sha256']!=protocol['input_hashes']['inputs/index.jsonl']):
            raise ValueError('official feedback stage/metric/index contract differs')
        if not feedback['scores'] or set(feedback['scores'])-set(POLICIES):
            raise ValueError('unknown or empty official feedback candidate')
        for name,item in feedback['scores'].items():
            score=item['official_f1']
            if type(score) not in (int,float) or not math.isfinite(score) or not 0<=score<=100:
                raise ValueError('invalid official score')
            if item['zip_sha256']!=manifest['candidates'][name]['zip_sha256']:
                raise ValueError('official score belongs to a different ZIP')
            source=(Path(feedback_root)/item['evidence_file']).resolve()
            if not source.is_relative_to(Path(feedback_root).resolve()) or sha(source)!=item['evidence_sha256']:
                raise ValueError('missing/changed official platform result evidence')
            scores[name]=float(score);evidence[name]={'file':str(source),'sha256':sha(source),
                'source_kind':'owner_provided_official_platform_receipt','independently_authenticated_by_agent':False}
    baseline=scores['C0'];eligible=[k for k in ['C1','C2','C3'] if baseline is not None and scores[k] is not None and scores[k]>baseline]
    winner=max(eligible,key=lambda k:scores[k]) if eligible else None
    return {'parent':'P2','metric':'official_f1_percent','new_official_evaluation_executed_by_agent':False,
        'by_candidate':{k:{'official_f1':v,'absolute_delta_from_C0':v-baseline if v is not None and baseline is not None else None,
                           'zip_sha256':manifest['candidates'][k]['zip_sha256'],'evidence':evidence.get(k),
                           'promotion_eligible':k in eligible} for k,v in scores.items()},
        'best_improving_measured_candidate':winner,'all_four_official_results_available':all(s is not None for s in scores.values()),
        'owner_reported_historical_P2_score':protocol['owner_reported_baseline_score'],
        'historical_score_substituted_for_new_C0_evaluation':False,
        'historical_score_conflict':protocol['historical_score_conflict'],
        'unmeasured_reason':'no authenticated official submission adapter or confirmed semifinal multi-submission quota' if feedback is None else 'partial official receipts if any candidate remains null',
        'official_labels_accessed':False,'proxy_IoU_converted_to_official_F1':False}


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--official-feedback',type=Path)
    a=p.parse_args();root=a.root.resolve();protocol=load_protocol(root)
    manifest=json.loads((root/'candidate_manifest.json').read_text())
    done=json.loads((root/'generation_complete.json').read_text())
    for name,digest in done['final_artifact_hashes'].items():
        if sha(root/name)!=digest:raise ValueError('final generation artifact changed: '+name)
    independent=json.loads((root/'independent_audit.json').read_text())
    geometry=json.loads((root/'geometry_audit.json').read_text())
    routes=json.loads((root/'route_coverage.json').read_text())
    temporal=json.loads((root/'temporal_identity_check.json').read_text())
    qp=json.loads((root/'qp_scale_audit.json').read_text())
    if independent['protocol_sha256']!=protocol['protocol_sha256']:raise ValueError('audit protocol changed')
    for k in POLICIES:
        if (not independent[k]['passed'] or independent[k]['max_position_drift_px']>1 or
            independent['zip_hashes'][k]!=manifest['candidates'][k]['zip_sha256'] or
            temporal['by_candidate'][k]['identity_rate']!=1 or
            independent[k]['by_route']['native']['changed_frames']!=0):
            raise ValueError('geometry/temporal gate failed before official result import')
    feedback=json.loads(a.official_feedback.read_text()) if a.official_feedback else None
    results=official_results(protocol,manifest,feedback,a.official_feedback.parent if a.official_feedback else None)
    previous=root/'official_results.json'
    if previous.exists():
        old=json.loads(previous.read_text())
        if any(r['official_f1'] is not None for r in old['by_candidate'].values()) and feedback is None:
            raise ValueError('do not replace measured official results with null')
        if feedback is not None and old!=results:
            archive=root/'official_results_history'/('results_'+sha(previous)+'.json')
            if not archive.exists():atomic_json(archive,old)
    atomic_json(previous,results)
    budget=budget_ledger(root/'gpu_budget_ledger.json')
    media=json.loads((root/'media_validation.json').read_text()) if (root/'media_validation.json').exists() else None
    if media and (not media['passed'] or media['protocol_sha256']!=protocol['protocol_sha256']):raise ValueError('media gate failed')
    winner=results['best_improving_measured_candidate']
    status='OFFICIAL_WINNER_AVAILABLE' if winner else 'AUDITED_PACKAGES_READY_OFFICIAL_EVALUATION_PENDING'
    atomic_json(root/'delivery_status.json',{'status':status,'parent':'P2','official_winner':winner,
        'geometry_gate_passed':True,'temporal_gate_passed':True,'media_validation_passed':bool(media),
        'GPU_seconds':budget['cumulative_GPU_seconds'],'GPU_cap_seconds':budget['cap_GPU_seconds'],
        'training_started':False,'best_existing_submission_overwritten':False})
    lines=['# P2 Route-specific Scale Policy Optimization',
        '',f"更新时间：{datetime.now(timezone.utc).isoformat()}。唯一父链 **P2**。状态：`{status}`。",'',
        '## 结果与候选', '',
        '|候选|A|B|Native|视频数|输出帧数|本轮官方 F1|相对新 C0|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for k,s in POLICIES.items():
        m=manifest['candidates'][k];r=results['by_candidate'][k]
        score='null' if r['official_f1'] is None else str(r['official_f1'])
        delta='null' if r['absolute_delta_from_C0'] is None else str(r['absolute_delta_from_C0'])
        lines.append(f"|{k}|{s['a']:.2f}|{s['b']:.2f}|{s['native']:.2f}|{m['videos']}|{m['output_frames']}|{score}|{delta}|")
    lines += ['', '原 P2 ZIP 与 C0 字节相同，SHA256 仍为 `'+manifest['candidates']['C0']['zip_sha256']+'`。',
        '本轮任务给定 P2 历史成绩为 44.63，旧记录为 44.65；二者保留来源冲突。它们没有被冒充为本轮新 C0 回分。',
        '常数选择依据来自此前 P1 公开 LIVE 诊断；本轮全部坐标、时间、路由来自 P2。公开空间 IoU 不换算为官方 F1。',
        '', '## 八项结论与证据', '',
        f"1. **最高官方 F1 策略**：{winner if winner else 'null，尚无四候选本轮官方回分，未晋级任何新候选'}。见 `official_results.json`、`official_interface_check.json`。",
        f"2. **A/B 改动与贡献**：A={routes['videos']['a']} 视频/{routes['output_frames']['a']} 帧；B={routes['videos']['b']} 视频/{routes['output_frames']['b']} 帧。C1/C2 同时修改 A/B；C3 只修改 A。官方分数贡献为 null，不能由框变动量推算。见 `route_coverage.json`。",
        f"3. **Native**：{routes['videos']['native']} 视频/{routes['output_frames']['native']} 帧，四候选逐框不变。见 `independent_audit.json`。",
        '4. **QP**：'+ '; '.join(f"{k} shot审计={qp['by_candidate'][k]['QP_shot_audits']}、失败={qp['by_candidate'][k]['QP_failed_audits']}、宽度变化={qp['by_candidate'][k]['QP_width_changed_frames']}、最大QP中心变化={independent[k]['max_QP_center_change_px']} px" for k in ['C1','C2','C3'])+'。额外固定位置等式是已登记的后处理修改，不能描述为完全未改 QP。见 `qp_scale_audit.json`。',
        '5. **Position Drift**：'+'; '.join(f"{k} 最大={g['maximum_position_drift_px']} px、非法窗口={g['invalid_window_triggers']}、回退输出帧={g['fallback_output_frames']}" for k,g in geometry['by_candidate'].items())+'。来自整数取整的偏差满足 ≤1 px。未 clamp、未修改 P2 Adapter。见 `geometry_audit.json` 与 ZIP 独立复算。',
        '6. **时间集合**：890/890 视频的 frame 顺序/成员/数量 100% 与 P2 一致；每包 263,970 帧，3 个空视频保持原样，新增/删除均为 0。见 `temporal_identity_check.json`。',
        '7. **完整复现**：890/890 原 P2 CPU 部署重放逐框相同；协议、输入、源码、每源可恢复收据和 ZIP 均已 SHA256 锁定。C0 字节级复用，其他 ZIP 固定时间戳。媒体复核：'+(f"{media['videos']}/890 原视频全解码，{media['decoded_frames']} 原始帧，SHA256/尺寸/帧数/PTS 全相同。" if media else 'null，完整媒体核验尚未完成。'),
        '8. **是否保留为最终提交**：四包保留为可评测候选；只有真实官方 F1 严格高于新 C0 才允许晋级。目前不覆盖原最佳提交。',
        '', '## 部署几何与稳定性', '',
        '从最终 P2 `[x,y,w]` 计算实际中心；新宽度为 `floor(w*Scale+1e-9)`，高度仅由目标画幅推导，两个轴分别就近取整。禁止 Scale 后再调用会恢复最大宽度的原整数导出器。',
        '先重放原完整事件的 P2 平滑与 QP（max_speed=0.15、P2 qp_floor=0.6），再缩放并执行登记了 `lo=hi=scaled_baseline_position` 的原 QP；最后只恢复 P2 原 target_frames。任何非法窗口按原镜头回退，QP 失败按原源回退，均保存原因。',
        'C1-frame 与 C1-segment 在原镜头/事件边界内分别计算；每路线固定一个动作，所以镜头内中位数仍为同一常数。两者 JSONL SHA256 均为 `'+qp['stability_controls']['frame_sha256']+'`，与 C1 相同。没有独立收益，不额外提交 segment 变体。',
        '**B 风险**：0.95/0.97 仅有转置 LIVE 的历史证据，没有原生竖屏转横屏公开域验证；本轮也未用隐藏标签验证。',
        '', '## 预算、合规和复现入口', '',
        f"本轮 GPU 实际消耗 {budget['cumulative_GPU_seconds']} 秒 / {budget['cap_GPU_seconds']} 秒；无模型推理、无 SFT/DPO、无权重修改。独立共享预算台账 `gpu_budget_ledger.json` 不随重启重置。",
        '完整推理参数仍为 8,999,771,458（含原保守 buffers）；本轮增加 0 参数。P2 原 base/A/B 身份沿用已核验记录，不把 ZIP 身份当作权重身份。',
        '', '```bash',
        'PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/build_scale_policy_candidates.py --root artifacts/scale_policy_optimization --workers 8',
        'PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/audit_scale_policy_candidates.py --root artifacts/scale_policy_optimization',
        '# 仅远端 CPU 全媒体核验（所有包共用一次解码）',
        'PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/audit_scale_policy_candidates.py --root artifacts/scale_policy_optimization --video-root /root/autodl-tmp/半决赛/video --workers 8',
        'PYTHONPATH=. python scripts/finalize_scale_policy_optimization.py --root artifacts/scale_policy_optimization',
        '```', '',
        '输入需保留 `inputs/`，官方媒体、原预测、缓存和 ZIP 只保留在本地/授权远端，不上传公开 GitHub。`protocol.json` 含输入摘要；`inputs/cache_manifest.json` 含逐缓存 SHA；`work/*/attempt_*` 含每帧四阶段中心/宽度/框与回退原因。',
        'CPU 验证证据见 `validation_cpu.json` 和测试日志；原路径源码保存在 `inputs/runtime/` 并锁定 SHA。',
        '', '## 剩余阻塞与下一步', '',
        '公开网页可访问，但本地/远端未配置可复用的已登录官方提交适配器；历史公开排行榜查询不是官方评测接口，半决赛多次提交额度未确认。已请求负责人提供既有登录入口/脚本或手工回传结果。',
        '在真实官方回分出现前，所有新成绩、A/B 分数贡献、获胜策略保持 null。保留四包，建议按 C0→C1→C2→C3 进行同集对照；不运行更多实验，不启动训练。',
        '可用 `--official-feedback` 导入带同一半决赛索引、对应 ZIP SHA256 和平台结果文件 SHA256 的反馈；不接受其他 ZIP 或代理 IoU。原结果会归档，不会被 null 重置。',
        '', '## 提交包 SHA256', '']
    for k,m in manifest['candidates'].items():lines += [f"- {k}: `{m['zip_sha256']}`；`candidates/{k}/{Path(m['zip']).name}`"]
    (root/'final_report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':status,'official_winner':winner,'GPU_seconds':budget['cumulative_GPU_seconds']}))


if __name__=='__main__':main()
