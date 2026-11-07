#!/usr/bin/env python3
"""Build a read-only-source inventory for the 2026-11-07 research review.

Run from any directory. Only docs/experiment_inventory.md is written.
Historical registry events are retained in source order, including duplicates;
they are not treated as current experiment states or counted as unique trials.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUTES = {
    1: ("01_classic_visual_baselines.md", "轻量基线与传统特征"),
    2: ("02_video_encoders_and_temporal_supervision.md", "视频表征与时间监督"),
    3: ("03_qwen32b_teacher.md", "32B教师"),
    4: ("04_lfm_spatial_students.md", "LFM空间学生"),
    5: ("05_temporal_decision_and_qwen4b.md", "时间决策与4B后训练"),
    6: ("06_timelens_events_and_boundaries.md", "TimeLens事件边界"),
    7: ("07_spatial_grounding_and_composition.md", "空间构图与路径"),
    8: ("08_joint_spatiotemporal_utility.md", "时空联合效用"),
    9: ("09_new_representations_and_transfer.md", "新表征与迁移"),
    10: ("10_videoora_frozen_pipeline.md", "Video-ORA冻结系统"),
    11: ("11_videoora_temporal_posttraining.md", "Video-ORA时间后训练"),
    12: ("12_videoora_spatial_fullcrop.md", "Video-ORA空间完整框"),
    13: ("13_data_annotation_and_evaluation.md", "数据标注与评价工程"),
}


def classify(name: str, *, report: bool = False) -> int:
    """Primary topic only; route prose records cross-topic dependencies."""
    s = name.lower()
    if any(x in s for x in ("fullcrop", "fullb_", "geometry_head", "final_submission", "research_ab", "3point", "retrain_arm", "push_retrain", "push_b_dense", "p1_newa", "p2_newa", "p3_raft", "sp3e")):
        return 12
    if any(x in s for x in ("videoora_rl", "videoora_finetuned", "videoora_route_deep", "videoora_rank1", "rank1_42", "native_rank1", "tp1_shrink", "temporal_rl", "videoora_post_training", "videoora_next_gain", "videoora_supervisor")):
        return 11
    if any(x in s for x in ("jev", "valen", "dinov3", "aiku", "avimos")):
        return 9
    if "videoora" in s or "video_ora" in s:
        return 10
    if any(x in s for x in ("annotation", "dataset", "source_audit", "media_audit", "provenance", "intake", "duplicate", "engineering_release", "release_clean", "dense_e2e", "remote_env", "repository_audit", "submission_contract", "official_baseline")):
        return 13
    number = re.search(r"(?:^|[/_])r(\d{1,3})(?=[_/bcd.]|$)", s)
    if number:
        n = int(number.group(1))
        if n in (25, 87, 102, 104, 105):
            return 9
        if n == 24 and "aha" in s:
            return 9
        if n in (42, 85):
            return 13
        if n in (89, 94):
            return 7
        if n in (90, 91, 92, 95):
            return 1
        if n in (28, 46, 53, 69, 84, 86, 88, 98, 100, 101, 103):
            # R46/R53 have separate temporal and spatial report branches.
            if n in (46, 53) and any(x in s for x in ("highfreq", "local_high", "semantic_transition", "quality_conditioned")):
                return 2
            return 7
        if n in (47, 54, 55, 56, 57, 58, 59, 60, 65, 66, 93, 106, 107, 108):
            return 8 if "joint" in s else 6
        if n in (61, 62, 63, 64, 67, 68, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83):
            return 8
        if n in (30, 31, 43, 44, 45, 48, 49, 50, 51, 52, 96, 97, 99):
            return 2
        if n == 22:
            return 13
        if 1 <= n <= 41:
            return 5
    if any(x in s for x in ("qwen32", "qwen_point", "qwen_noface", "teacher", "crophead_v3", "temporal_v2", "dense_temp_v3", "temp_crophead", "t4_temp_boundary", "t5_crophead", "v4_", "v5_", "temp_h1", "max_window_qwen", "q2_qwen")):
        return 3
    if any(x in s for x in ("lfm", "v7_", "v8_", "v9_", "v6_", "e2_1b")):
        return 4
    if any(x in s for x in ("internvideo", "videomae", "qvh", "native_candidate", "native_foundation", "sub_e", "sub_f", "sub_g", "data_scale", "stage2", "foundation", "temporal_ablation")):
        return 2
    if any(x in s for x in ("spatial", "subject", "max_window", "obs_cache", "t5_")):
        return 7
    if any(x in s for x in ("a0_", "a1_", "a2_", "b0_", "bs0_", "rg_", "summe", "tsm", "lockbox", "proxy_baseline", "representation_generalization", "local_validation", "postprocess_a0", "pairwise_ranking")):
        return 1
    if any(x in s for x in ("keep", "v10", "v11", "vpb", "randprobe", "r15", "gpt6pro", "round", "real_score_anchor", "weight_sweep")):
        return 5
    return 13


def clean(value: object, limit: int = 170) -> str:
    if isinstance(value, (list, dict)):
        value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    text = str(value).replace("|", "／").replace("\n", " ").replace("\r", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def link(route: int) -> str:
    filename, title = ROUTES[route]
    return f"[{route:02d} {title}]({filename})"


def registry_rows() -> list[tuple[str, int, dict]]:
    rows = []
    for relative in ("experiments/registry.jsonl", "registry.jsonl"):
        for line_no, line in enumerate((ROOT / relative).read_text().splitlines(), 1):
            if line.strip():
                rows.append((relative, line_no, json.loads(line)))
    return rows


def main() -> None:
    rows = registry_rows()
    for filename, _ in ROUTES.values():
        assert (ROOT / "docs" / filename).is_file(), filename
    tracked = set(subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0"))
    all_paths = subprocess.check_output(
        ["rg", "--files", "-g", "*.md", "-g", "!docs/**", "-g", "!vendor/**", "-g", "!submissions/**", "-g", "!提交要求/**", "-g", "!repro_packages/**", "-g", "!temp/**", "-g", "!node_modules/**"],
        cwd=ROOT,
    ).decode().splitlines()
    # Plans and reports are indexed even when their proposed trials did not run.
    paths = sorted(p for p in all_paths if (ROOT / p).is_file())
    reports = [p for p in paths if p.startswith("reports/")]
    configs = sorted((ROOT / "configs").glob("*.json"))
    counts = Counter(classify(str(r.get("run_id", r.get("experiment_id", "")))) for _, _, r in rows)
    out = [
        "# 实验事件台账与证据文件清单",
        "",
        "生成日期：2026-11-07。总入口：[index.md](index.md)。由 `scripts/build_research_history_inventory.py` 从工作区原件生成；重建命令：`python scripts/build_research_history_inventory.py`。",
        "",
        "## 统计口径",
        "",
        f"- 两份registry共 **{len(rows)}条非空JSON事件**，来自experiments/registry.jsonl与根registry.jsonl；空行忽略，非空行全部解析成功。",
        "- 一条事件不等于一个独立实验：包含同一run的启动/结束/回分、重复登记、CV折和种子；不去重、不累计成所谓独立试验次数。",
        f"- 补充扫描 **{len(reports)}份reports Markdown**，连同根文档、old_docs、ext_data等共 **{len(paths)}份研究/状态Markdown**；另列 **{len(configs)}份JSON配置**。",
        "- 报告数量只统计本节所述目录和扩展名，未声称读取所有1434项reports产物或逐个重算全部checkpoint。JSON/日志关键结果由13份路线回顾与下列摘录补充。",
        "- 表中的“原事件状态”是当时原文，不是今天的最终状态。最终P2=42.74、rank-1=42.46、R91=33.87等以后续回分为准。",
        "- `当前Git已跟踪`表示本次生成时路径在git索引中；`工作区原件`表示未跟踪的新文件/迁移副本。后者路径是定位信息，未伪造GitHub可点证据链接；路线正文已保留讨论所需结果。",
        "- 分组只有一个主归属，跨路线依赖请读正文。整个台账为现存可追溯记录盘点，不保证恢复未落盘聊天、丢失远程实验或不存在的checkpoint。",
        "",
        "## 路线覆盖与registry事件数",
        "",
        "部分路线没有登记到这两份registry，但有实际报告；0条事件不表示没有做实验。",
        "",
        "| 主归属 | 原事件条数 | reports Markdown数 |",
        "|---|---:|---:|",
    ]
    report_counts = Counter(classify(p, report=True) for p in reports)
    for route in ROUTES:
        out.append(f"| {link(route)} | {counts[route]} | {report_counts[route]} |")
    out += ["", "## 逐条registry事件", "", "| 来源行 | 实验／事件ID | 主归属 | 原事件状态或结果 |", "|---|---|---|---|"]
    for source, no, row in rows:
        name = str(row.get("run_id", row.get("experiment_id", row.get("id", "未命名"))))
        state = row.get("status", row.get("result", row.get("decision", "原记录未设状态字段")))
        out.append(f"| `{source}:{no}` | `{clean(name, 130)}` | {link(classify(name))} | {clean(state)} |")
    out += [
        "", "## 两份registry不足以覆盖的末期结果", "",
        "这些项目由报告/交付JSON补足；不伪造它们在旧registry中的记录。", "",
        "| 实验／产物 | 最终可确认状态 | 主归属 |",
        "|---|---|---|",
        f"| T1/T2 candidate rerank | 实际执行，未形成留出增益，2026-10-22关闭；无新官方提升 | {link(11)} |",
        f"| native manual/mixed/phase SFT | 训练与fresh reload已做；小源留出不稳，详见路线11 | {link(11)} |",
        f"| merged temporal T1-base | 官方39.67，结构污染；未当纯时间消融 | {link(11)} |",
        f"| rank-1 temporal step50 | 已提交，官方42.46，覆盖早期待回分 | {link(11)} |",
        f"| P3与RAFT-A | P3仅171视频临时输出；RAFT两次启动失败无新权重；均撤销 | {link(12)} |",
        f"| 连续geometry head/联合LoRA | raw微正，真实40视频轨迹−.000620，未进入426 | {link(12)} |",
        f"| FULLCROP_B_V2 | raw .872317，但真实路径低于新B，未进入426 | {link(12)} |",
        f"| FULLCROP_B_MIX_V1 | 240步，3840消费/3456唯一样本；完整426包已交付；无公共质量分、无官方分 | {link(12)} |",
        f"| P2最终结果 | 42.74，426视频/133921帧/2空；旧42.73事件保留为过程 | {link(12)} |",
    ]
    out += ["", "## 关键原件hash与机器结果摘录", "", "下列字段直接从当前工作区JSON取值；是保存结果的摘录，不是本次重跑模型。", ""]
    evidence_fields = {
        "reports/geometry_head_20261024/gate_a.json": ["raw.full_minus_current.mean_delta", "final.full_minus_current.mean_delta", "eligible_for_426", "official_score"],
        "reports/geometry_head_20261024/chain_state.json": ["event", "official_candidate_ready", "reason"],
        "reports/fullcrop_b_v2/deploy_gate.json": ["raw_selection.macro.full300", "final.source_macro_iou.new_b", "final.source_macro_iou.full300", "eligible_for_426", "official_score"],
        "reports/fullcrop_b_mix_v1/training_config.json": ["steps", "batch", "consumed_samples", "unique_samples", "official_score"],
        "reports/fullcrop_b_mix_v1/selection.json": ["best", "selection_rule", "public_metric_evaluation_performed"],
        "reports/fullcrop_b_mix_v1/delivery_finish_state.json": ["time", "official_candidate_ready", "public_metric_evaluation", "predictions_sha256", "zip_sha256", "parameter_elements"],
    }
    for relative, fields in evidence_fields.items():
        path = ROOT / relative
        data = json.loads(path.read_text())
        selected = {}
        for field in fields:
            value = data
            for part in field.split("."):
                value = value[part]
            selected[field] = value
        out += [f"### `{relative}`", "", f"SHA256：`{hashlib.sha256(path.read_bytes()).hexdigest()}`", "", "```json", json.dumps(selected, ensure_ascii=False, indent=2), "```", ""]
    out += ["## 研究与报告文件逐项导航", "", "计划、调查、失败报告与最终报告全部保留标题；列入本表不表示计划已执行。", "", "| 文件 | 主归属 | 原件状态 | 首个标题 | SHA256前16位 |", "|---|---|---|---|---|"]
    for relative in paths:
        path = ROOT / relative
        content = path.read_text(errors="replace")
        title = next((s.lstrip("# ").strip() for s in content.splitlines() if s.startswith("#")), "无标题")
        status = "当前Git已跟踪" if relative in tracked else "工作区原件"
        out.append(f"| `{relative}` | {link(classify(relative, report=True))} | {status} | {clean(title, 90)} | `{hashlib.sha256(path.read_bytes()).hexdigest()[:16]}` |")
    out += ["", "## JSON配置清单", "", "配置存在不代表该实验完成；需结合上述事件、路线正文与日志。", "", "| 配置文件 | 主归属 | SHA256前16位 |", "|---|---|---|"]
    for path in configs:
        relative = path.relative_to(ROOT).as_posix()
        out.append(f"| `{relative}` | {link(classify(relative))} | `{hashlib.sha256(path.read_bytes()).hexdigest()[:16]}` |")
    out += ["", "## 复核边界", "", "此回顾未启动新训练、GPU推理或比赛上传，也未重新核验官方登录态/全部远程权重。2026-11-06跨主机审计是已保存证据；2026-11-07直接核对的是工作区文档、登记、JSON和产物身份记录。原始实验环境、启动命令、输入清单和checkpoint位置按各报告追溯。", ""]
    target = ROOT / "docs" / "experiment_inventory.md"
    target.write_text("\n".join(out))
    print(json.dumps({"registry_events": len(rows), "report_markdown": len(reports), "research_markdown": len(paths), "json_configs": len(configs), "route_events": dict(sorted(counts.items())), "output": str(target)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
