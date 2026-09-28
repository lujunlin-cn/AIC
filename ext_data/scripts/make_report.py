#!/usr/bin/env python3
"""Render the machine state of the external-data round into a Markdown status report.

Reads only ``_registry/{status,validation_summary,overlap_report}.json``, each
dataset's ``processed/dataset_card.json`` / ``logs/validation.json`` and the
YouTube queues, so it can be re-run after every ``refresh_all.sh``.

  python scripts/make_report.py --out reports/registry_status.md
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aicext.common import EXT_ROOT, REGISTRY_DIR, now, read_jsonl  # noqa: E402

READY_LABEL = {
    "raw_video_trainable": "原视频可训练",
    "frame_folder_trainable": "帧目录可训练",
    "image_trainable": "图像可训练",
    "feature_head_trainable": "仅特征头可训练",
    "labels_only": "仅标注（媒体未就绪）",
    "labels_ready_media_sample_only": "标注齐全，媒体仅样例",
    "sample_only": "仅小样本",
    "partial": "部分就绪",
    "blocked": "阻塞",
}


def load(p: Path, default=None):
    return json.loads(p.read_text()) if p.exists() else default


def gib(n) -> str:
    return f"{(n or 0) / 2**30:.1f} GiB"


def queue_stats(ds: str) -> dict | None:
    q = EXT_ROOT / ds / "logs" / "yt_queue.jsonl"
    if not q.exists():
        return None
    last = {}
    for r in read_jsonl(q):
        last[r["yt_id"]] = r
    return dict(collections.Counter(v["status"] for v in last.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    st = load(REGISTRY_DIR / "status.json", {})
    val = load(REGISTRY_DIR / "validation_summary.json", {}).get("results", {})
    ov = load(REGISTRY_DIR / "overlap_report.json", {})
    L = [f"# 外部数据注册表状态（自动生成 {now()}）", "",
         f"数据根目录：`{EXT_ROOT}`；schema 错误 {st.get('schema_error_count', '?')} 条。", "",
         "| 数据集 | 就绪状态 | 媒体（就绪/总数） | 标注条数 | aic_split | 校验 | 占用 |", "|---|---|---|---|---|---|---|"]
    for ds, s in sorted(st.get("datasets", {}).items()):
        du = sum(f.stat().st_size for f in (EXT_ROOT / ds).rglob("*") if f.is_file())
        L.append(f"| {ds} | {READY_LABEL.get(s.get('readiness'), s.get('readiness'))} | {s['media_ready']}/{s['media']} | "
                 f"{s['annotations']} | {json.dumps(s.get('aic_split'), ensure_ascii=False)} | {val.get(ds, '-')} | {gib(du)} |")
    L += ["", "## 各数据集要点", ""]
    for ds, s in sorted(st.get("datasets", {}).items()):
        card = load(EXT_ROOT / ds / "processed" / "dataset_card.json", {})
        v = load(EXT_ROOT / ds / "logs" / "validation.json", {})
        L.append(f"### {ds}")
        L.append(f"- 版本：{card.get('release')}")
        L.append(f"- 来源：{card.get('source')}")
        L.append(f"- 可训练类型：{', '.join(card.get('trainable_as') or [])}")
        if card.get("not_trainable_as"):
            L.append(f"- 不可用于：{card['not_trainable_as']}")
        if card.get("blockers"):
            L.append(f"- 阻塞/缺口：{'；'.join(card['blockers'])}")
        L.append(f"- 下载状态：{json.dumps(s.get('download_status'), ensure_ascii=False)}；官方划分："
                 f"{json.dumps(s.get('official_split'), ensure_ascii=False)}")
        L.append(f"- 标注类型：{json.dumps(s.get('annotation_types'), ensure_ascii=False)}")
        if s.get("anomalies"):
            L.append(f"- 异常：{json.dumps(s['anomalies'], ensure_ascii=False)}（明细 `{ds}/processed/anomalies.jsonl`）")
        q = queue_stats(ds)
        if q:
            L.append(f"- YouTube 队列：{json.dumps(q, ensure_ascii=False)}（失败清单 `{ds}/logs/yt_failures.json` / "
                     f"`yt_queue.jsonl`）")
        if v.get("loader"):
            L.append(f"- DataLoader 批次形状：{json.dumps(v['loader'].get('shapes', [])[:1], ensure_ascii=False)}")
        L.append("")
    L += ["## 重叠与泄漏", "",
          f"- 跨数据集同源组：{json.dumps({k: v['groups'] for k, v in ov.get('cross_dataset_group_overlap', {}).items()}, ensure_ascii=False)}",
          f"- AIC 官方测试集字节级命中：{ov.get('aic_official_test_byte_identical_hits')}（{ov.get('aic_official_test_note')}）",
          f"- 官方 train 被移出 train：{json.dumps(ov.get('official_train_items_moved_out_of_train'), ensure_ascii=False)}",
          f"- 官方评测项因同组被划入 train：{json.dumps(ov.get('official_eval_items_assigned_train'), ensure_ascii=False)}",
          f"- 组内 split 冲突：{len(ov.get('groups_with_conflicting_split', []))}",
          f"- 划分策略：{json.dumps(ov.get('split_policy'), ensure_ascii=False)}", ""]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(L) + "\n")
    print(a.out)


if __name__ == "__main__":
    main()
