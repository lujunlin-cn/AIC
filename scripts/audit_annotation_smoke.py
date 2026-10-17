#!/usr/bin/env python3
"""Review a read-only Label Studio snapshot without changing human labels.

The snapshot has tasks, annotations and drafts in separate arrays. Raw
exports and contact sheets stay in ignored artifacts/. This script only
copies source IDs, choices, geometry and audit findings into its report.
No label agreement or annotation accuracy can be inferred from one rater.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time
from urllib.parse import parse_qs, urlsplit
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    started = time.perf_counter()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", type=Path, required=True)
    ap.add_argument("--label-config", type=Path, required=True)
    ap.add_argument("--qc-config", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--document-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--sheets", type=Path, required=True)
    ap.add_argument("--manual-review", type=Path,
                    help="Optional, separately recorded visual-review findings; never overwrites labels")
    args = ap.parse_args()
    raw = json.loads(args.export.read_text())
    qc = json.loads(args.qc_config.read_text())
    manifest = json.loads(args.manifest.read_text())
    sources = {v["video_id"]: v for v in manifest["items"]}
    schema = ET.parse(args.label_config).getroot()
    controls = {e.attrib["name"]: e for e in schema.iter() if "name" in e.attrib}
    allowed = {name: {c.attrib["value"] for c in e.findall("Choice")}
               for name, e in controls.items() if e.tag == "Choices"}
    tasks = {t["id"]: t for t in raw["tasks"]}
    if len(tasks) != len(raw["tasks"]):
        raise ValueError("Duplicate task IDs in snapshot; refusing to overwrite tasks")
    annotations = defaultdict(list)
    cancelled = []
    for a in raw["annotations"]:
        if a["was_cancelled"]:
            cancelled.append(a["id"])
        else:
            annotations[a["task_id"]].append(a)
    args.output.mkdir(parents=True, exist_ok=True)
    args.sheets.mkdir(parents=True, exist_ok=True)
    issues, rows = [], []
    media_hashes, seen_keys = {}, set()

    def issue(level: str, code: str, task_id: int, **detail) -> None:
        issues.append(dict(level=level, code=code, task_id=task_id, **detail))

    orphan_annotations = sorted(set(annotations) - set(tasks))
    if orphan_annotations:
        raise ValueError(f"Annotations reference unknown tasks: {orphan_annotations}")

    def local_media(value: str) -> Path:
        rel = parse_qs(urlsplit(value).query)["d"][0]
        path = (args.document_root / rel).resolve()
        path.relative_to(args.document_root.resolve())
        return path

    for tid, task in sorted(tasks.items()):
        d = task["data"]
        key = (d["video_id"], float(d["time_sec"]))
        if key in seen_keys:
            issue("error", "duplicate_source_time", tid)
        seen_keys.add(key)
        source = sources[d["video_id"]]
        expected_source_time = source["clip_start_seconds"] + d["time_sec"]
        if not math.isclose(d["source_time_sec"], expected_source_time, abs_tol=1e-6):
            issue("error", "source_time_mapping", tid)
        if d["source_offset_sec"] != source["clip_start_seconds"]:
            issue("error", "source_offset_mapping", tid)
        if not 0 <= d["time_sec"] < manifest["clip_duration_seconds"]:
            issue("error", "time_outside_window", tid)
        if any(k in d for k in ("qvh_query", "category", "annotation_reason")):
            issue("error", "blind_metadata_exposed", tid)
        image_path = local_media(d["image"])
        video_path = local_media(d["context_video"])
        for media in (image_path, video_path):
            rel = str(media.relative_to(args.document_root))
            if rel not in media_hashes:
                media_hashes[rel] = digest(media)
        with Image.open(image_path) as im:
            image_width, image_height = im.size
        if len(annotations[tid]) != 1:
            issue("error", "completed_annotation_count", tid, count=len(annotations[tid]))
        for a in annotations[tid]:
            choice_results, boxes, notes = defaultdict(list), [], []
            for r in a["result"]:
                name = r["from_name"]
                if name not in controls or r["to_name"] != "image":
                    issue("error", "unknown_control_or_target", tid, control=name)
                if r["type"] == "choices":
                    choice_results[name].append(r["value"]["choices"])
                    values = r["value"]["choices"]
                    if name not in allowed or not set(values) <= allowed[name]:
                        issue("error", "illegal_choice", tid, control=name)
                    if controls.get(name) is not None and controls[name].attrib.get("choice") == "single-radio" and len(values) != 1:
                        issue("error", "single_choice_cardinality", tid, control=name)
                elif r["type"] == "rectanglelabels":
                    b = r["value"]
                    x, y, w, h = [float(b[k]) for k in ("x", "y", "width", "height")]
                    valid = (all(math.isfinite(v) for v in (x, y, w, h)) and
                             x >= -1e-6 and y >= -1e-6 and w > 0 and h > 0 and
                             x+w <= 100+1e-6 and y+h <= 100+1e-6)
                    if not valid:
                        issue("error", "invalid_box_coordinates", tid, region_id=r["id"])
                    if (r["original_width"], r["original_height"]) != (image_width, image_height):
                        issue("error", "box_image_dimensions", tid)
                    if b.get("rotation", 0) != 0 or r.get("image_rotation", 0) != 0:
                        issue("error", "unsupported_rotation", tid)
                    wp, hp = w*image_width/100, h*image_height/100
                    boxes.append(dict(region_id=r["id"], x=x, y=y, width=w, height=h,
                                      width_px=wp, height_px=hp,
                                      labels=b["rectanglelabels"]))
                    if min(wp, hp) < qc["near_degenerate_min_side_pixels"]:
                        issue("repair", "near_degenerate_box", tid, region_id=r["id"], width_px=wp, height_px=hp)
                elif r["type"] == "textarea":
                    notes.extend(r["value"]["text"])
            choices = {k: vals[0][0] if len(vals[0]) == 1 else vals[0]
                       for k, vals in choice_results.items()}
            for name in qc["required_choices"]:
                if name not in choice_results:
                    issue("error", "missing_core_field", tid, control=name)
            for name, vals in choice_results.items():
                if len(vals) != 1:
                    issue("error", "duplicate_choice_result", tid, control=name)
            grade = choices.get("generic_highlight")
            positive = grade in qc["positive_grades"]
            if positive and "needs_context" not in choices:
                issue("repair", "positive_missing_context", tid)
            if grade == "不确定" and choices.get("confidence") != "低":
                issue("repair", "uncertain_without_low_confidence", tid)
            if len(boxes) > 1:
                issue("repair", "multiple_crop_windows", tid, count=len(boxes))
            if boxes and not positive:
                issue("repair", "box_on_nonhighlight", tid)
            if positive and not boxes and choices.get("composition_confidence") != "低或无合理框":
                issue("incomplete_diagnostic", "unrecorded_crop_feasibility", tid)
            if boxes and "composition_confidence" not in choices:
                issue("incomplete_diagnostic", "box_missing_composition_confidence", tid)
            if boxes and "crop_reason" not in choices:
                issue("incomplete_diagnostic", "box_missing_crop_reason", tid)
            rows.append(dict(task_id=tid, annotation_id=a["id"], video_id=d["video_id"],
                             source_group=source["source_group"],
                             time_sec=d["time_sec"], source_time_sec=d["source_time_sec"],
                             choices=choices, boxes=boxes, notes=notes,
                             image=str(image_path.relative_to(args.document_root))))

    groups = defaultdict(list)
    for row in rows:
        groups[row["video_id"]].append(row)
    summaries = {}
    for vid, group in groups.items():
        positives = [r for r in group if r["choices"].get("generic_highlight") in qc["positive_grades"]]
        summaries[vid] = dict(
            frames=len(group), seconds=sorted(r["time_sec"] for r in group),
            distributions={name: dict(Counter(r["choices"].get(name, "MISSING") for r in group))
                           for name in qc["required_choices"] + ["needs_context"]},
            highlight_frames=len(positives),
            highlight_context=dict(Counter(r["choices"].get("needs_context", "MISSING") for r in positives)),
            box_count=sum(len(r["boxes"]) for r in group),
            frames_with_box=sum(bool(r["boxes"]) for r in group))

    formal_by_id = {a["id"]: a for a in raw["annotations"]}
    draft_conflicts = []
    for d in raw["drafts"]:
        a = formal_by_id.get(d.get("annotation_id"))
        if a and a["result"] != d["result"]:
            draft_conflicts.append(dict(task_id=d["task_id"], draft_id=d["id"], annotation_id=a["id"], action="Keep formal result; never merge stale draft automatically"))
    report = dict(
        purpose=qc["purpose"], protocol_version=qc["protocol_version"],
        provenance=dict(export_sha256=digest(args.export), label_config_sha256=digest(args.label_config),
                        qc_config_sha256=digest(args.qc_config), manifest_sha256=digest(args.manifest),
                        script_sha256=digest(Path(__file__)),
                        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.document_root, text=True).strip(),
                        python=platform.python_version(), command=__import__("sys").argv,
                        exported_at_utc=raw["exported_at_utc"], export_method=raw["method"]),
        tasks=len(tasks), completed_annotations=len(rows), cancelled_annotation_ids=cancelled,
        annotator_count=len({a["completed_by_id"] for a in raw["annotations"] if not a["was_cancelled"]}),
        source_count=len({r["source_group"] for r in rows}), video_window_count=len(groups),
        draft_count=len(raw["drafts"]), draft_conflicts=draft_conflicts,
        core_and_mapping_errors=sum(i["level"] == "error" for i in issues),
        issue_counts=dict(Counter(i["code"] for i in issues)), issues=issues,
        per_source=summaries, frames=rows, media_sha256=media_hashes,
        scope="Single-rater development QC; not annotation accuracy, independent confirmation, or native crop GT",
        labels_mutated=False, training_started=False)
    with (args.output / "frame_labels.csv").open("w", newline="") as f:
        keys = ["task_id", "video_id", "time_sec", "source_time_sec"] + list(allowed) + ["box_count"]
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        for r in rows:
            writer.writerow({**{k: r[k] for k in keys[:4]},
                             **{k: r["choices"].get(k, "") for k in allowed}, "box_count": len(r["boxes"])})

    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 16)
    for vid, group in groups.items():
        canvas = Image.new("RGB", (1800, 245*math.ceil(len(group)/5)), "#f4f4f4")
        draw = ImageDraw.Draw(canvas)
        for i, row in enumerate(group):
            with Image.open(args.document_root / row["image"]) as original:
                im = original.convert("RGB").resize((356, 200))
            frame_draw = ImageDraw.Draw(im)
            for b in row["boxes"]:
                frame_draw.rectangle((b["x"]*3.56, b["y"]*2,
                                      (b["x"]+b["width"])*3.56, (b["y"]+b["height"])*2), outline="red", width=2)
            x, y = i%5*360, i//5*245
            canvas.paste(im, (x, y))
            c = row["choices"]
            draw.text((x+2, y+201), f'#{row["task_id"]} {row["time_sec"]:g}s {c["generic_highlight"]} {c["event_phase"]}', font=font, fill="black")
            draw.text((x+2, y+222), c.get("needs_context", ""), font=font, fill="black")
        canvas.save(args.sheets / f"{vid}_overview.jpg", quality=92)

    # Keep recommendations separate from formal annotations. Multiple issue
    # codes for one task produce a single row; none is an automatic label edit.
    advice = {
        "multiple_crop_windows": "只保留一个能表达事件价值的完整窗口；不要分别框对象，也不要机械取并集",
        "near_degenerate_box": "删除近零面积的误操作框，保留并检查正常窗口",
        "box_on_nonhighlight": "若0/1等级成立则删除框；不要为保框抬高等级",
        "box_missing_crop_reason": "补充画框原因",
        "box_missing_composition_confidence": "补充构图置信度",
        "unrecorded_crop_feasibility": "能画则画一个窗口；确无稳定窗口时记录低或无合理框",
        "positive_missing_context": "按真实价值证据填写信息依赖，不自动补为当前帧足够",
        "uncertain_without_low_confidence": "复核不确定等级与置信度是否一致",
    }
    queue = {}
    for finding in issues:
        tid = finding["task_id"]
        item = queue.setdefault(tid, dict(codes=[], evidence=[], actions=[], priority="repair"))
        item["codes"].append(finding["code"])
        item["evidence"].append("automated_contract_check")
        item["actions"].append(advice.get(finding["code"], "检查数据合同；不自动改写标签"))
    if args.manual_review:
        manual = json.loads(args.manual_review.read_text())
        if manual["export_sha256"] != report["provenance"]["export_sha256"]:
            raise ValueError("Manual review references a different annotation snapshot")
        report["provenance"]["manual_review_sha256"] = digest(args.manual_review)
        for finding in manual["findings"]:
            tid = finding["task_id"]
            if tid not in tasks:
                raise ValueError(f"Manual review references unknown task {tid}")
            item = queue.setdefault(tid, dict(codes=[], evidence=[], actions=[], priority=finding["priority"]))
            item["codes"].append(finding["code"])
            item["evidence"].append("visual_review_suggestion_not_ground_truth")
            item["actions"].append(finding["action"])
    with (args.output / "repair_queue.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "video_id", "time_sec", "priority", "issue_codes", "evidence", "suggested_action", "task_url"], lineterminator="\n")
        writer.writeheader()
        for tid, finding in sorted(queue.items()):
            data = tasks[tid]["data"]
            writer.writerow(dict(task_id=tid, video_id=data["video_id"], time_sec=data["time_sec"],
                                 priority=finding["priority"], issue_codes=";".join(dict.fromkeys(finding["codes"])),
                                 evidence=";".join(dict.fromkeys(finding["evidence"])),
                                 suggested_action="；".join(dict.fromkeys(finding["actions"])),
                                 task_url=f"http://47.108.220.174:3002/projects/1/data?task={tid}"))
    report["automatic_flagged_task_count"] = len({i["task_id"] for i in issues})
    report["review_queue_task_count"] = len(queue)
    report["resources"] = dict(device="CPU", gpu_used=False, wall_seconds=time.perf_counter()-started)
    (args.output / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k: report[k] for k in ("tasks", "completed_annotations", "source_count", "annotator_count", "core_and_mapping_errors", "issue_counts")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
