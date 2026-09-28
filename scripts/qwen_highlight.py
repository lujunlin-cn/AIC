"""Frozen Qwen-VL teacher -> competition JSONL inference.

Raw video -> uniform anchor grid -> Qwen temporal selection -> expand to
original frame indices -> existing spatial crop path -> validated JSONL.

Supports ``--mock`` for pipeline/format checks without loading weights; real
runs use a local HF snapshot and ``device_map`` across the authorized GPUs.
Resumable per video via ``--skip-existing`` shard files.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aic.contract import (ContractError, load_index, load_jsonl, write_submission,
                          center_crop)
from aic.qwen_teacher import (PROMPT_VERSION, QwenTeacher, build_anchors,
                              extract_anchor_frames, segments_to_frames)
from aic.video import frame_timeline, probe_video


def _record_path(record, video_root):
    raw = record.get("video_path", record.get("path", record.get("video")))
    path = Path(raw)
    if not path.is_absolute() and video_root is not None:
        path = Path(video_root) / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def run(index_path, output_path, *, teacher, video_root=None,
        sample_fps=1.0, spatial_mode="true_face_smooth",
        detector_path=None, stage="final", model_size_mb=None,
        image_size=448, shard_dir=None, report_path=None):
    records = load_jsonl(index_path)
    index = load_index(index_path)
    detector_bytes = Path(detector_path).stat().st_size if detector_path else 0
    rows, diagnostics = [], []
    shard_dir = Path(shard_dir) if shard_dir else None
    if shard_dir:
        shard_dir.mkdir(parents=True, exist_ok=True)

    for record in records:
        vid = record["video_id"]
        meta = index[vid]
        path = _record_path(record, video_root)
        info = probe_video(path)
        if (info.width, info.height, info.frame_count) != (meta.width, meta.height, meta.frame_count):
            raise ContractError(f"{vid}: index dims {meta.width}x{meta.height}/{meta.frame_count} != decoded {info.width}x{info.height}/{info.frame_count}")
        shard_file = shard_dir / f"{vid}.json" if shard_dir else None
        if shard_file and shard_file.is_file():
            rows.append(load_jsonl(shard_file)[0])
            continue

        t0 = time.time()
        timeline = frame_timeline(path)
        anchors = build_anchors(path, sample_fps)
        frames = extract_anchor_frames(path, anchors, size=image_size)
        segs, status, raw = teacher.select_segments(
            frames, anchors, record["targetRatioWH"], sample_fps, info.duration)
        selected = segments_to_frames(segs, anchors, timeline)

        # Spatial crop on every original frame (dense_v1), keep only selected.
        from aic.spatial_pipeline import SpatialPath
        from aic.video import _decoded
        spatial = SpatialPath(spatial_mode, record["targetRatioWH"], detector_path)
        selected_set = set(selected)
        crops = {}
        for stamp, frame in _decoded(path):
            crop, _ = spatial.step(frame.to_ndarray(format="rgb24"))
            if stamp.index in selected_set:
                crops[stamp.index] = crop
        center = center_crop(info.width, info.height, record["targetRatioWH"])
        preds = [{"frame": int(f), "bboxes": list(crops.get(int(f), center))}
                 for f in selected]
        row = {"video_id": vid, "targetRatioWH": list(record["targetRatioWH"]),
               "predictions": preds}
        if model_size_mb is not None:
            row["model_size_mb"] = float(model_size_mb)
        rows.append(row)
        diag = {"video_id": vid, "anchors": len(anchors), "selected": len(selected),
                "segments": len(segs), "status": status, "seconds": round(time.time() - t0, 2),
                "resets": spatial.resets, "face_detections": spatial.detections,
                "raw": raw}
        diagnostics.append(diag)
        if shard_file:
            from aic.contract import write_jsonl
            write_jsonl(shard_file, [row])
        print(json.dumps({k: v for k, v in diag.items() if k != "raw"}), flush=True)

    report = write_submission(output_path, rows, index, stage=stage,
                              actual_model_size_mb=model_size_mb)
    out = {"output": str(output_path), "rows": len(rows),
           "selected_predictions": sum(len(r["predictions"]) for r in rows),
           "model_size_mb": model_size_mb, "prompt_version": PROMPT_VERSION,
           "detector_weight_bytes": detector_bytes,
           "validation": report.to_dict(), "diagnostics": diagnostics}
    if report_path:
        Path(report_path).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--video-root")
    p.add_argument("--model", help="local HF snapshot dir")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--mock-segments", default="[]")
    p.add_argument("--sample-fps", type=float, default=1.0)
    p.add_argument("--image-size", type=int, default=448)
    p.add_argument("--spatial-mode", default="true_face_smooth")
    p.add_argument("--detector")
    p.add_argument("--stage", choices=["preliminary", "final"], default="final")
    p.add_argument("--model-size-mb", type=float)
    p.add_argument("--device-map", default="auto")
    p.add_argument("--attn", default="sdpa")
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    p.add_argument("--tensor-parallel-size", type=int, default=4)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-num-seqs", type=int, default=16)
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--shard-dir")
    p.add_argument("--report")
    args = p.parse_args(argv)

    teacher = QwenTeacher(model_path=args.model,
                          backend=("mock" if args.mock else args.backend),
                          device_map=args.device_map,
                          attn_implementation=args.attn,
                          max_new_tokens=args.max_new_tokens,
                          image_size=args.image_size,
                          block_size=args.block_size,
                          tensor_parallel_size=args.tensor_parallel_size,
                          gpu_memory_utilization=args.gpu_memory_utilization,
                          max_model_len=args.max_model_len,
                          max_num_seqs=args.max_num_seqs,
                          mock_segments=json.loads(args.mock_segments))
    try:
        result = run(args.index, args.output, teacher=teacher,
                     video_root=args.video_root, sample_fps=args.sample_fps,
                     spatial_mode=args.spatial_mode, detector_path=args.detector,
                     stage=args.stage, model_size_mb=args.model_size_mb,
                     image_size=args.image_size, shard_dir=args.shard_dir,
                     report_path=args.report)
        print(json.dumps({k: v for k, v in result.items() if k != "diagnostics"},
                         ensure_ascii=False, indent=2))
        return 0
    except (ContractError, OSError, ValueError) as e:
        print(json.dumps({"valid": False, "error": str(e)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
