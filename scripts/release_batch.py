#!/usr/bin/env python3
"""Run a frozen release candidate on raw videos without training data.

The competition index normally comes from the platform and must be preserved
verbatim.  For a new local evaluation batch this helper can instead build the
required metadata index directly from one or more video paths (ffprobe/actual
decoded frame count), then delegates to the exact ``python -m aic.release``
entrypoint.  No feature cache, labels, or training manifest is read.

Examples::

  # one or more new evaluation videos, output kept for inspection
  python scripts/release_batch.py --manifest releases/20260925_v1/manifest.json \
    --candidate A0_center --weights-dir artifacts/engineering_release_20260925/weights \
    --video /path/new_a.mp4 --video /path/new_b.mp4 --ratio 9 16 \
    --output /tmp/a0_batch.jsonl --report /tmp/a0_batch.report.json

  # dependency-free smoke input (requires ffmpeg, creates a temporary testsrc)
  python scripts/release_batch.py --smoke --manifest releases/20260925_v1/manifest.json \
    --candidate A0_center --weights-dir artifacts/engineering_release_20260925/weights
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

# Running ``python scripts/release_batch.py`` does not automatically put the
# repository root on sys.path (only ``scripts/`` is added).  Make the documented
# from-repo command work without requiring an editable package install.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _probe(path: Path) -> dict:
    from aic.video import probe_video
    info = probe_video(path)
    return {
        "video_id": path.stem,
        "video_path": str(path.resolve()),
        "width": int(info.width),
        "height": int(info.height),
        "frame_count": int(info.frame_count),
    }


def build_index(paths: Iterable[str | Path], ratio: tuple[float, float], output: Path) -> Path:
    rows = []
    seen: set[str] = set()
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        row = _probe(path)
        if row["video_id"] in seen:
            raise ValueError(f"duplicate video_id from filename stem: {row['video_id']!r}")
        seen.add(row["video_id"])
        row["targetRatioWH"] = [float(ratio[0]), float(ratio[1])]
        rows.append(row)
    if not rows:
        raise ValueError("at least one video is required")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                      for row in rows), encoding="utf-8")
    return output


def _smoke_video(path: Path) -> None:
    command = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
               "-i", "testsrc=size=128x72:rate=5:duration=1.2",
               "-an", "-c:v", "libx264", "-threads", "1",
               "-pix_fmt", "yuv420p", str(path)]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("--smoke requires ffmpeg on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg smoke generation failed with exit code {exc.returncode}") from exc


def run_release(args: argparse.Namespace, index: Path, output: Path, report: Path) -> int:
    # Call the same module entrypoint used by the documented release command;
    # this avoids a second implementation of weight verification/inference.
    from aic.release import main as release_main
    argv = ["--manifest", str(args.manifest), "--candidate", args.candidate,
            "--weights-dir", str(args.weights_dir), "--index", str(index),
            "--output", str(output), "--report", str(report),
            "--device", args.device, "--stage", args.stage]
    if args.video_root:
        argv.extend(["--video-root", str(args.video_root)])
    return int(release_main(argv) or 0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--index", type=Path, help="Existing official/enriched JSONL index")
    source.add_argument("--video", type=Path, nargs="+", help="Raw videos; metadata index is generated")
    source.add_argument("--smoke", action="store_true", help="Generate a temporary ffmpeg test video")
    parser.add_argument("--ratio", type=float, nargs=2, default=(9.0, 16.0), metavar=("W", "H"))
    parser.add_argument("--output", type=Path, help="Submission JSONL (required for --index/--video)")
    parser.add_argument("--report", type=Path, help="Evidence report JSON (required for --index/--video)")
    parser.add_argument("--keep-smoke-dir", type=Path,
                        help="Keep generated smoke index/video/output/report in this directory")
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stage", choices=("preliminary", "final"), default="preliminary")
    args = parser.parse_args(argv)

    if args.index is not None and (args.output is None or args.report is None):
        parser.error("--output and --report are required with --index")
    if args.video is not None and (args.output is None or args.report is None):
        parser.error("--output and --report are required with --video")
    if (not all(math.isfinite(value) for value in args.ratio)
            or args.ratio[0] <= 0 or args.ratio[1] <= 0):
        parser.error("--ratio values must be positive")

    if args.smoke:
        if args.keep_smoke_dir:
            work = args.keep_smoke_dir
            work.mkdir(parents=True, exist_ok=True)
            cleanup = False
        else:
            work_context = tempfile.TemporaryDirectory(prefix="aic_release_smoke_")
            work = Path(work_context.name)
            cleanup = True
        video = work / "smoke.mp4"
        index = work / "index.jsonl"
        output = args.output or work / "submission.jsonl"
        report = args.report or work / "report.json"
        _smoke_video(video)
        build_index([video], (args.ratio[0], args.ratio[1]), index)
        try:
            status = run_release(args, index, output, report)
            print(json.dumps({"smoke": True, "work_dir": str(work), "output": str(output),
                              "report": str(report), "status": status}, ensure_ascii=False))
            return status
        finally:
            if cleanup:
                work_context.cleanup()

    if args.index is not None:
        index = args.index
    else:
        # Place generated metadata next to the output, keeping all artifact
        # paths explicit in the resulting release report.
        index = args.output.with_suffix(args.output.suffix + ".index.jsonl")
        build_index(args.video, (args.ratio[0], args.ratio[1]), index)
    return run_release(args, index, args.output, args.report)


if __name__ == "__main__":
    raise SystemExit(main())
