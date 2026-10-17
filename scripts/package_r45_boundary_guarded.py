#!/usr/bin/env python3
"""Build R45 by restricting the R44 perturbation to the R26 window edges.

R45 is a deterministic decision intervention, not a parameter sweep.  R26 is
the parent set; only additions/removals that lie within a fixed 5 percent band
around either edge of that parent window are retained from R44.  All other
frames revert to R26.  Spatial boxes are copied from R26 byte-for-byte.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import zipfile
from pathlib import Path


KEEP = 0.85
BOUNDARY_FRAC = 0.05
MODEL_MB = 450.283377
MODEL_B = 0.450283377


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def choose_local_swaps(base: list[int], candidate: list[int], full_n: int) -> tuple[list[int], dict]:
    """Return R26 plus only R44 swaps in the two 5% edge bands."""
    base_set, cand_set = set(base), set(candidate)
    additions = sorted(cand_set - base_set)
    removals = sorted(base_set - cand_set)
    if len(additions) != len(removals):
        raise ValueError("R44 set difference is unbalanced")
    if not additions:
        return list(base), {"additions": 0, "retained_swaps": 0, "reverted_swaps": 0}
    left, right = min(base), max(base)
    band = max(1, int(math.ceil(BOUNDARY_FRAC * full_n)))
    add_ok = [f for f in additions if left - band <= f < left or right < f <= right + band]
    rem_ok = [f for f in removals if left <= f < left + band or right - band < f <= right]
    count = min(len(add_ok), len(rem_ok))

    # Select the closest permitted frames to the existing parent boundaries.
    # This gives a deterministic minimum-displacement intervention if the two
    # sides have unequal counts, while retaining all R44 swaps when balanced.
    def edge_distance(f: int) -> tuple[int, int]:
        return (min(abs(f - left), abs(f - right)), f)

    add_keep = sorted(add_ok, key=edge_distance)[:count]
    rem_keep = sorted(rem_ok, key=edge_distance)[:count]
    out = sorted((base_set - set(rem_keep)) | set(add_keep))
    if len(out) != len(base):
        raise AssertionError("boundary guard changed budget")
    return out, {
        "additions": len(additions),
        "retained_swaps": count,
        "reverted_swaps": len(additions) - count,
        "boundary_frames": band,
        "left_edge": left,
        "right_edge": right,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r26", type=Path, required=True)
    ap.add_argument("--r44", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--r44-manifest", type=Path, required=True)
    args = ap.parse_args()

    r26_rows = read_jsonl(args.r26)
    r44_rows = read_jsonl(args.r44)
    r26 = {str(row["video_id"]): row for row in r26_rows}
    r44 = {str(row["video_id"]): row for row in r44_rows}
    if set(r26) != set(r44) or len(r26) != 426:
        raise ValueError("expected identical 426-video coverage")
    r44_manifest = json.loads(args.r44_manifest.read_text(encoding="utf-8"))
    n_full_by_vid = {str(item["video_id"]): int(item["n_full"]) for item in r44_manifest["per_video"]}

    out_rows: list[dict] = []
    details: list[dict] = []
    total = 0
    r44_changed = 0
    retained = 0
    for vid in sorted(r26, key=lambda x: int(x) if x.isdigit() else x):
        base_row, cand_row = r26[vid], r44[vid]
        base_preds = {int(p["frame"]): p for p in base_row["predictions"]}
        cand_preds = {int(p["frame"]): p for p in cand_row["predictions"]}
        cand_frames = [int(p["frame"]) for p in cand_row["predictions"]]
        base_frames = sorted(base_preds)
        full_n = n_full_by_vid[vid]
        expected = int(round(KEEP * full_n))
        if len(base_frames) != expected or len(cand_frames) != expected:
            raise ValueError(f"{vid}: exact budget mismatch")
        chosen, d = choose_local_swaps(base_frames, cand_frames, full_n)
        if chosen != base_frames:
            retained += d["retained_swaps"]
        if cand_frames != base_frames:
            r44_changed += 1
        # Common frames are taken from R26; newly admitted edge frames use
        # R44's B3 box.  R44 and R26 boxes are checked equal on their common
        # frame IDs below, so the spatial function is unchanged.
        for f in sorted(set(base_frames) & set(cand_frames)):
            if base_preds[f].get("bboxes") != cand_preds[f].get("bboxes"):
                raise ValueError(f"{vid}: R44 changed B3 box on common frame {f}")
        preds = [base_preds[f] if f in base_preds else cand_preds[f] for f in chosen]
        # R26 rows are the authoritative spatial parent and metadata source.
        row = dict(base_row)
        row["predictions"] = preds
        row["model_size_mb"] = MODEL_MB
        row["model_params_b"] = MODEL_B
        out_rows.append(row)
        total += len(preds)
        details.append({"video_id": vid, "n_full": full_n, "n_keep": len(preds),
                        "r44_changed": cand_frames != base_frames,
                        "r45_changed_vs_r26": chosen != base_frames,
                        **d})

    args.output.mkdir(parents=True, exist_ok=False)
    pred_path = args.output / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    zip_path = args.output / "R45_BOUNDARY_GUARDED_RESIDUAL_B3.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(pred_path, "predictions.jsonl")
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zip_path) as z:
            if z.namelist() != ["predictions.jsonl"]:
                raise ValueError("unexpected ZIP members")
            z.extractall(td)
        if sha256(Path(td) / "predictions.jsonl") != sha256(pred_path):
            raise ValueError("ZIP roundtrip changed predictions")
    manifest = {
        "submission_id": "R45_BOUNDARY_GUARDED_RESIDUAL_B3",
        "status": "PACKAGED_NOT_UPLOADED",
        "uploaded": False,
        "official_platform_score": None,
        "parent_submission": "R26_POS_CENTER_EXACT085_B3",
        "candidate_source": "R44_HUMAN_CALIBRATED_RESIDUAL_ETA020_B3",
        "parent_official_score": 36.04,
        "mechanism": "fixed 5 percent edge-band guard around R26 center window; retain only local R44 swaps",
        "changed_factor": "temporal frame ranking only; R26 B3 spatial boxes copied byte-for-byte",
        "keep_rule": "round(0.85 * full original frame count)",
        "boundary_fraction": BOUNDARY_FRAC,
        "video_count": len(out_rows),
        "prediction_count": total,
        "r44_changed_videos": r44_changed,
        "r45_changed_videos": sum(int(d["r45_changed_vs_r26"]) for d in details),
        "r44_changed_frame_positions": sum(int(d["additions"]) for d in details),
        "retained_swaps": retained,
        "total_parameters": 450283377,
        "model_params_b": MODEL_B,
        "model_size_mb": MODEL_MB,
        "predictions_sha256": sha256(pred_path),
        "zip_sha256": sha256(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "spatial_boxes_source": "R26_POS_CENTER_EXACT085_B3",
        "per_video": details,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (args.output / (zip_path.name + ".sha256")).write_text(sha256(zip_path) + "  " + zip_path.name + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "per_video"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
