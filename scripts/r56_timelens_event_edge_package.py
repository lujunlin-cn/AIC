#!/usr/bin/env python3
"""Package a TimeLens2 event-boundary residual candidate.

The R54 mask residual treats all frames inside an interval alike except for
an accidental stable-rank frame-order ramp.  R56 uses a predeclared event
boundary prior: frames near either start or end of a model interval are more
likely to contain an action onset or visible outcome than its static middle.
The centre-85 policy remains the anchor and no official labels are read.
"""
from __future__ import annotations
import argparse, hashlib, json, math, re, tempfile, zipfile
from pathlib import Path
import numpy as np

TIME_LENS_PARAMS = 8_767_123_696
V9_PARAMS = 87_407_729
ETA = 0.20
SPAN_RE = re.compile(r"\[\s*\[.*?\]\s*\]")

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

def parse(text: str) -> list[tuple[float, float]]:
    m = SPAN_RE.search(text or "")
    if not m: return []
    try: arr = json.loads(m.group(0))
    except Exception: return []
    out = []
    for x in arr:
        if len(x) < 2: continue
        a, b = float(x[0]), float(x[1])
        if math.isfinite(a) and math.isfinite(b) and b >= a: out.append((a, b))
    return out

def rank01(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort").astype(float) / max(len(x) - 1, 1)

def edge_evidence(times: np.ndarray, intervals: list[tuple[float, float]]) -> np.ndarray:
    """q=2|normalized interval phase-.5| inside each interval, max over spans."""
    out = np.zeros(len(times), dtype=float)
    for a, b in intervals:
        inside = (times >= a - 1e-6) & (times <= b + 1e-6)
        phase = np.clip((times - a) / max(b - a, 1e-6), 0.0, 1.0)
        q = np.where(inside, 2.0 * np.abs(phase - 0.5), 0.0)
        out = np.maximum(out, q)
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", type=Path, required=True)
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    response_rows = json.loads(args.responses.read_text()).get("results", [])
    spans = {str(x["video_id"]): parse(x.get("response", "")) for x in response_rows}
    intake = {str(x["video_id"]): x for x in rows(args.index)}
    parent = rows(args.parent)
    if set(spans) != set(intake): raise ValueError("response/index coverage mismatch")
    out, details, total = [], [], 0
    key = lambda x: int(str(x["video_id"])) if str(x["video_id"]).isdigit() else str(x["video_id"])
    for row in sorted(parent, key=key):
        vid = str(row["video_id"]); meta = intake[vid]; allp = row["predictions"]
        n = int(meta["frame_count"])
        if len(allp) != n: raise ValueError(f"{vid}: parent coverage {len(allp)} != {n}")
        fps = float(meta.get("fps", 0.0))
        if fps <= 0: raise ValueError(f"{vid}: missing positive fps")
        frame = np.asarray([int(p["frame"]) for p in allp], dtype=np.int64); times = frame / fps
        intervals = spans.get(vid, []); edge = edge_evidence(times, intervals)
        centre = 1.0 - np.abs(times / max(float(times[-1]), 1e-9) - 0.5) * 2.0
        score = rank01(centre) + ETA * (rank01(edge) - 0.5)
        k = int(round(0.85 * n)); chosen = np.sort(frame[np.argsort(-score, kind="mergesort")[:k]])
        source = {int(p["frame"]): p for p in allp}; q = dict(row)
        q["predictions"] = [source[int(f)] for f in chosen]
        q["model_size_mb"] = (TIME_LENS_PARAMS + V9_PARAMS) / 1e6
        q["model_params_b"] = (TIME_LENS_PARAMS + V9_PARAMS) / 1e9
        total += len(q["predictions"]); out.append(q)
        details.append({"video_id": vid, "n_full": n, "n_keep": k, "intervals": intervals,
                        "edge_nonzero": int(np.count_nonzero(edge)),
                        "selected_edge": int(sum(edge[np.isin(frame, chosen)] > 0))})
    args.output.mkdir(parents=True, exist_ok=False); pred = args.output / "predictions.jsonl"
    with pred.open("w") as f:
        for q in out: f.write(json.dumps(q, ensure_ascii=False, separators=(",", ":")) + "\n")
    zip_path = args.output / (args.output.name + ".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z: z.write(pred, "predictions.jsonl")
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zip_path) as z: z.extractall(td)
        if sha(Path(td) / "predictions.jsonl") != sha(pred): raise RuntimeError("ZIP SHA mismatch")
    total_params = TIME_LENS_PARAMS + V9_PARAMS
    manifest = {"submission_id": args.output.name, "status": "PACKAGED_NOT_UPLOADED",
        "official_platform_score": None, "parent_submission": "R26_POS_CENTER_EXACT085_B3",
        "mechanism": "TimeLens2 interval boundary evidence q=2*abs(normalized_phase-.5), max over spans, fixed eta=.20 residual around exact centre-85 rank",
        "spatial_mode": "v9_b3_artifact", "spatial_note": "V9 vision-only B3 boxes copied from audited full-frame artifact",
        "eligibility": "PARAMETER_COMPLIANT_IF_V9_ARTIFACT_ACCEPTED",
        "parameter_components": [{"name": "TimeLens2-8B", "parameters": TIME_LENS_PARAMS},
                                 {"name": "V9 vision-only SigLIP2 + B3 head + YuNet", "parameters": V9_PARAMS}],
        "total_parameters": total_params, "model_params_b": total_params / 1e9, "eta": ETA,
        "video_count": len(out), "prediction_count": total, "prediction_sha256": sha(pred),
        "zip_sha256": sha(zip_path), "zip_bytes": zip_path.stat().st_size,
        "responses_sha256": sha(args.responses), "parent_sha256": sha(args.parent),
        "index_sha256": sha(args.index), "per_video": details}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (args.output / (zip_path.name + ".sha256")).write_text(sha(zip_path) + "  " + zip_path.name + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "per_video"}, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
