#!/usr/bin/env python3
"""Package an aggressive TimeLens2 temporal ensemble around the V9 center parent.

The deployed frame set is the exact R26 centre-85 set with V9 spatial boxes.
For each frame we fuse two independent views of the same TimeLens2 response:
interval membership (event coverage) and distance to an interval boundary
(onset/outcome evidence).  Both are residuals around the within-window centre
rank.  Coefficients are frozen at eta_inside=.20 and eta_edge=.50 from the
source-held-out annotation proxy; there is no official-label fitting here.
"""
from __future__ import annotations
import argparse, hashlib, json, math, re, tempfile, zipfile
from pathlib import Path
import numpy as np

TIME_LENS_PARAMS = 8_767_123_696
V9_PARAMS = 87_407_729
ETA_INSIDE = 0.20
ETA_EDGE = 0.50
SPAN_RE = re.compile(r"\[\s*\[.*?\]\s*\]")

def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def rows(p: Path):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

def parse(text: str):
    m = SPAN_RE.search(text or "")
    if not m: return []
    try: arr = json.loads(m.group(0))
    except Exception: return []
    out = []
    for z in arr:
        if len(z) < 2: continue
        a, b = float(z[0]), float(z[1])
        if math.isfinite(a) and math.isfinite(b) and b >= a: out.append((a, b))
    return out

def rank01(x: np.ndarray) -> np.ndarray:
    if len(x) <= 1: return np.zeros(len(x), dtype=float)
    return np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort").astype(float) / (len(x) - 1)

def inside_edge(times: np.ndarray, spans):
    inside = np.zeros(len(times), dtype=float)
    edge = np.zeros(len(times), dtype=float)
    for a, b in spans:
        mask = (times >= a - 1e-6) & (times <= b + 1e-6)
        inside[mask] = 1.0
        phase = np.clip((times - a) / max(b - a, 1e-6), 0.0, 1.0)
        q = np.where(mask, 2.0 * np.abs(phase - 0.5), 0.0)
        edge = np.maximum(edge, q)
    return inside, edge

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", type=Path, required=True)
    ap.add_argument("--parent", type=Path, required=True,
                    help="V9 exact centre-85 predictions (all selected frames)")
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    response = json.loads(args.responses.read_text())
    spans = {str(x["video_id"]): parse(x.get("response", ""))
             for x in response.get("results", [])}
    intake = {str(x["video_id"]): x for x in rows(args.index)}
    parent = rows(args.parent)
    if set(spans) != set(intake): raise ValueError("response/index coverage mismatch")
    out, details, total = [], [], 0
    key = lambda x: int(str(x["video_id"])) if str(x["video_id"]).isdigit() else str(x["video_id"])
    for row in sorted(parent, key=key):
        vid = str(row["video_id"]); meta = intake[vid]; preds = row["predictions"]
        n_full = int(meta["frame_count"]); fps = float(meta["fps"])
        n_keep = int(round(0.85 * n_full))
        if len(preds) != n_full: raise ValueError(f"{vid}: parent must contain all {n_full} frames, got {len(preds)}")
        frame = np.asarray([int(p["frame"]) for p in preds], dtype=np.int64)
        times = frame / max(fps, 1e-9)
        inside, edge = inside_edge(times, spans.get(vid, []))
        # The centre rank is calculated on the parent window exactly as in
        # R54.  Stable mergesort makes ties reproducible.
        centre = 1.0 - np.abs(times / max(float(times[-1]), 1e-9) - 0.5) * 2.0
        score = (rank01(centre) + ETA_INSIDE * (rank01(inside) - 0.5)
                 + ETA_EDGE * (rank01(edge) - 0.5))
        chosen = np.argsort(-score, kind="mergesort")[:n_keep]
        selected = np.sort(frame[chosen])
        source = {int(p["frame"]): p for p in preds}
        q = dict(row); q["predictions"] = [source[int(f)] for f in selected]
        # Keep the per-row legacy field consistent with the actual inference
        # stack (TimeLens2 plus V9 spatial artifact), rather than inheriting
        # the full-frame artifact's historical placeholder value.
        q["model_size_mb"] = (TIME_LENS_PARAMS + V9_PARAMS) / 1e6
        q["model_params_b"] = (TIME_LENS_PARAMS + V9_PARAMS) / 1e9
        total += len(selected); out.append(q)
        details.append({"video_id": vid, "n_full": n_full, "n_keep": n_keep,
                        "intervals": spans.get(vid, []),
                        "inside_parent": int(np.count_nonzero(inside)),
                        "edge_parent": int(np.count_nonzero(edge)),
                        "selected_inside": int(np.count_nonzero(inside[np.isin(frame, selected)])),
                        "selected_edge": int(np.count_nonzero(edge[np.isin(frame, selected)] > 0)),
                        "changed_vs_center": int(len(set(frame.tolist()) ^ set(selected.tolist())) // 2)})
    args.output.mkdir(parents=True, exist_ok=False)
    pred = args.output / "predictions.jsonl"
    pred.write_text("".join(json.dumps(q, ensure_ascii=False, separators=(",", ":")) + "\n" for q in out))
    zpath = args.output / (args.output.name + ".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z: z.write(pred, "predictions.jsonl")
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zpath) as z: z.extractall(td)
        if sha(Path(td) / "predictions.jsonl") != sha(pred): raise RuntimeError("ZIP round trip mismatch")
    total_params = TIME_LENS_PARAMS + V9_PARAMS
    manifest = {
        "submission_id": args.output.name,
        "status": "PACKAGED_NOT_UPLOADED",
        "official_platform_score": None,
        "parent_submission": "R47_V9_B3_CENTER85",
        "mechanism": "TimeLens2 interval-membership + interval-boundary multi-view residual around exact centre-85 rank",
        "eta_inside": ETA_INSIDE, "eta_edge": ETA_EDGE,
        "spatial_mode": "v9_b3_artifact",
        "eligibility": "PARAMETER_COMPLIANT_IF_V9_ARTIFACT_ACCEPTED",
        "parameter_components": [{"name": "TimeLens2-8B", "parameters": TIME_LENS_PARAMS},
                                 {"name": "V9 vision-only SigLIP2 + B3 head + YuNet", "parameters": V9_PARAMS}],
        "total_parameters": total_params, "model_params_b": total_params / 1e9,
        "video_count": len(out), "prediction_count": total,
        "prediction_sha256": sha(pred), "zip_sha256": sha(zpath), "zip_bytes": zpath.stat().st_size,
        "responses_sha256": sha(args.responses), "parent_sha256": sha(args.parent),
        "index_sha256": sha(args.index), "per_video": details,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (args.output / (zpath.name + ".sha256")).write_text(sha(zpath) + "  " + zpath.name + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "per_video"}, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
