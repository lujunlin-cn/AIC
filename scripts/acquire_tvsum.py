#!/usr/bin/env python3
"""Acquire/verify TVSum and emit a provenance-first manifest.

The script never fabricates competition labels. TVSum's ``user_anno`` contains
2-second-shot *summary importance* scores (20 raters); these are exported as
``annotation_type=summary_importance`` only. No spatial/crop ground truth is
created. The archive is downloaded separately (usually with the companion
range downloader) and may be resumed safely.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, shutil, subprocess, tarfile, urllib.request, zipfile
from pathlib import Path
from typing import Any
import numpy as np
from aic.data import ManifestRecord, grouped_splits, probe_video, sha256_file, write_manifest

ARCHIVE_URL = "https://people.csail.mit.edu/yalesong/tvsum/tvsum50_ver_1_1.tgz"
README_URL = "https://raw.githubusercontent.com/yalesong/tvsum/master/README.md"
LICENSE_URL = "https://creativecommons.org/licenses/by/3.0/"
# SHA-256 of WebscopeReadMe.txt shipped in the pinned archive.
LICENSE_TEXT_HASH = "341fb9cd2b4f276ae6d497ea5f600b596698627bad1db54af1944171ce1e7eba"
# Keep the archive version and proxy-label version explicit.  A new label
# protocol writes new filenames instead of silently reusing older NPZs.
DATASET_VERSION = "tvsum50_v1.1_2019-11-06_proxy-v2"
LABEL_PROTOCOL = "tvsum_summary_mean_norm_frame_expand_uniform_v1"
BINARY_TARGET_PROTOCOL = "tvsum_summary_mean_norm_ge_0.5_v1"


def _mat_records(path: Path) -> list[dict[str, Any]]:
    """Read ydata-tvsum50.mat (v5 or v7.3 when h5py is available)."""
    try:
        import scipy.io
        mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
        arr = np.atleast_1d(mat["tvsum50"])
        out=[]
        for x in arr:
            out.append({k: getattr(x,k) for k in x._fieldnames})
        return out
    except NotImplementedError as e:
        try:
            import h5py
        except ImportError as ie:
            raise RuntimeError("TVSum MAT is MATLAB v7.3; install h5py in an isolated env to parse it") from ie
        # MATLAB v7.3 stores each field as a (50,1) dataset of HDF5 refs.
        with h5py.File(path, "r") as h:
            root = h["tvsum50"]
            def resolve(ref):
                a = h[ref][()]
                if a.dtype.kind in {"u", "i"} and a.size and a.ndim >= 1 and a.size < 1000:
                    # MATLAB char arrays are uint16 code points, stored as a
                    # column for IDs/categories/titles.
                    vals = a.reshape(-1)
                    try: return "".join(chr(int(v)) for v in vals).rstrip("\x00")
                    except Exception: pass
                return np.asarray(a)
            fields = {}
            for name in root:
                refs = root[name][()]
                fields[name] = [resolve(refs.reshape(-1)[i]) for i in range(refs.size)]
            out = []
            for i in range(50):
                out.append({k: v[i] for k,v in fields.items()})
            return out


def _scalar(x: Any) -> Any:
    while isinstance(x, np.ndarray) and x.size == 1: x=x.reshape(-1)[0]
    if isinstance(x, bytes): return x.decode(errors="replace")
    return x.item() if isinstance(x, np.generic) else x


def _video_id(raw: Any) -> str:
    s=str(_scalar(raw)); return s.strip()


def audit_annotation_record(raw: dict[str, Any], video_id: str) -> dict[str, Any]:
    """Return a machine-readable audit of one TVSum MAT record."""
    try: nframes = int(_scalar(raw.get("nframes")))
    except Exception: nframes = None
    try:
        fps = float(_scalar(raw.get("fps", raw.get("frame_rate"))))
    except Exception: fps = None
    try: duration = float(_scalar(raw.get("duration")))
    except Exception: duration = None
    original = np.asarray(raw.get("user_anno")); squeezed = np.squeeze(original)
    shape = list(squeezed.shape)
    if squeezed.ndim == 1: squeezed = squeezed[:, None]
    orientation = "rows_are_shots"
    if squeezed.ndim == 2 and squeezed.shape[0] == 20 and squeezed.shape[1] != 20:
        orientation = "transposed_20_raters"
        rows, raters = int(squeezed.shape[1]), int(squeezed.shape[0])
    elif squeezed.ndim == 2:
        rows, raters = int(squeezed.shape[0]), int(squeezed.shape[1])
    else:
        rows, raters = None, None
    return {"video_id": video_id, "nframes": nframes, "fps": fps,
            "duration_seconds": duration, "user_anno_raw_shape": list(original.shape),
            "user_anno_squeezed_shape": shape, "annotation_rows": rows,
            "annotator_columns": raters, "orientation": orientation,
            "row_semantics": "TVSum summary-importance shot/segment score; not AIC frame GT",
            "nominal_segment_seconds": 2.0,
            "observed_uniform_segment_seconds": (duration / rows if duration and rows else None),
            "rows_equal_nframes": bool(rows is not None and nframes is not None and rows == nframes),
            "alignment_convention": "uniform_edges_linspace_0_nframes_rows_plus_1",
            "label_protocol": LABEL_PROTOCOL,
            "binary_target_protocol": BINARY_TARGET_PROTOCOL}


def _load_info_urls(root: Path) -> dict[str, str]:
    """Read the author's info TSV, preserving original YouTube URLs."""
    for p in root.rglob("ydata-tvsum50-info.tsv"):
        out = {}
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            cols = line.split("\t")
            if len(cols) >= 4 and cols[1].strip() and cols[3].strip(): out[cols[1].strip()] = cols[3].strip()
        if out: return out
    return {}


def _extract_safe(archive: Path, root: Path) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True); out=[]
    with tarfile.open(archive, "r:gz") as tf:
        base=root.resolve()
        for m in tf.getmembers():
            target=(root/m.name).resolve()
            if not str(target).startswith(str(base)+os.sep): raise RuntimeError(f"unsafe archive member: {m.name}")
        tf.extractall(root)
    # The author package intentionally wraps MATLAB/video payloads in ZIPs.
    # Extract those too, with the same traversal check, so one command is
    # sufficient and the raw archive remains available for hash verification.
    for zpath in sorted(root.rglob("*.zip")):
        dest=zpath.parent / zpath.stem
        dest.mkdir(parents=True, exist_ok=True); base=dest.resolve()
        with zipfile.ZipFile(zpath) as zf:
            for name in zf.namelist():
                target=(dest/name).resolve()
                if not str(target).startswith(str(base)+os.sep): raise RuntimeError(f"unsafe ZIP member: {name}")
            zf.extractall(dest)
    return list(root.rglob("*"))


def _download_archive(archive: Path) -> None:
    """Resume the author archive through curl; leave a .part on interruption."""
    archive.parent.mkdir(parents=True, exist_ok=True)
    part = archive.with_suffix(archive.suffix + ".part")
    cmd = ["curl", "-fL", "--retry", "5", "--connect-timeout", "30", "--max-time", "7200", "-C", "-", ARCHIVE_URL, "-o", str(part)]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise RuntimeError("curl is required for resumable TVSum acquisition") from e
    os.replace(part, archive)



def _export_video_labels(raw: dict[str, Any], out_dir: Path, video_id: str) -> Path | None:
    """Expand TVSum 2-second rater scores to a frame-aligned *proxy* array.

    The expansion uses evenly spaced shot boundaries because TVSum's public MAT
    stores shot scores rather than per-frame crop/highlight labels. The NPZ
    metadata explicitly names this as summary importance and keeps all raters.
    """
    try:
        out_dir.mkdir(parents=True, exist_ok=True); path = out_dir / f"{video_id}.proxy-v2.npz"
        if path.exists(): return path
        nframes = int(_scalar(raw.get("nframes")))
        anno = np.asarray(raw.get("user_anno"), dtype=np.float32)
        anno = np.squeeze(anno)
        if anno.ndim == 1: anno = anno[:, None]
        if anno.ndim != 2 or nframes <= 0: return None
        # TVSum convention is shots x 20; accept the transposed representation.
        if anno.shape[0] == 20 and anno.shape[1] != 20: anno = anno.T
        if anno.shape[0] > nframes and anno.shape[1] <= nframes: anno = anno.T
        nshots = anno.shape[0]
        edges = np.linspace(0, nframes, nshots + 1, dtype=np.int64)
        scores = np.empty((nframes, anno.shape[1]), dtype=np.float32)
        for i in range(nshots): scores[edges[i]:edges[i+1]] = anno[i]
        labels = (scores.mean(axis=1) - 1.0) / 4.0
        labels = np.clip(labels, 0.0, 1.0)
        np.savez_compressed(path, scores=scores, labels=labels,
                            mask=np.ones(nframes,dtype=np.bool_),
                            frame_indices=np.arange(nframes,dtype=np.int64),
                            video_id=np.asarray(video_id),
                            annotation_type=np.asarray("summary_importance_2s"),
                            label_protocol=np.asarray(LABEL_PROTOCOL),
                            binary_target_protocol=np.asarray(BINARY_TARGET_PROTOCOL),
                            source_annotation_rows=np.asarray(nshots, dtype=np.int64),
                            source_nframes=np.asarray(nframes, dtype=np.int64))
        return path
    except Exception:
        return None

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--archive", type=Path, required=True); ap.add_argument("--output-root", type=Path, required=True); ap.add_argument("--manifest", type=Path, required=True); ap.add_argument("--extract", action="store_true"); ap.add_argument("--download", action="store_true", help="download/resume the author archive before extraction"); ap.add_argument("--labels-root", type=Path, default=None, help="where to write frame-aligned summary proxy NPZs"); ap.add_argument("--audit-output", type=Path, default=None, help="machine-readable JSONL annotation audit"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--split-seed", default="aic-tvsum-v1")
    args=ap.parse_args(); archive=args.archive; root=args.output_root
    if args.download: _download_archive(archive)
    if not archive.exists(): raise SystemExit(f"archive missing: {archive}; pass --download to fetch it")
    if args.extract: _extract_safe(archive, root)
    mat_candidates=list(root.rglob("ydata-tvsum50.mat")) + list(root.parent.rglob("ydata-tvsum50.mat"))
    if not mat_candidates: raise SystemExit("could not find ydata-tvsum50.mat after extraction")
    mat=mat_candidates[0]; raw=_mat_records(mat)
    video_urls = _load_info_urls(root)
    if args.limit: raw=raw[:args.limit]
    # Archive paths vary slightly between source releases. Match by basename/id.
    videos=[]
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".mp4", ".webm", ".mkv", ".avi", ".mov"}: videos.append(p)
    by_name={p.stem.lower():p for p in videos}
    groups=[f"tvsum:{_video_id(x.get('video'))}" for x in raw]
    splits=grouped_splits(groups, seed=args.split_seed)
    archive_hash=sha256_file(archive)
    labels_root = args.labels_root or (root / "labels")
    recs=[]; missing=[]; audits=[]
    for x in raw:
        vid=_video_id(x.get("video")); group=f"tvsum:{vid}"; p=by_name.get(vid.lower())
        audits.append(audit_annotation_record(x, vid))
        # Some package names include numeric prefix; use URL/video ID substring fallback.
        if p is None:
            candidates=[q for q in videos if vid.lower() in q.name.lower()]
            p=candidates[0] if len(candidates)==1 else None
        meta={}; status="verified" if p else "missing"
        if p:
            try: meta=probe_video(p)
            except Exception as e: status="failed"; missing.append(f"{vid}: ffprobe {e}")
        else: missing.append(vid)
        label_path = _export_video_labels(x, labels_root, vid)
        recs.append(ManifestRecord(dataset="TVSum",version=DATASET_VERSION,video_id=vid,source_id=vid,source_group=group,path=str(p.resolve()) if p else None,source_url=video_urls.get(vid),license="CC BY 3.0 claim in dataset README; Webscope DSA gate unresolved",license_url=LICENSE_URL,license_text_hash=LICENSE_TEXT_HASH,license_gate="blocked",download_status=status,sha256=sha256_file(p) if p else None,split=splits[group],annotation_type="summary_importance_2s",annotation_path=str(label_path.resolve()) if label_path else str(mat.resolve()),**meta,notes="TVSum summary proxy only; not competition highlight/crop GT. Package source="+ARCHIVE_URL+"; WebscopeReadMe requires signed Yahoo DSA, approved non-commercial academic use, and forbids redistribution. archive_sha256="+archive_hash+". label_protocol="+LABEL_PROTOCOL))
    write_manifest(recs,args.manifest)
    audit_output = args.audit_output or args.manifest.with_name("tvsum_annotation_audit.jsonl")
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in audits), encoding="utf-8")
    summary={"dataset":"TVSum","version":DATASET_VERSION,"archive":str(archive),"archive_sha256":archive_hash,"archive_url":ARCHIVE_URL,"readme_url":README_URL,"license":"CC BY 3.0","records":len(recs),"verified_videos":sum(r.download_status=="verified" for r in recs),"missing_or_failed":missing,"annotation_type":"summary_importance_2s","label_protocol":LABEL_PROTOCOL,"binary_target_protocol":BINARY_TARGET_PROTOCOL,"annotation_audit":str(audit_output),"spatial_ground_truth":False}
    args.manifest.with_suffix(".summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
