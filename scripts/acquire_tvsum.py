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
DATASET_VERSION = "tvsum50_v1.1_2019-11-06"


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
        # MATLAB v7.3 stores each struct field as object references. Resolve
        # refs recursively and normalize byte/string arrays.
        with h5py.File(path, "r") as h:
            root = h["tvsum50"]
            def deref(v):
                if hasattr(v, "shape") and v.dtype == h5py.ref_dtype:
                    return np.array([deref(h[r]) for r in v.flat], dtype=object).reshape(v.shape)
                if hasattr(v, "shape"):
                    a=v[()]
                    if isinstance(a, bytes): return a.decode(errors="replace")
                    if np.ndim(a)==0 and isinstance(a.item(), bytes): return a.item().decode(errors="replace")
                    return a
                return v
            names=list(root.dtype.names or [])
            # Typical orientation is 1x50 references; flatten each field and zip.
            fields={n:deref(root[n]) for n in names}
            n=max(np.asarray(v,dtype=object).size for v in fields.values())
            out=[]
            for i in range(n):
                d={}
                for k,v in fields.items():
                    a=np.asarray(v,dtype=object).reshape(-1); z=a[i]
                    while isinstance(z,np.ndarray) and z.size==1: z=z.reshape(-1)[0]
                    d[k]=z
                out.append(d)
            return out


def _scalar(x: Any) -> Any:
    while isinstance(x, np.ndarray) and x.size == 1: x=x.reshape(-1)[0]
    if isinstance(x, bytes): return x.decode(errors="replace")
    return x.item() if isinstance(x, np.generic) else x


def _video_id(raw: Any) -> str:
    s=str(_scalar(raw)); return s.strip()


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
        out_dir.mkdir(parents=True, exist_ok=True); path = out_dir / f"{video_id}.npz"
        np.savez_compressed(path, scores=scores, labels=labels, mask=np.ones(nframes,dtype=np.bool_), frame_indices=np.arange(nframes,dtype=np.int64), video_id=np.asarray(video_id), annotation_type=np.asarray("summary_importance_2s"))
        return path
    except Exception:
        return None

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--archive", type=Path, required=True); ap.add_argument("--output-root", type=Path, required=True); ap.add_argument("--manifest", type=Path, required=True); ap.add_argument("--extract", action="store_true"); ap.add_argument("--download", action="store_true", help="download/resume the author archive before extraction"); ap.add_argument("--labels-root", type=Path, default=None, help="where to write frame-aligned summary proxy NPZs"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--split-seed", default="aic-tvsum-v1")
    args=ap.parse_args(); archive=args.archive; root=args.output_root
    if args.download: _download_archive(archive)
    if not archive.exists(): raise SystemExit(f"archive missing: {archive}; pass --download to fetch it")
    if args.extract: _extract_safe(archive, root)
    mat_candidates=list(root.rglob("ydata-tvsum50.mat")) + list(root.parent.rglob("ydata-tvsum50.mat"))
    if not mat_candidates: raise SystemExit("could not find ydata-tvsum50.mat after extraction")
    mat=mat_candidates[0]; raw=_mat_records(mat)
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
    recs=[]; missing=[]
    for x in raw:
        vid=_video_id(x.get("video")); group=f"tvsum:{vid}"; p=by_name.get(vid.lower())
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
        recs.append(ManifestRecord(dataset="TVSum",version=DATASET_VERSION,video_id=vid,source_id=vid,source_group=group,path=str(p.resolve()) if p else None,source_url=ARCHIVE_URL,license="CC BY 3.0 (source YouTube videos; retain attribution)",license_url=LICENSE_URL,license_text_hash=None,download_status=status,sha256=sha256_file(p) if p else None,split=splits[group],annotation_type="summary_importance_2s",annotation_path=str(label_path.resolve()) if label_path else str(mat.resolve()),**meta,notes="TVSum summary importance proxy: 20-rater shot scores; not competition highlight/crop GT. archive_sha256="+archive_hash))
    write_manifest(recs,args.manifest)
    summary={"dataset":"TVSum","version":DATASET_VERSION,"archive":str(archive),"archive_sha256":archive_hash,"archive_url":ARCHIVE_URL,"readme_url":README_URL,"license":"CC BY 3.0","records":len(recs),"verified_videos":sum(r.download_status=="verified" for r in recs),"missing_or_failed":missing,"annotation_type":"summary_importance_2s","spatial_ground_truth":False}
    args.manifest.with_suffix(".summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
