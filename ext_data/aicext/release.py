"""Immutable data releases and their readers.

A release lives in ``<EXT_ROOT>/_releases/<name>/`` and is never modified:
the builder writes into a hidden ``.<name>.building-*`` directory, adds
``RELEASE.json`` (sources, splits, label versions, code hashes) and
``SHA256SUMS`` (every file), renames it into place and makes it read-only.
An existing name is refused; changes go into a new ``_v<N+1>``.

The working registry (``processed/*.jsonl``, ``_registry``) can be refreshed at
any time; it never changes a finished release.  Training should read releases:

    from aicext.release import Release, SpatialCropUnits, TemporalEvidenceUnits
    rel = Release("spatial_crop_v1")          # checks RELEASE.json/manifests against SHA256SUMS
    ds = SpatialCropUnits(rel, "train")
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .common import EXT_ROOT, append_jsonl, dumps, now, read_jsonl, sha256_file
from .media import read_frames, uniform_sample
from . import window_scorer as ws

RELEASE_ROOT = EXT_ROOT / "_releases"
CODE_ROOT = Path(__file__).resolve().parents[1]
CODE_PARTS = ("aicext", "scripts", "configs", "tests", "examples")
SHA_CACHE = EXT_ROOT / "_registry" / "sha256_cache.jsonl"


# ---------------------------------------------------------------- hashing
def code_tree(root: Path = CODE_ROOT) -> dict:
    files = {}
    for part in CODE_PARTS:
        for p in sorted((root / part).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix in (".py", ".sh", ".json", ".md"):
                files[str(p.relative_to(root))] = sha256_file(p)
    tree = hashlib.sha256("".join(f"{k}\0{v}\n" for k, v in sorted(files.items())).encode()).hexdigest()
    return {"tree_sha256": tree, "files": files}


def _cache() -> dict:
    return {(r["path"], r["size"], r["mtime_ns"]): r["sha256"] for r in read_jsonl(SHA_CACHE)}


def sha256_many(paths: Iterable[str], workers: int = 4) -> dict[str, str]:
    """SHA-256 of media files, cached by (path, size, mtime_ns)."""
    from concurrent.futures import ProcessPoolExecutor

    cache = _cache()
    out, todo = {}, []
    for p in sorted(set(paths)):
        st_ = os.stat(p)
        key = (p, st_.st_size, st_.st_mtime_ns)
        if key in cache:
            out[p] = cache[key]
        else:
            todo.append((p, key))
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for (p, key), h in zip(todo, ex.map(sha256_file, [t[0] for t in todo], chunksize=4)):
                out[p] = h
                append_jsonl(SHA_CACHE, {"path": key[0], "size": key[1], "mtime_ns": key[2], "sha256": h})
    return out


# ---------------------------------------------------------------- writer
class ReleaseWriter:
    def __init__(self, name: str, root: Path = RELEASE_ROOT):
        self.name, self.root = name, Path(root)
        self.final = self.root / name
        if self.final.exists():
            raise FileExistsError(f"{self.final} exists; releases are immutable, publish a new version instead")
        self.dir = self.root / f".{name}.building-{os.getpid()}"
        if self.dir.exists():
            shutil.rmtree(self.dir)
        self.dir.mkdir(parents=True)

    def p(self, rel: str) -> Path:
        path = self.dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def copy(self, src: str | Path, rel: str, expect_sha: Optional[str] = None) -> str:
        dst = self.p(rel)
        shutil.copyfile(src, dst)
        h = sha256_file(dst)
        if expect_sha and h != expect_sha:
            raise IOError(f"copy of {src} changed bytes")
        return h

    def json(self, rel: str, obj) -> Path:
        path = self.p(rel)
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=_default) + "\n", encoding="utf-8")
        return path

    def jsonl(self, rel: str, rows: Iterable[dict]) -> int:
        path, n = self.p(rel), 0
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(dumps(r) + "\n")
                n += 1
        return n

    def npz(self, rel: str, **arrays) -> Path:
        path = self.p(rel)
        with open(path, "wb") as f:
            np.savez_compressed(f, **arrays)
        return path

    def snapshot_code(self) -> dict:
        tree = code_tree()
        for rel in tree["files"]:
            self.copy(CODE_ROOT / rel, f"code/ext_data/{rel}")
        return tree

    def finalize(self, meta: dict) -> Path:
        meta = {"release": self.name, "created": now(), "immutable": True, **meta}
        self.json("RELEASE.json", meta)
        sums = []
        for p in sorted(self.dir.rglob("*")):
            if p.is_file() and p.name != "SHA256SUMS":
                sums.append(f"{sha256_file(p)}  {p.relative_to(self.dir)}")
        (self.dir / "SHA256SUMS").write_text("\n".join(sums) + "\n")
        os.rename(self.dir, self.final)
        for p in sorted(self.final.rglob("*"), key=lambda q: -len(q.parts)):
            p.chmod(0o444 if p.is_file() else 0o555)
        self.final.chmod(0o555)
        append_jsonl(self.root / "INDEX.jsonl", {"release": self.name, "created": meta["created"],
                                                 "sha256sums": sha256_file(self.final / "SHA256SUMS"),
                                                 "files": len(sums)})
        return self.final


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


# ---------------------------------------------------------------- reader
class Release:
    """Read-only view of a frozen release.

    ``verify``: ``"none"`` | ``"manifest"`` (RELEASE.json, manifests and
    SHA256SUMS consistency; default) | ``"all"`` (every file, slow).
    """

    def __init__(self, name: Optional[str] = None, *, path: Optional[Path] = None, verify: str = "manifest"):
        self.dir = Path(path) if path else RELEASE_ROOT / name
        self.sums = {}
        sums = self.dir / "SHA256SUMS"
        if sums.exists():
            for line in sums.read_text().splitlines():
                h, rel = line.split("  ", 1)
                self.sums[rel] = h
        elif verify != "none":
            raise IOError(f"{self.dir} has no SHA256SUMS (not a finalized release)")
        self.meta = json.loads((self.dir / "RELEASE.json").read_text())
        if verify != "none":
            self.verify(deep=(verify == "all"))

    def verify_media(self, split: str, limit: Optional[int] = None) -> dict:
        """Re-hash referenced media (not copied into the release) against the manifest."""
        rows = self.manifest(split)[:limit] if limit else self.manifest(split)
        bad = [r["media_path"] for r in rows if sha256_file(r["media_path"]) != r["media_sha256"]]
        return {"checked": len(rows), "bad": bad}

    def verify(self, deep: bool = False) -> dict:
        keys = [k for k in self.sums if deep or k == "RELEASE.json" or k.startswith("manifests/")]
        bad = [k for k in keys if sha256_file(self.dir / k) != self.sums[k]]
        if bad:
            raise IOError(f"release {self.dir.name}: {len(bad)} files differ from SHA256SUMS, e.g. {bad[:3]}")
        return {"checked": len(keys), "bad": 0}

    def path(self, rel: str) -> Path:
        return self.dir / rel

    def manifest(self, split: str) -> list[dict]:
        return list(read_jsonl(self.dir / "manifests" / f"{split}.jsonl"))

    def jsonl(self, rel: str) -> list[dict]:
        return list(read_jsonl(self.dir / rel))


def _times(rel: Release, row: dict) -> np.ndarray:
    return np.load(rel.path(row["timeline"]))["time_s"]


class SpatialCropUnits:
    """spatial_crop_v1: one item = one human crop unit.

    RetargetVid unit = (video, ratio), 6 annotators, every frame annotated.
    LIVE-YT-VC unit = video, 1 annotator per frame, 30 annotated frames.
    ``boxes_xywh[A, K, 4]`` are half-open coded pixels converted from the raw
    ``ltrb`` without clipping (``ltrb_raw`` is returned too); ``valid[A, K]``.
    ``frames``: ``"annotated"`` (all annotated frames) or an fps on the real
    PTS clock (first frame at/after each grid time; sparse units always use
    their annotated frames).  Derived (interpolated) boxes are returned only
    with ``include_derived=True`` and only under ``derived_*`` keys.
    """

    def __init__(self, release: Release, split: str, datasets: Optional[Iterable[str]] = None,
                 frames: str | float = "annotated", max_frames: Optional[int] = None,
                 include_derived: bool = False, decode: bool = False, size: Optional[int] = 256):
        self.rel = release
        keep = set(datasets) if datasets else None
        self.rows = [r for r in release.manifest(split) if keep is None or r["dataset"] in keep]
        self.frames, self.max_frames = frames, max_frames
        self.include_derived, self.decode, self.size = include_derived, decode, size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        z = np.load(self.rel.path(r["human_annotation"]["packed"]))
        ltrb = z["ltrb_raw"].astype(np.float64)          # [A, K, 4]
        fidx = z["frame_idx"].astype(np.int64)            # [K]
        t_all = _times(self.rel, r)
        pick = np.arange(len(fidx))
        if r["human_annotation"]["density"] == "dense" and self.frames != "annotated":
            want, _ = uniform_sample(t_all, float(self.frames))
            pick = np.nonzero(np.isin(fidx, want))[0]
        if self.max_frames and len(pick) > self.max_frames:
            pick = pick[:self.max_frames]
        ltrb, fidx = ltrb[:, pick], fidx[pick]
        xywh = np.concatenate([ltrb[..., :2], ltrb[..., 2:] - ltrb[..., :2]], -1)
        W, H = r["width"], r["height"]
        valid = np.isfinite(xywh).all(-1) & (xywh[..., 2] > 0) & (xywh[..., 3] > 0) & (fidx[None] < r["frame_count"])
        out = {"unit_id": r["unit_id"], "dataset": r["dataset"], "group_id": r["group_id"],
               "target_ratio_wh": r["target_ratio_wh"], "width": W, "height": H,
               "frame_idx": fidx, "time_s": t_all[np.clip(fidx, 0, len(t_all) - 1)],
               "boxes_xywh": xywh.astype(np.float32), "ltrb_raw": ltrb.astype(np.float32),
               "boxes_xywh_norm": (xywh / np.array([W, H, W, H])).astype(np.float32), "valid": valid,
               "annotators": list(z["annotators"]), "gt_source": "human",
               "gt_is_max_window": r["gt_is_max_window"], "coord_version": r["coord_version"],
               "max_window_wh_axis": ws.max_window(W, H, r["target_ratio_wh"])}
        d = r.get("derived_annotation")
        if self.include_derived and d:
            zd = np.load(self.rel.path(d["packed"]))
            out.update(derived_boxes_xywh=zd["boxes_xywh"].astype(np.float32), derived_valid=zd["valid"],
                       derived_source="derived", derived_version=d["version"])
        if self.decode and len(fidx):
            got = read_frames(r["media_path"], fidx, size=self.size)
            out["frames"] = np.stack([got[int(k)] for k in fidx])
        return out


def score_unit_frame(item: dict, k: int, candidates_xywh: Optional[np.ndarray] = None, n: int = 65,
                     iou: str = "halfopen", gt_boundary: str = "none", with_ceiling: bool = True) -> dict:
    """Window-scorer targets for frame ``k`` of a SpatialCropUnits item (human boxes only)."""
    W, H, ratio = item["width"], item["height"], item["target_ratio_wh"]
    if candidates_xywh is None:
        candidates_xywh = ws.legal_candidates(W, H, ratio, n=n)["boxes_xywh"]
    return ws.score_candidates(W, H, ratio, candidates_xywh, item["boxes_xywh"][:, k], item["valid"][:, k],
                               gt_source=item["gt_source"], gt_is_max_window=item["gt_is_max_window"],
                               iou=iou, gt_boundary=gt_boundary, annotators=item["annotators"],
                               with_ceiling=with_ceiling)


def score_unit(item: dict, n: int = 65, iou: str = "halfopen", gt_boundary: str = "none") -> dict:
    """All annotated frames of a unit: iou[K, C, A], valid[K, C, A], iou_mean[K, C], candidate offsets."""
    W, H, ratio = item["width"], item["height"], item["target_ratio_wh"]
    cand = ws.legal_candidates(W, H, ratio, n=n)
    per = [score_unit_frame(item, k, cand["boxes_xywh"], iou=iou, gt_boundary=gt_boundary, with_ceiling=False)
           for k in range(len(item["frame_idx"]))]
    stack = lambda key: np.stack([p[key] for p in per]) if per else np.zeros((0,))  # noqa: E731
    return {"frame_idx": item["frame_idx"], "offsets": cand["offsets"], "axis": cand["axis"],
            "iou": stack("iou"), "valid": stack("valid"), "iou_mean": stack("iou_mean"),
            "n_valid": stack("n_valid"), "conventions": per[0]["conventions"] if per else None}


def preference_pairs(item: dict, margin: float = 1.0) -> np.ndarray:
    """Within-video relative preferences from raw MTurk soft votes: rows (i, j) with votes_i - votes_j >= margin.

    Clip indices refer to ``clip_*`` arrays of a TemporalEvidenceUnits item.  These are
    relative (same video, same turkers), not absolute positives/negatives.
    """
    v = np.asarray(item["clip_mturk_votes"], float)
    ok = np.nonzero(np.isfinite(v))[0]
    if len(ok) < 2:
        return np.zeros((0, 2), np.int64)
    d = v[ok][:, None] - v[ok][None, :]
    i, j = np.nonzero(d >= margin)
    return np.stack([ok[i], ok[j]], 1)


class FeatureSubsetUnits:
    """mrhisum_feat_subset_v1: YT-8M frame features + Mr.HiSum gtscore (feature head only; no raw video).

    ``features[T, 1024]`` (``with_audio`` -> 1152) dequantized as upstream
    (``q * 4/255 - 2``), ``gtscore[T]``, ``valid[T]`` (both a feature row and a
    label exist), ``grid_time_s = k * 1.0`` (anchor 0).
    """

    def __init__(self, release: Release, split: str, max_steps: Optional[int] = None, with_audio: bool = False):
        import h5py

        self._h5py = h5py
        self.rel = release
        self.rows = release.manifest(split)
        self.max_steps, self.with_audio = max_steps, with_audio
        self._files: dict[str, object] = {}

    def __len__(self):
        return len(self.rows)

    def _f(self, rel_path: str):
        if rel_path not in self._files:
            if len(self._files) > 32:
                for f in self._files.values():
                    f.close()
                self._files.clear()
            self._files[rel_path] = self._h5py.File(self.rel.path(rel_path), "r")
        return self._files[rel_path]

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        score = np.asarray(self._f(r["labels_h5"])[r["video_id"]]["gtscore"], np.float32)
        g = self._f(r["feature_h5"])[r["video_id"]]
        feats = np.asarray(g["rgb"], np.float32) * (4.0 / 255.0) - 2.0
        if self.with_audio:
            aud = np.asarray(g["audio"], np.float32) * (4.0 / 255.0) - 2.0
            feats = np.concatenate([feats, aud[:len(feats)]], 1)
        n = max(len(feats), len(score))
        valid = np.zeros(n, bool)
        valid[:min(len(feats), len(score))] = True
        f_pad = np.zeros((n, feats.shape[1]), np.float32)
        f_pad[:len(feats)] = feats
        s_pad = np.zeros(n, np.float32)
        s_pad[:len(score)] = score
        out = {"video_id": r["video_id"], "group_id": r["group_id"], "features": f_pad, "gtscore": s_pad,
               "valid": valid, "grid_time_s": np.arange(n, dtype=np.float32) * 1.0}
        if self.max_steps and n > self.max_steps:
            for k in ("features", "gtscore", "valid", "grid_time_s"):
                out[k] = out[k][:self.max_steps]
        return out


HUMAN_CODES = {0: "unlabelled", 1: "human_unselected_weak", 2: "human_selected"}
AUTO_CODES = {0: "unlabelled", 1: "auto_unmatched_weak", 2: "auto_borderline", 3: "auto_matched"}


class TemporalEvidenceUnits:
    """temporal_evidence_v1 (YouTube Highlights): per-frame evidence channels.

    Clips are half-open ``[start_idx, end_idx)`` display-frame intervals; the
    per-clip raw values are returned unchanged (``clip_mturk_votes`` soft vote
    count, NaN when the video has no MTurk labels; ``clip_match_label`` 1/0/-1).
    Per sampled frame: ``votes_max``/``votes_mean`` over covering MTurk clips
    (NaN where none), ``human_code`` (see ``HUMAN_CODES``), ``auto_code``
    (``AUTO_CODES``) and ``human_valid``/``auto_valid``.  Nothing is a negative
    label: unselected / unmatched clips are ``*_weak`` codes and uncovered
    frames are ``unlabelled``.
    """

    def __init__(self, release: Release, split: str, fps: Optional[float] = 2.0,
                 max_frames: Optional[int] = None, require_mturk: bool = False):
        self.rel = release
        self.rows = [r for r in release.manifest(split) if not require_mturk or r["has_mturk"]]
        self.fps, self.max_frames = fps, max_frames

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        z = np.load(self.rel.path(r["packed"]))
        s, e = z["clip_start_idx"], z["clip_end_idx"]
        votes, match = z["clip_mturk_votes"], z["clip_match_label"]
        t_all = _times(self.rel, r)
        idx = np.arange(len(t_all)) if self.fps is None else uniform_sample(t_all, self.fps)[0]
        if self.max_frames and len(idx) > self.max_frames:
            idx = idx[:self.max_frames]
        cover = (idx[:, None] >= s[None]) & (idx[:, None] < e[None])          # [T, K]
        has_v = cover & np.isfinite(votes)[None]
        n_v = has_v.sum(1)
        vv = np.where(has_v, votes[None], 0.0)
        votes_max = np.where(n_v > 0, np.where(has_v, votes[None], -np.inf).max(1), np.nan)
        votes_mean = np.where(n_v > 0, vv.sum(1) / np.maximum(n_v, 1), np.nan)
        human = np.where(n_v == 0, 0, np.where(votes_max > 0, 2, 1)).astype(np.int8)
        n_c = cover.sum(1)
        m = np.where(cover, match[None], -9)
        auto = np.where(n_c == 0, 0, np.where((m == 1).any(1), 3, np.where((m == 0).any(1), 2, 1))).astype(np.int8)
        return {"unit_id": r["unit_id"], "group_id": r["group_id"], "domain": r["domain"],
                "frame_idx": idx, "time_s": t_all[idx], "votes_max": votes_max.astype(np.float32),
                "votes_mean": votes_mean.astype(np.float32), "n_mturk_clips": n_v.astype(np.int16),
                "human_code": human, "human_valid": n_v > 0, "auto_code": auto, "auto_valid": n_c > 0,
                "clip_start_idx": s, "clip_end_idx": e, "clip_mturk_votes": votes, "clip_match_label": match,
                "has_mturk": r["has_mturk"], "label_version": r["label_version"]}


def make_readonly_check(path: Path) -> bool:
    """True if nothing under ``path`` is writable by its owner."""
    return all(not (p.stat().st_mode & stat.S_IWUSR) for p in [path, *path.rglob("*")])
