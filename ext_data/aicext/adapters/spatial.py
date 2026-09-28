"""Spatial supervision readers: crop windows, saliency / fixation maps, object masks.

Output coordinate convention for all box adapters: ``xywh`` in source coded
pixels, float32, ``x,y`` = top-left corner, frame-aligned to 0-based display
indices; ``xywh_norm`` divides by (W, H, W, H).  The per-dataset source format
is recorded in the annotation index (``coord_format``) and converted here only.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .base import Index, load_video_frames, media_times, sample_indices


@lru_cache(maxsize=4096)
def _read_ltrb(path: str) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", ndmin=2).astype(np.float32)


def ltrb_to_xywh(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    out[:, 2] = a[:, 2] - a[:, 0]
    out[:, 3] = a[:, 3] - a[:, 1]
    return out


class DenseCropAdapter:
    """RetargetVid: one sample = (video, ratio); all annotators stacked.

    Returns ``boxes[A, T, 4]`` xywh, ``valid[A, T]``, sampled ``frame_idx``/``time_s``.
    Different annotators and ratios of one video share ``group_id``.
    """

    def __init__(self, dataset: str = "RetargetVid", split: Optional[str | Iterable[str]] = None,
                 ratios: Iterable[str] = ("1-3", "3-1"), fps: Optional[float] = 2.0,
                 max_frames: Optional[int] = None, decode: bool = False, size: Optional[int] = 256):
        self.ix = Index.load(dataset)
        pairs = self.ix.select("crop_box_dense", split=split, where=lambda m, a: a["variant"] in set(ratios))
        groups: dict[tuple[str, str], list[dict]] = {}
        for m, a in pairs:
            groups.setdefault((m["uid"], a["variant"]), []).append(a)
        self.items = sorted(groups.items())
        self.fps, self.max_frames, self.decode, self.size = fps, max_frames, decode, size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        (uid, ratio), anns = self.items[i]
        m = self.ix.media[uid]
        idx, t = sample_indices(m, self.fps, self.max_frames)
        boxes = np.zeros((len(anns), len(idx), 4), np.float32)
        valid = np.zeros((len(anns), len(idx)), bool)
        for k, a in enumerate(sorted(anns, key=lambda a: a["annotator"])):
            b = ltrb_to_xywh(_read_ltrb(a["annotation_path"]))
            ok = idx < len(b)
            boxes[k, ok] = b[idx[ok]]
            valid[k, ok] = True
        W, H = m["width"], m["height"]
        out = {"uid": uid, "group_id": m["group_id"], "ratio": ratio, "frame_idx": idx, "time_s": t,
               "boxes_xywh": boxes, "boxes_xywh_norm": boxes / np.array([W, H, W, H], np.float32),
               "valid": valid, "width": W, "height": H,
               "annotators": [a["annotator"] for a in sorted(anns, key=lambda a: a["annotator"])]}
        if self.decode:
            out["frames"] = load_video_frames(m, idx, self.size)
        return out


class SparseCropAdapter:
    """Sparse human crop boxes (LIVE-YT VC): only annotated frames are valid.

    The annotation JSON (written by the ingest script) holds
    ``{"frames": [...display idx...], "boxes_xywh": [[x,y,w,h], ...]}`` per
    (annotator, ratio).  Interpolated tracks live in separate
    ``crop_box_derived`` records and are not returned here.
    """

    def __init__(self, dataset: str = "LIVE_YT_VC", split=None, annotation_type: str = "crop_box_sparse",
                 decode: bool = False, size: Optional[int] = 256):
        import json

        self._json = json
        self.ix = Index.load(dataset)
        self.items = self.ix.select(annotation_type, split=split)
        self.decode, self.size = decode, size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        m, a = self.items[i]
        d = self._json.loads(Path(a["annotation_path"]).read_text())
        idx = np.asarray(d["frames"], np.int64)
        boxes = np.asarray(d["boxes_xywh"], np.float32).reshape(-1, 4)
        W, H = m["width"], m["height"]
        out = {"uid": m["uid"], "group_id": m["group_id"], "annotation_uid": a["uid"], "variant": a["variant"],
               "annotator": a["annotator"], "frame_idx": idx, "time_s": media_times(m)[idx] if len(idx) else idx,
               "boxes_xywh": boxes, "boxes_xywh_norm": boxes / np.array([W, H, W, H], np.float32),
               "valid": np.isfinite(boxes).all(1), "width": W, "height": H,
               "source": a["annotation_source"]}
        if self.decode and len(idx):
            out["frames"] = load_video_frames(m, idx, self.size)
        return out


class SaliencyMapAdapter:
    """DHF1K saliency maps / fixation maps sampled on the real clock.

    ``kind='saliency_map'`` -> float map in [0,1]; ``kind='fixation_points'`` ->
    binary map.  Frames without a PNG are invalid (``valid=False``), not zero.
    """

    def __init__(self, dataset: str = "DHF1K", kind: str = "saliency_map", split=None,
                 fps: Optional[float] = 2.0, max_frames: Optional[int] = 16, map_size: Optional[int] = 128,
                 decode: bool = False, size: Optional[int] = 256, seed: int = 0):
        self.ix = Index.load(dataset)
        self.items = self.ix.select(kind, split=split)
        self.kind, self.fps, self.max_frames, self.map_size = kind, fps, max_frames, map_size
        self.decode, self.size = decode, size
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        import cv2

        m, a = self.items[i]
        idx, t = sample_indices(m, self.fps, self.max_frames, self.rng)
        d = Path(a["annotation_path"])
        maps, valid = [], []
        for k in idx:
            p = d / f"{int(k) + 1:04d}.png"
            if not p.exists():
                maps.append(None)
                valid.append(False)
                continue
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if self.kind == "fixation_points":
                # keep fixation points: max-pool style resize so points survive downscaling
                img = (img > 0).astype(np.float32)
                if self.map_size:
                    img = cv2.resize(img, self._dsize(img), interpolation=cv2.INTER_AREA)
                    img = (img > 0).astype(np.float32)
            else:
                img = img.astype(np.float32) / 255.0
                if self.map_size:
                    img = cv2.resize(img, self._dsize(img), interpolation=cv2.INTER_AREA)
            maps.append(img)
            valid.append(True)
        shape = next((x.shape for x in maps if x is not None), (1, 1))
        arr = np.stack([x if x is not None else np.zeros(shape, np.float32) for x in maps])
        out = {"uid": m["uid"], "group_id": m["group_id"], "kind": self.kind, "frame_idx": idx, "time_s": t,
               "maps": arr, "valid": np.asarray(valid), "width": m["width"], "height": m["height"]}
        if self.decode:
            out["frames"] = load_video_frames(m, idx, self.size)
        return out

    def _dsize(self, img):
        h, w = img.shape[:2]
        s = self.map_size / max(h, w)
        return (max(1, round(w * s)), max(1, round(h * s)))


class CropCandidatesAdapter:
    """GAICD grid-anchor candidates: boxes + MOS per image; unrated candidates are invalid.

    Returns ``boxes_xywh[K,4]`` (pixels), ``boxes_xywh_norm``, ``mos[K]``,
    ``valid[K]``; with ``max_candidates`` the arrays are padded (valid=False).
    """

    def __init__(self, dataset: str = "GAICD", split=None, variant: str = "journal",
                 decode: bool = False, size: Optional[int] = 256, max_candidates: Optional[int] = 90):
        self.ix = Index.load(dataset)
        self.items = self.ix.select("image_crop_candidates", split=split,
                                    where=lambda m, a: a["variant"] == variant)
        self.decode, self.size, self.max_candidates = decode, size, max_candidates

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        import json

        m, a = self.items[i]
        d = json.loads(Path(a["annotation_path"]).read_text())
        boxes = np.asarray(d["boxes_xywh"], np.float32).reshape(-1, 4)
        mos = np.asarray(d["mos"], np.float32)
        valid = np.asarray(d["valid"], bool)
        W, H = d["W"], d["H"]
        if self.max_candidates:
            K = self.max_candidates
            if len(boxes) > K:
                raise ValueError(f"{m['uid']}: {len(boxes)} candidates > max_candidates={K}")
            pad = K - len(boxes)
            boxes = np.concatenate([boxes, np.zeros((pad, 4), np.float32)])
            mos = np.concatenate([mos, np.zeros(pad, np.float32)])
            valid = np.concatenate([valid, np.zeros(pad, bool)])
        out = {"uid": m["uid"], "group_id": m["group_id"], "variant": a["variant"], "boxes_xywh": boxes,
               "boxes_xywh_norm": boxes / np.array([W, H, W, H], np.float32), "mos": mos, "valid": valid,
               "width": W, "height": H}
        if self.decode:
            out["image"] = load_video_frames(m, np.array([0]), self.size)[0]
        return out


class MaskSequenceAdapter:
    """Salient-object masks (DAVSOD): only frames with a GT PNG are valid.

    The annotation record's coverage lists ``annotated_frames`` (0-based
    indices into the sequence's frame list) and ``mask_pattern``.
    """

    def __init__(self, dataset: str = "DAVSOD", split=None, max_frames: Optional[int] = 8,
                 size: Optional[int] = 256, decode: bool = True, seed: int = 0, variant: str = "object_level"):
        self.ix = Index.load(dataset)
        # object_level = binary PNG; instance_level = palette ids (returned as ids, not thresholded)
        self.items = self.ix.select("salient_object_mask", split=split,
                                    where=lambda m, a: a.get("variant") == variant)
        self.variant = variant
        self.max_frames, self.size, self.decode = max_frames, size, decode
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        import cv2

        m, a = self.items[i]
        cov = a["coverage"]
        ann = np.asarray(cov["annotated_frame_idx"], np.int64)
        if self.max_frames and len(ann) > self.max_frames:
            ann = np.sort(self.rng.choice(ann, self.max_frames, replace=False))
        mask_files = cov["mask_files"]
        lookup = dict(zip(cov["annotated_frame_idx"], mask_files))
        masks = []
        for k in ann:
            path = str(Path(a["annotation_path"]) / lookup[int(k)])
            if self.variant == "instance_level":
                from PIL import Image

                g = np.asarray(Image.open(path)).astype(np.uint8)  # palette index = instance id
            else:
                g = (cv2.imread(path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
            if self.size:
                h, w = g.shape
                s = self.size / max(h, w)
                g = cv2.resize(g, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_NEAREST)
            masks.append(g)
        out = {"uid": m["uid"], "group_id": m["group_id"], "frame_idx": ann,
               "time_s": media_times(m)[ann] if len(ann) else ann, "masks": np.stack(masks) if masks else None,
               "width": m["width"], "height": m["height"]}
        if self.decode and len(ann):
            out["frames"] = load_video_frames(m, ann, self.size)
        return out


class ObjectTrackAdapter:
    """LaSOT-style single-object tracks: per-frame box + visibility, identity kept.

    Returns ``boxes_xywh[T,4]`` in 0-based pixels (shifted by -1 when the
    annotation notes say the source is 1-based), ``visible[T]``, ``valid[T]``
    (annotated), ``full_occlusion``/``out_of_view`` flags and the target text.
    Invisible frames keep their raw box but ``visible=False``.
    """

    def __init__(self, dataset: str = "LaSOT", split=None, fps: Optional[float] = None,
                 max_frames: Optional[int] = 32, seed: int = 0, decode: bool = False, size: Optional[int] = 256):
        self.ix = Index.load(dataset)
        self.items = self.ix.select("object_track_box", split=split)
        self.fps, self.max_frames, self.decode, self.size = fps, max_frames, decode, size
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        import json

        m, a = self.items[i]
        d = json.loads(Path(a["annotation_path"]).read_text())
        one_based = json.loads(a.get("notes") or "{}").get("one_based", False)
        gt = np.asarray(d["raw_xywh"], np.float32)
        occ, oov = np.asarray(d["full_occlusion"], bool), np.asarray(d["out_of_view"], bool)
        L = min(len(gt), len(occ), len(oov), int(m["frame_count"]))
        idx, t = sample_indices(m, self.fps, self.max_frames, self.rng)
        idx, t = idx[idx < L], t[idx < L]
        b = gt[idx].copy()
        if one_based:
            b[:, :2] -= 1.0
        W, H = d["W"], d["H"]
        vis = ~occ[idx] & ~oov[idx] & (b[:, 2] > 0) & (b[:, 3] > 0)
        out = {"uid": m["uid"], "group_id": m["group_id"], "target": d.get("nlp"), "category": d.get("category"),
               "frame_idx": idx, "time_s": t, "boxes_xywh": b, "boxes_xywh_norm": b / np.array([W, H, W, H], np.float32),
               "visible": vis, "valid": np.ones(len(idx), bool), "full_occlusion": occ[idx], "out_of_view": oov[idx],
               "width": W, "height": H}
        if self.decode and len(idx):
            out["frames"] = load_video_frames(m, idx, self.size)
        return out


class MaskletAdapter:
    """SA-V masklets (COCO RLE) at the annotation rate, manual and auto kept apart.

    ``source='manual'|'auto'`` selects the annotation record.  Masks exist only
    on annotated frames (every 4th frame at 24 fps); ``frame_idx`` are those
    video frames.  Returns ``masks[K,T,h,w]`` uint8 for up to ``max_objects``
    masklets, ``visible[K,T]`` (non-empty mask) and masklet ids.
    """

    def __init__(self, dataset: str = "SA_V", split=None, source: str = "manual", max_objects: int = 4,
                 max_frames: Optional[int] = 8, size: Optional[int] = 128, seed: int = 0, decode: bool = False):
        self.ix = Index.load(dataset)
        self.items = self.ix.select("object_masklet", split=split,
                                    where=lambda m, a: a["annotation_source"] == ("human" if source == "manual"
                                                                                  else "auto"))
        self.max_objects, self.max_frames, self.size, self.decode = max_objects, max_frames, size, decode
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        import json

        import cv2
        from pycocotools import mask as mask_utils

        m, a = self.items[i]
        d = json.loads(Path(a["annotation_path"]).read_text())
        step = int(a["coverage"]["frame_step"])
        rles = d["masklet"]  # [T_annot][K]
        T = len(rles)
        k_all = len(rles[0]) if T else 0
        ks = np.arange(min(self.max_objects, k_all))
        ts = np.arange(T)
        if self.max_frames and T > self.max_frames:
            s0 = int(self.rng.integers(0, T - self.max_frames + 1))
            ts = ts[s0:s0 + self.max_frames]
        H, W = int(d["video_height"]), int(d["video_width"])
        oh, ow = (H, W) if not self.size else (max(1, round(H * self.size / max(H, W))),
                                               max(1, round(W * self.size / max(H, W))))
        masks = np.zeros((len(ks), len(ts), oh, ow), np.uint8)
        vis = np.zeros((len(ks), len(ts)), bool)
        for jj, tt in enumerate(ts):
            for ii, kk in enumerate(ks):
                mk = mask_utils.decode(rles[tt][kk])
                vis[ii, jj] = bool(mk.any())
                if self.size:
                    mk = cv2.resize(mk, (ow, oh), interpolation=cv2.INTER_NEAREST)
                masks[ii, jj] = mk
        fidx = ts * step
        out = {"uid": m["uid"], "group_id": m["group_id"], "source": a["annotation_source"],
               "masklet_ids": [d["masklet_id"][k] for k in ks], "frame_idx": fidx,
               "time_s": media_times(m)[np.clip(fidx, 0, int(m["frame_count"]) - 1)] if m.get("timeline_path")
               else fidx / 24.0, "masks": masks, "visible": vis, "width": W, "height": H}
        if self.decode:
            out["frames"] = load_video_frames(m, np.clip(fidx, 0, int(m["frame_count"]) - 1), self.size)
        return out
