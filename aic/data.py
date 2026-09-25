"""Dataset manifests and deterministic video metadata helpers for AIC."""
from __future__ import annotations
import dataclasses, hashlib, json, math, os, subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

MANIFEST_FIELDS = (
    "dataset", "version", "video_id", "source_id", "source_group", "path",
    "source_url", "license", "license_url", "license_text_hash", "license_gate",
    "download_status", "sha256", "duration", "fps", "frame_count", "width",
    "height", "rotation", "has_audio", "split", "annotation_type",
    "annotation_path", "notes",
)
REQUIRED_FIELDS = ("dataset", "version", "video_id", "source_id", "source_group", "download_status", "license_gate", "split", "annotation_type")
VALID_STATUS = {"pending", "downloaded", "verified", "missing", "failed", "not_licensed", "metadata_only"}
VALID_SPLITS = {"train", "val", "test", "unassigned", "excluded"}
# ``user_authorized_downloadable_source`` records provenance while honoring the
# project instruction that resources reachable by the Agent are authorized.
VALID_LICENSE_GATES = {"approved", "blocked", "pending", "unknown",
                        "user_authorized_downloadable_source"}

@dataclasses.dataclass
class ManifestRecord:
    dataset: str
    version: str
    video_id: str
    source_id: str
    source_group: str
    path: Optional[str] = None
    source_url: Optional[str] = None
    license: Optional[str] = None
    license_url: Optional[str] = None
    license_text_hash: Optional[str] = None
    license_gate: str = "unknown"
    download_status: str = "pending"
    sha256: Optional[str] = None
    duration: Optional[float] = None
    fps: Optional[float] = None
    frame_count: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    rotation: Optional[int] = None
    has_audio: Optional[bool] = None
    split: str = "unassigned"
    annotation_type: str = "none"
    annotation_path: Optional[str] = None
    notes: Optional[str] = None
    def as_dict(self) -> dict[str, Any]: return dataclasses.asdict(self)
    def validate(self, *, require_path: bool = False) -> None:
        d = self.as_dict(); missing = [f for f in REQUIRED_FIELDS if d.get(f) in (None, "")]
        if missing: raise ValueError(f"manifest record {self.video_id!r} missing fields: {missing}")
        if self.download_status not in VALID_STATUS: raise ValueError(f"{self.video_id}: invalid download_status={self.download_status!r}")
        if self.license_gate not in VALID_LICENSE_GATES: raise ValueError(f"{self.video_id}: invalid license_gate={self.license_gate!r}")
        if self.split not in VALID_SPLITS: raise ValueError(f"{self.video_id}: invalid split={self.split!r}")
        if require_path and self.download_status in {"downloaded", "verified"} and not self.path: raise ValueError(f"{self.video_id}: downloaded record has no path")
        if self.path and self.download_status in {"downloaded", "verified"} and not Path(self.path).exists(): raise ValueError(f"{self.video_id}: path does not exist: {self.path}")
        if self.fps is not None and (not math.isfinite(float(self.fps)) or self.fps <= 0): raise ValueError(f"{self.video_id}: invalid fps={self.fps}")
        if self.frame_count is not None and int(self.frame_count) < 0: raise ValueError(f"{self.video_id}: invalid frame_count={self.frame_count}")
        if self.duration is not None and (not math.isfinite(float(self.duration)) or self.duration < 0): raise ValueError(f"{self.video_id}: invalid duration={self.duration}")

def record_from_mapping(obj: Mapping[str, Any]) -> ManifestRecord:
    unknown = set(obj) - set(MANIFEST_FIELDS)
    if unknown: raise ValueError(f"unknown manifest fields: {sorted(unknown)}")
    rec = ManifestRecord(**{f: obj.get(f) for f in MANIFEST_FIELDS if f in obj}); rec.validate(); return rec

def read_manifest(path: os.PathLike[str] | str, *, require_path: bool = False) -> list[ManifestRecord]:
    out = []
    with Path(path).open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip() or line.lstrip().startswith("#"): continue
            try:
                rec = record_from_mapping(json.loads(line)); rec.validate(require_path=require_path)
            except Exception as e: raise ValueError(f"{path}:{lineno}: {e}") from e
            out.append(rec)
    return out

def write_manifest(records: Iterable[ManifestRecord | Mapping[str, Any]], path: os.PathLike[str] | str) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True); seen = set()
    with p.open("w", encoding="utf-8") as f:
        for item in records:
            rec = item if isinstance(item, ManifestRecord) else record_from_mapping(item); rec.validate()
            if rec.video_id in seen: raise ValueError(f"duplicate video_id: {rec.video_id}")
            seen.add(rec.video_id); f.write(json.dumps(rec.as_dict(), ensure_ascii=False, separators=(",", ":")) + "\n")

def sha256_file(path: os.PathLike[str] | str, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(chunk_size): h.update(chunk)
    return h.hexdigest()

def stable_split(source_group: str, *, seed: str = "aic-tvsum-v1", train: float = .70, val: float = .20) -> str:
    if not (0 <= train <= 1 and 0 <= val <= 1 and train + val <= 1): raise ValueError("invalid split fractions")
    value = int.from_bytes(hashlib.sha256(f"{seed}:{source_group}".encode()).digest()[:8], "big") / 2**64
    return "train" if value < train else "val" if value < train + val else "test"

def grouped_splits(source_groups: Iterable[str], *, seed: str = "aic-tvsum-v1", train: float = .70, val: float = .20) -> dict[str, str]:
    return {g: stable_split(g, seed=seed, train=train, val=val) for g in sorted(set(str(x) for x in source_groups))}

def probe_video(path: os.PathLike[str] | str) -> dict[str, Any]:
    cmd = ["ffprobe", "-v", "error", "-count_frames", "-print_format", "json",
           "-show_streams", "-show_format", str(path)]
    try: data = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)
    except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError) as e: raise RuntimeError(f"ffprobe failed for {path}: {e}") from e
    streams = data.get("streams", []); video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None: raise ValueError(f"no video stream: {path}")
    def ratio(s: Any) -> Optional[float]:
        try:
            a,b = str(s).split("/",1); return float(a)/float(b) if float(b) else None
        except Exception: return None
    fps = ratio(video.get("avg_frame_rate")) or ratio(video.get("r_frame_rate")); duration = video.get("duration") or data.get("format",{}).get("duration"); frames = video.get("nb_read_frames") or video.get("nb_frames")
    rotation = 0
    for x in video.get("side_data_list") or []:
        if "rotation" in x:
            try: rotation = int(x["rotation"])
            except (TypeError, ValueError): pass
    try: rotation = int((video.get("tags") or {}).get("rotate", rotation))
    except (TypeError, ValueError): pass
    return {"duration": float(duration) if duration not in (None,"N/A") else None, "fps": fps, "frame_count": int(frames) if frames not in (None,"N/A") else None, "width": int(video["width"]) if video.get("width") is not None else None, "height": int(video["height"]) if video.get("height") is not None else None, "rotation": rotation, "has_audio": any(s.get("codec_type") == "audio" for s in streams)}
