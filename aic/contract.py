"""Local submission contract derived from rules document 01, version 2026-09-24.

This is not the unavailable official validator. Coordinates use continuous pixel
geometry (no rounding of derived height). Video frames are zero based decoded
indices. MB's byte base, rotation conventions, and official rounding remain open.
The metadata index must describe the exact coordinate/decode convention used.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence


class ContractError(ValueError):
    """Malformed data or a locally invalid competition submission."""


def finite_number(value: Any) -> bool:
    """JSON numeric values, excluding bool, NaN, Infinity, and overflow."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def parse_ratio(value: Any) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ContractError("targetRatioWH must contain exactly two positive numbers")
    if not all(finite_number(v) and v > 0 for v in value):
        raise ContractError("targetRatioWH must contain finite positive numbers")
    rw, rh = float(value[0]), float(value[1])
    if not math.isfinite(rh / rw) or rh / rw == 0:
        raise ContractError("targetRatioWH has an unrepresentable aspect ratio")
    return rw, rh


@dataclass(frozen=True)
class VideoMetadata:
    video_id: str
    width: int
    height: int
    frame_count: int
    target_ratio: tuple[float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.video_id, str) or not self.video_id:
            raise ContractError("video_id must be a nonempty string; IDs are never coerced")
        for name in ("width", "height", "frame_count"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ContractError(f"{name} must be a positive integer")
        object.__setattr__(self, "target_ratio", parse_ratio(self.target_ratio))

    @property
    def targetRatioWH(self) -> tuple[float, float]:
        """JSON contract spelling, retained alongside Pythonic target_ratio."""
        return self.target_ratio

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "VideoMetadata":
        try:
            return cls(record["video_id"], record["width"], record["height"],
                       record["frame_count"], record["targetRatioWH"])
        except KeyError as error:
            raise ContractError(f"metadata index missing {error.args[0]}") from error


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    video_id: str | None = None
    prediction_index: int | None = None
    severity: str = "error"


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)
    video_count: int = 0
    prediction_count: int = 0
    model_size_mb: float | None = None
    stage: str = "preliminary"

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    def raise_for_errors(self) -> None:
        if not self.valid:
            message = "; ".join(f"{i.code}: {i.message}" for i in self.issues
                                if i.severity == "error")
            raise ContractError(message)

    def to_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, **asdict(self),
                "validator": "local_rules_2026-09-24_not_official"}


def _reject_constant(value: str) -> None:
    raise ContractError(f"nonstandard JSON numeric constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _assert_finite_tree(value: Any) -> None:
    if isinstance(value, dict):
        for nested in value.values():
            _assert_finite_tree(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _assert_finite_tree(nested)
    elif isinstance(value, (float, int)) and not isinstance(value, bool):
        if not finite_number(value):
            raise ContractError("NaN, Infinity, or numeric overflow in JSON record")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read strict UTF-8 JSONL; never ignore malformed or blank lines."""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                if not line.strip():
                    raise ContractError("blank JSONL line")
                row = json.loads(line, parse_constant=_reject_constant,
                                 object_pairs_hook=_unique_object)
                if not isinstance(row, dict):
                    raise ContractError("each JSONL line must be an object")
                _assert_finite_tree(row)
                rows.append(row)
            except (ValueError, TypeError, RecursionError) as error:
                raise ContractError(f"{path}:{line_number}: {error}") from error
    if not rows:
        raise ContractError(f"{path}: empty JSONL file")
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Atomic JSONL serialization; use write_submission for schema validation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary_path = Path(stream.name)
            count = 0
            for row in rows:
                if not isinstance(row, Mapping):
                    raise ContractError("each JSONL line must be an object")
                _assert_finite_tree(row)
                stream.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False,
                                        separators=(",", ":")) + "\n")
                count += 1
            if count == 0:
                raise ContractError("cannot write an empty JSONL file")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_index(path: str | Path) -> dict[str, VideoMetadata]:
    """Load an enriched index with verified dimensions and decoded frame counts."""
    result: dict[str, VideoMetadata] = {}
    for row in load_jsonl(path):
        metadata = VideoMetadata.from_record(row)
        if metadata.video_id in result:
            raise ContractError(f"duplicate index video_id: {metadata.video_id}")
        result[metadata.video_id] = metadata
    return result


def crop_to_xywh(crop: Sequence[float], target_ratio: Sequence[float]) -> tuple[float, ...]:
    if not isinstance(crop, (list, tuple)) or len(crop) != 3:
        raise ContractError("bboxes must be a single flat [x,y,w] triple")
    if not all(finite_number(v) for v in crop):
        raise ContractError("crop coordinates must be finite numbers, excluding bool")
    x, y, width = (float(v) for v in crop)
    rw, rh = parse_ratio(target_ratio)
    height = width * (rh / rw)
    if x < 0 or y < 0 or width <= 0 or not math.isfinite(height) or height <= 0:
        raise ContractError("crop requires x>=0, y>=0, w>0, and finite h>0")
    return x, y, width, height


def validate_prediction(prediction: Any, metadata: VideoMetadata) -> list[str]:
    """Return geometric/frame errors, without order or duplicate checks."""
    if not isinstance(prediction, Mapping):
        return ["prediction must be an object"]
    errors: list[str] = []
    frame = prediction.get("frame")
    if type(frame) is not int or not 0 <= frame < metadata.frame_count:
        errors.append(f"frame must be integer in [0,{metadata.frame_count - 1}]")
    try:
        x, y, width, height = crop_to_xywh(prediction.get("bboxes"), metadata.target_ratio)
        if x + width > metadata.width or y + height > metadata.height:
            errors.append("crop extends outside the original video dimensions")
    except (ContractError, TypeError) as error:
        errors.append(str(error))
    try:
        _assert_finite_tree(prediction)
    except ContractError as error:
        errors.append(str(error))
    return errors


def size_coefficient(model_size_mb: float) -> float:
    if not finite_number(model_size_mb) or not 0 < model_size_mb <= 9216:
        raise ContractError("model_size_mb must satisfy 0 < M <= 9216")
    if model_size_mb <= 100:
        return 1.0
    if model_size_mb <= 500:
        return 0.95
    return 0.90


def _normalise_stage(stage: str) -> str:
    aliases = {"preliminary": "preliminary", "initial": "preliminary", "prelim": "preliminary",
               "final": "final", "replay": "final", "semifinal": "final", "semi-final": "final"}
    try:
        return aliases[stage.lower()]
    except (AttributeError, KeyError) as error:
        raise ContractError("stage must be preliminary or final") from error


def validate_submission(
    rows: Sequence[Mapping[str, Any]],
    index: Mapping[str, VideoMetadata],
    *,
    stage: str = "preliminary",
    actual_model_size_mb: float | None = None,
    require_sorted: bool = True,
) -> ValidationReport:
    """Check complete index coverage, fields, geometry, and model-size consistency.

    Unsorted frames are rejected by default because the submission contract asks
    for deterministic ordered output; set require_sorted=False only for a
    diagnostic inspection. Document 01 recommends sorting but does not itself
    establish a whole-submission rejection rule.
    Empty prediction lists are legal. Empty submissions/indexes are not.
    Actual file-size matching, if requested, uses the caller's explicit MB base.
    """
    stage = _normalise_stage(stage)
    if not index:
        raise ContractError("metadata index must not be empty")
    if any(not isinstance(meta, VideoMetadata) or key != meta.video_id
           for key, meta in index.items()):
        raise ContractError("index keys must match VideoMetadata.video_id")
    if actual_model_size_mb is not None:
        size_coefficient(actual_model_size_mb)
    report = ValidationReport(stage=stage)
    seen_ids: set[str] = set()
    sizes: list[float] = []
    size_present = 0
    for row_number, row in enumerate(rows, 1):
        report.video_count += 1
        if not isinstance(row, Mapping):
            report.issues.append(ValidationIssue("row_type", f"row {row_number} must be an object"))
            continue
        raw_id = row.get("video_id")
        video_id = raw_id if isinstance(raw_id, str) else None
        def add(code: str, message: str, prediction_index: int | None = None,
                severity: str = "error") -> None:
            report.issues.append(ValidationIssue(code, message, video_id, prediction_index, severity))
        try:
            _assert_finite_tree(row)
        except ContractError as error:
            add("nonfinite", str(error))
        if not video_id or video_id not in index:
            add("unknown_video", f"row {row_number} video_id is missing or absent from index")
            metadata = None
        else:
            metadata = index[video_id]
            if video_id in seen_ids:
                add("duplicate_video", "a video may occur only once")
            seen_ids.add(video_id)
        try:
            ratio = parse_ratio(row.get("targetRatioWH"))
            if metadata is not None and ratio != metadata.target_ratio:
                add("ratio_mismatch", "targetRatioWH must equal the index pair exactly")
        except ContractError as error:
            add("ratio", str(error))
        if "model_size_mb" in row:
            size_present += 1
            try:
                size = row["model_size_mb"]
                size_coefficient(size)
                sizes.append(float(size))
                if actual_model_size_mb is not None and not math.isclose(
                    size, actual_model_size_mb, rel_tol=0, abs_tol=1e-9
                ):
                    add("size_mismatch", "declared size differs from measured weight file size")
            except ContractError as error:
                add("model_size", str(error))
        elif stage == "final":
            add("model_size_missing", "model_size_mb is required for final stage")
        predictions = row.get("predictions")
        if not isinstance(predictions, list):
            add("predictions_type", "predictions must be a list; [] is legal")
            continue
        report.prediction_count += len(predictions)
        seen_frames: set[int] = set()
        previous_frame: int | None = None
        for position, prediction in enumerate(predictions):
            if metadata is not None:
                for error in validate_prediction(prediction, metadata):
                    add("invalid_prediction", error, position)
            frame = prediction.get("frame") if isinstance(prediction, Mapping) else None
            if type(frame) is int:
                if frame in seen_frames:
                    add("duplicate_frame", f"frame {frame} repeated", position)
                if previous_frame is not None and frame < previous_frame:
                    add("unsorted_frames", "frames should be in increasing order", position,
                        "error" if require_sorted else "warning")
                previous_frame = frame
                seen_frames.add(frame)
    for missing in sorted(set(index) - seen_ids):
        report.issues.append(ValidationIssue("missing_video", "index video missing from submission", missing))
    if sizes:
        report.model_size_mb = sizes[0]
        if any(size != sizes[0] for size in sizes):
            report.issues.append(ValidationIssue("inconsistent_size", "all video rows must use identical model_size_mb"))
    if 0 < size_present < report.video_count:
        report.issues.append(ValidationIssue("partial_size", "include model_size_mb consistently on every video row"))
    return report


def write_submission(path: str | Path, rows: Sequence[Mapping[str, Any]],
                     index: Mapping[str, VideoMetadata], *, stage: str = "preliminary",
                     actual_model_size_mb: float | None = None) -> ValidationReport:
    """Validate before atomically writing; refuse unsorted/duplicate submissions."""
    # The measured size is authoritative at the exporter boundary. Populate an
    # omitted field for convenience; explicit conflicting declarations remain an
    # error and are never silently corrected.
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        if actual_model_size_mb is not None and "model_size_mb" not in copied:
            copied["model_size_mb"] = actual_model_size_mb
        output_rows.append(copied)
    report = validate_submission(output_rows, index, stage=stage,
                                 actual_model_size_mb=actual_model_size_mb,
                                 require_sorted=True)
    report.raise_for_errors()
    write_jsonl(path, output_rows)
    return report


def center_crop(width: int, height: int, target_ratio: Sequence[float]) -> list[float]:
    """Largest legal centered crop, represented in original pixel coordinates."""
    if not finite_number(width) or not finite_number(height) or width <= 0 or height <= 0:
        raise ContractError("source dimensions must be finite and positive")
    rw, rh = parse_ratio(target_ratio)
    crop_width = min(float(width), float(height) / (rh / rw))
    crop_height = crop_width * (rh / rw)
    # Roundoff may make a boundary height exceed source by an ULP.
    if crop_height > height:
        crop_width = math.nextafter(crop_width, 0.0)
        crop_height = crop_width * (rh / rw)
    return [(width - crop_width) / 2, (height - crop_height) / 2, crop_width]


@dataclass(frozen=True)
class LetterboxTransform:
    """Record exact resized dimensions and integer padding, including rounding.

    Boxes are [x,y,w,h] in continuous pixel-edge coordinates. No silent clipping
    occurs; a padded-area prediction may map out of bounds and must be rejected or
    projected by the spatial policy. This transform does not apply video rotation.
    """

    source_width: int
    source_height: int
    target_width: int
    target_height: int
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int

    @classmethod
    def from_sizes(cls, source_width: int, source_height: int,
                   target_width: int, target_height: int) -> "LetterboxTransform":
        dimensions = (source_width, source_height, target_width, target_height)
        if any(type(v) is not int or v <= 0 for v in dimensions):
            raise ContractError("letterbox dimensions must be positive integers")
        scale = min(target_width / source_width, target_height / source_height)
        resized_width = min(target_width, max(1, round(source_width * scale)))
        resized_height = min(target_height, max(1, round(source_height * scale)))
        return cls(*dimensions, resized_width, resized_height,
                   (target_width - resized_width) // 2, (target_height - resized_height) // 2)

    @property
    def scale_x(self) -> float:
        return self.resized_width / self.source_width

    @property
    def scale_y(self) -> float:
        return self.resized_height / self.source_height

    def forward_point(self, point: Sequence[float]) -> tuple[float, float]:
        if len(point) != 2 or not all(finite_number(v) for v in point):
            raise ContractError("transform expects a finite [x,y] point")
        return (float(point[0]) * self.scale_x + self.pad_left,
                float(point[1]) * self.scale_y + self.pad_top)

    def inverse_point(self, point: Sequence[float]) -> tuple[float, float]:
        if len(point) != 2 or not all(finite_number(v) for v in point):
            raise ContractError("transform expects a finite [x,y] point")
        return ((float(point[0]) - self.pad_left) / self.scale_x,
                (float(point[1]) - self.pad_top) / self.scale_y)

    @staticmethod
    def _box(box: Sequence[float]) -> tuple[float, ...]:
        if len(box) != 4 or not all(finite_number(v) for v in box) or box[2] <= 0 or box[3] <= 0:
            raise ContractError("transform expects a finite [x,y,w,h] with w,h>0")
        return tuple(float(v) for v in box)

    def forward_box(self, box: Sequence[float]) -> tuple[float, ...]:
        x, y, width, height = self._box(box)
        return (x * self.scale_x + self.pad_left, y * self.scale_y + self.pad_top,
                width * self.scale_x, height * self.scale_y)

    def inverse_box(self, box: Sequence[float]) -> tuple[float, ...]:
        x, y, width, height = self._box(box)
        return ((x - self.pad_left) / self.scale_x, (y - self.pad_top) / self.scale_y,
                width / self.scale_x, height / self.scale_y)


def inverse_letterbox_box(box: Sequence[float], transform: LetterboxTransform) -> tuple[float, ...]:
    """Functional alias useful to inference code and notebooks."""
    return transform.inverse_box(box)


def forward_letterbox_box(box: Sequence[float], transform: LetterboxTransform) -> tuple[float, ...]:
    """Functional alias useful to inference code and notebooks."""
    return transform.forward_box(box)


def audit_weights(paths: Iterable[str | Path]) -> dict[str, Any]:
    """Measure explicit, unpacked inference weight files; never infer from params.

    Callers supply every file loaded during inference, including auxiliaries.
    Archives must be unpacked first. Distinct duplicate formats are counted; no
    content deduplication exemption is assumed. Listing one path twice is an error.
    """
    files: list[dict[str, Any]] = []
    seen: set[Path] = set()
    total = 0
    for raw_path in paths:
        path = Path(raw_path).resolve(strict=True)
        if not path.is_file():
            raise ContractError(f"weight path is not a regular file: {path}")
        if path in seen:
            raise ContractError(f"weight file supplied twice: {path}")
        seen.add(path)
        if path.suffix.lower() in {".zip", ".gz", ".bz2", ".xz", ".tar", ".7z", ".rar"}:
            raise ContractError("supply unpacked model weight files, not compressed archives")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        size = path.stat().st_size
        if size <= 0:
            raise ContractError(f"weight file is empty: {path}")
        files.append({"path": str(path), "bytes": size, "sha256": digest.hexdigest()})
        total += size
    if not files:
        raise ContractError("at least one actual weight file is required for a size audit")
    decimal_mb, binary_mib = total / 1_000_000, total / 1_048_576
    def coefficient_or_none(size: float) -> float | None:
        return size_coefficient(size) if 0 < size <= 9216 else None
    return {"files": files, "total_bytes": total, "decimal_mb": decimal_mb,
            # binary_mib is the unambiguous name; binary_mb is retained as a
            # convenience alias because the competition document says "MB".
            "binary_mib": binary_mib, "binary_mb": binary_mib,
            "coefficient_decimal_mb": coefficient_or_none(decimal_mb),
            "coefficient_binary_mb": coefficient_or_none(binary_mib),
            "mb_base_status": "official_MB_byte_base_unconfirmed",
            "measurement": "sum_of_explicit_unpacked_inference_weight_files"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate JSONL against enriched metadata index")
    validate.add_argument("--index", required=True)
    validate.add_argument("--submission", required=True)
    validate.add_argument("--stage", choices=["preliminary", "initial", "prelim", "final", "replay", "semifinal"], default="preliminary")
    validate.add_argument("--require-sorted", action="store_true")
    validate.add_argument("--weights", nargs="+")
    validate.add_argument("--mb-base", choices=["decimal", "binary"])
    size = commands.add_parser("size", help="audit every actual inference weight file")
    size.add_argument("weights", nargs="+")
    args = parser.parse_args(argv)
    try:
        if args.command == "size":
            result = audit_weights(args.weights)
            status = 0
        else:
            actual_size = None
            if args.weights:
                if args.mb_base is None:
                    parser.error("--weights requires explicit --mb-base because official MB base is unresolved")
                audit = audit_weights(args.weights)
                actual_size = audit["decimal_mb" if args.mb_base == "decimal" else "binary_mib"]
            report = validate_submission(load_jsonl(args.submission), load_index(args.index),
                                         stage=args.stage, actual_model_size_mb=actual_size,
                                         require_sorted=args.require_sorted)
            result = report.to_dict()
            status = 0 if report.valid else 2
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
        return status
    except (ContractError, OSError) as error:
        print(json.dumps({"valid": False, "error": str(error)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
