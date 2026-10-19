"""Native-pixel crop observations for public R85 research inputs.

All boxes are continuous ``[x1, y1, x2, y2]`` in the original decoded image.
There is no normalized-coordinate detection, scale inference, inclusive-pixel
``+1``, coordinate rounding, or EXIF rotation. Public T5 GT values are preserved
as stored and evaluated with continuous IoU, matching R70's arithmetic rather
than the older rounded/inclusive benchmark protocol.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import io
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


NC = 129
OBSERVATION_VERSION = "r85.native_extent.v1"
MANIFEST_VERSION = "r85.public_crop_manifest.v1"
GT_CONVENTION = "table_xyxy_as_stored_continuous"
PUBLIC_TABLES = ("rv_train", "live_train", "rv_dev", "live_dev")


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def native_dimensions(W: Any, H: Any) -> tuple[int, int]:
    W, H = _integer(W, "W", 1), _integer(H, "H", 1)
    try:
        finite = math.isfinite(W) and math.isfinite(H)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError("native dimensions must be representable finite numbers")
    return W, H


def numeric_array(values: Any, name: str) -> np.ndarray:
    """Reject strings/bools before conversion; accept ordinary numeric ndarrays."""
    if isinstance(values, np.ndarray) and values.dtype.kind in "fiu":
        result = values.astype(np.float64, copy=False)
    else:
        objects = np.asarray(values, dtype=object)
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real)
               for v in objects.flat):
            raise ValueError(f"{name} must contain real numbers, excluding bool")
        try:
            result = objects.astype(np.float64)
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError(f"{name} has an unrepresentable number") from error
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite numbers")
    return result


def checked_ratio(ratio: Sequence[float]) -> tuple[float, float]:
    value = numeric_array(ratio, "ratio")
    if value.shape != (2,) or np.any(value <= 0):
        raise ValueError("ratio must contain exactly two positive numbers")
    rw, rh = map(float, value)
    aspect = rw / rh
    inverse = rh / rw
    if not math.isfinite(aspect) or aspect <= 0 or not math.isfinite(inverse) or inverse <= 0:
        raise ValueError("ratio is not representable")
    return rw, rh


def validate_boxes(boxes: Any, W: int | None = None, H: int | None = None) -> np.ndarray:
    """Validate an [N,4] array of positive-area native continuous xyxy boxes."""
    values = numeric_array(boxes, "boxes")
    if values.ndim != 2 or values.shape[1] != 4 or len(values) == 0:
        raise ValueError("boxes must have nonempty shape [N,4]")
    if np.any(values[:, :2] < 0) or np.any(values[:, 2:] <= values[:, :2]):
        raise ValueError("boxes require nonnegative origins and finite positive area")
    if (W is None) != (H is None):
        raise ValueError("both native W and H are required for bounds checks")
    if W is not None:
        W, H = native_dimensions(W, H)
        if np.any(values[:, 2] > W) or np.any(values[:, 3] > H):
            raise ValueError("box extends outside native image dimensions")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        areas = np.prod(values[:, 2:] - values[:, :2], axis=1)
    if not np.isfinite(areas).all() or np.any(areas <= 0):
        raise ValueError("box area must be finite and positive")
    return values


def candidate_boxes(W: int, H: int, ratio: Sequence[float], nc: int = NC) -> np.ndarray:
    """Return [nc,4] float64 max-window xyxy boxes, ordered along the free axis.

    A wider source moves in x; a taller source moves in y. Equal aspect ratios
    return nc identical full-image boxes so the model's candidate axis is stable.
    Coordinates stay fractional. Endpoint min guards only floating-point ULPs.
    """
    W, H = native_dimensions(W, H)
    nc = _integer(nc, "nc", 2)
    rw, rh = checked_ratio(ratio)
    target = rw / rh
    result = np.zeros((nc, 4), dtype=np.float64)
    if W / H > target:
        width = min(float(W), H * target)
        offsets = np.linspace(0.0, W - width, nc, dtype=np.float64)
        result[:, 0] = offsets
        result[:, 2] = np.minimum(offsets + width, W)
        result[:, 3] = H
    elif W / H < target:
        height = min(float(H), W / target)
        offsets = np.linspace(0.0, H - height, nc, dtype=np.float64)
        result[:, 1] = offsets
        result[:, 2] = W
        result[:, 3] = np.minimum(offsets + height, H)
    else:
        result[:] = [0.0, 0.0, W, H]
    return validate_boxes(result, W, H)


def mean_iou(boxes: Any, gt_xyxy: Any) -> np.ndarray:
    """Continuous IoU averaged across the human GT boxes, one value per candidate."""
    boxes, gt = validate_boxes(boxes), validate_boxes(gt_xyxy)
    a, b = boxes[:, None, :], gt[None, :, :]
    extent = np.maximum(0.0, np.minimum(a[..., 2:], b[..., 2:]) - np.maximum(a[..., :2], b[..., :2]))
    intersection = np.prod(extent, axis=-1)
    areas_a = np.prod(a[..., 2:] - a[..., :2], axis=-1)
    areas_b = np.prod(b[..., 2:] - b[..., :2], axis=-1)
    union = areas_a + areas_b - intersection
    if not np.isfinite(union).all() or np.any(union <= 0):
        raise ValueError("IoU union must be finite and positive")
    return (intersection / union).mean(axis=1)


def pool_grid_2x2(grid: np.ndarray, W: int, H: int, boxes: Any) -> np.ndarray:
    """Exact-area ROI pooling, returning [N,4*D] in TL/TR/BL/BR order.

    Each grid cell represents an equal-area native-image rectangle. Pooling
    uses cell/sub-rectangle overlap area, including fractional boundary cells.
    Positive-area sub-rectangles cannot become empty floor/ceil masks. For a
    re-encoded full crop grid, use W=grid.shape[1], H=grid.shape[0] and the single
    box [0,0,W,H]. The output retains float32 or float64 input precision.
    """
    W, H = native_dimensions(W, H)
    if not isinstance(grid, np.ndarray) or grid.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("grid must be a float32 or float64 ndarray")
    if grid.ndim != 3 or any(n <= 0 for n in grid.shape) or not np.isfinite(grid).all():
        raise ValueError("grid must have finite nonempty shape [fh,fw,D]")
    boxes = validate_boxes(boxes, W, H)
    fh, fw, depth = grid.shape
    # Extended-precision boundaries keep midpoint arithmetic meaningful even
    # when a valid float64 rectangle lies between adjacent representable values.
    b = boxes.astype(np.longdouble)
    mid = b[:, :2] + (b[:, 2:] - b[:, :2]) / 2
    left = np.stack((b[:, 0], mid[:, 0], b[:, 0], mid[:, 0]), axis=1).reshape(-1)
    right = np.stack((mid[:, 0], b[:, 2], mid[:, 0], b[:, 2]), axis=1).reshape(-1)
    top = np.stack((b[:, 1], b[:, 1], mid[:, 1], mid[:, 1]), axis=1).reshape(-1)
    bottom = np.stack((mid[:, 1], mid[:, 1], b[:, 3], b[:, 3]), axis=1).reshape(-1)
    gx = np.linspace(0, W, fw + 1, dtype=np.longdouble)
    gy = np.linspace(0, H, fh + 1, dtype=np.longdouble)
    wx = np.maximum(0, np.minimum(right[:, None], gx[None, 1:]) - np.maximum(left[:, None], gx[None, :-1]))
    wy = np.maximum(0, np.minimum(bottom[:, None], gy[None, 1:]) - np.maximum(top[:, None], gy[None, :-1]))
    sx, sy = wx.sum(axis=1, keepdims=True), wy.sum(axis=1, keepdims=True)
    if np.any(sx <= 0) or np.any(sy <= 0):
        raise ValueError("positive-area ROI has no representable pooling coverage")
    # Normalizing each axis before their product is equivalent to normalized
    # overlap area and avoids underflow for very small legal rectangles.
    wx, wy = (wx / sx).astype(np.float64), (wy / sy).astype(np.float64)
    weights = (wy[:, :, None] * wx[:, None, :]).reshape(-1, fh * fw)
    pooled = weights @ grid.reshape(fh * fw, depth).astype(np.float64)
    return pooled.reshape(len(boxes), 4 * depth).astype(grid.dtype)


def crop_extent(image: Image.Image, box: Sequence[float],
                output_size: tuple[int, int] | None = None,
                resample: str = "bicubic") -> Image.Image:
    """Resample an exact fractional native xyxy rectangle using PIL EXTENT.

    With output_size=None, dimensions are ceil(box width), ceil(box height).
    Only the output sampling grid is integer; source coordinates are unchanged.
    A caller may explicitly request a different output grid. No automatic crop
    enlargement, aspect-ratio correction, scale conversion, or EXIF transpose
    occurs. Feed the result through the same processor as the control arm.
    """
    if not isinstance(image, Image.Image):
        raise ValueError("image must be a PIL Image")
    extent = validate_boxes([box], *image.size)[0]
    if output_size is None:
        output_size = (math.ceil(extent[2] - extent[0]), math.ceil(extent[3] - extent[1]))
    if not isinstance(output_size, (list, tuple)) or len(output_size) != 2:
        raise ValueError("output_size must be a (width,height) pair")
    output_size = native_dimensions(*output_size)
    filters = {"nearest": Image.Resampling.NEAREST,
               "bilinear": Image.Resampling.BILINEAR,
               "bicubic": Image.Resampling.BICUBIC}
    if resample not in filters:
        raise ValueError("resample must be nearest, bilinear, or bicubic")
    return image.convert("RGB").transform(output_size, Image.Transform.EXTENT,
                                         tuple(map(float, extent)), resample=filters[resample])


def image_record(path: str | Path) -> dict[str, Any]:
    """Hash and decode the same file bytes; no rotation or resolution conversion."""
    path = Path(path).resolve(strict=True)
    payload = path.read_bytes()
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        W, H = native_dimensions(*image.size)
    return {"image_path": str(path), "image_sha256": hashlib.sha256(payload).hexdigest(),
            "W": W, "H": H}


def load_manifest_image(row: Mapping[str, Any]) -> Image.Image:
    """Verify frozen file identity and native size before yielding RGB pixels."""
    path = Path(row["image_path"])
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != row["image_sha256"]:
        raise ValueError(f"image SHA256 mismatch for {row.get('row_id', path)}")
    with Image.open(io.BytesIO(payload)) as original:
        original.load()
        if original.size != native_dimensions(row["W"], row["H"]):
            raise ValueError(f"native image size mismatch for {row.get('row_id', path)}")
        return original.convert("RGB")


def make_row_id(source: str, frame_id: int, ratio: Sequence[float]) -> str:
    """Identity depends on source/frame/ratio, never shard position or table order."""
    if not isinstance(source, str) or not source:
        raise ValueError("source must be a nonempty string")
    frame_id = _integer(frame_id, "frame_id")
    rw, rh = checked_ratio(ratio)
    fraction = Fraction(str(rw)) / Fraction(str(rh))
    key = json.dumps([source, frame_id, str(fraction.numerator), str(fraction.denominator)],
                     ensure_ascii=False, separators=(",", ":"))
    return "r85_" + hashlib.sha256(key.encode("utf-8")).hexdigest()


def validate_manifest_rows(rows: Sequence[Mapping[str, Any]], *, verify_images: bool = True) -> None:
    """Reject duplicate identities, source leakage, altered images, or illegal GT."""
    if not rows:
        raise ValueError("manifest must contain at least one public labeled row")
    seen, source_splits = set(), {}
    table_identities = {}
    checked_images = set()
    for row in rows:
        if not isinstance(row, Mapping) or row.get("schema_version") != MANIFEST_VERSION:
            raise ValueError("manifest row has an unsupported schema_version")
        table = row.get("table")
        if table not in PUBLIC_TABLES:
            raise ValueError(f"table is outside the public whitelist: {table!r}")
        dataset, split = table.split("_", 1)
        if row.get("dataset") != dataset or row.get("split") != split:
            raise ValueError("dataset/split does not match the public table")
        vid = row.get("video_id")
        if not isinstance(vid, str) or not vid or vid in {".", ".."} or "/" in vid or "\\" in vid:
            raise ValueError("video_id must be a nonempty source directory component")
        source = row.get("source")
        if source != f"{dataset}:{row.get('video_id')}":
            raise ValueError("source must match dataset:video_id")
        if source in source_splits and source_splits[source] != split:
            raise ValueError(f"source train/dev leakage: {source}")
        source_splits[source] = split
        frame_id = _integer(row.get("frame_id"), "frame_id")
        if _integer(row.get("kf"), "kf") != frame_id:
            raise ValueError("kf must equal frame_id")
        identity = make_row_id(source, frame_id, row["ratio"])
        if row.get("row_id") != identity:
            raise ValueError("row_id does not match source/frame/ratio")
        if identity in seen:
            raise ValueError(f"duplicate row_id: {identity}")
        seen.add(identity)
        W, H = native_dimensions(row["W"], row["H"])
        validate_boxes(row["gt_xyxy"], W, H)
        if row.get("gt_coordinate_convention") != GT_CONVENTION:
            raise ValueError("GT coordinate convention must be explicit")
        if row.get("coordinate_scale") != "native_pixels" or row.get("exif_transpose") is not False:
            raise ValueError("manifest must use native_pixels with no EXIF transpose")
        if row.get("label_source") != "public_human_gt_from_trusted_T5_table":
            raise ValueError("manifest labels must declare the trusted public T5 source")
        for name in ("image_sha256", "table_sha256"):
            value = row.get(name)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"{name} must be a lowercase SHA256")
        for name in ("image_path", "table_path"):
            if not isinstance(row.get(name), str) or not Path(row[name]).is_absolute():
                raise ValueError(f"{name} must be an explicit absolute path")
        if Path(row["table_path"]).name != f"{table}.pkl":
            raise ValueError("table_path must name the whitelisted source table")
        table_identity = (row["table_path"], row["table_sha256"])
        if table in table_identities and table_identities[table] != table_identity:
            raise ValueError("source table path/SHA256 must be consistent across rows")
        table_identities[table] = table_identity
        image_key = (row["image_path"], row["image_sha256"], W, H)
        if verify_images and image_key not in checked_images:
            load_manifest_image(row)
            checked_images.add(image_key)
