"""Contact sheets with frame index + real timestamp burned in."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def _label(img: np.ndarray, text: str) -> np.ndarray:
    import cv2

    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(out, text, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_boxes(img: np.ndarray, boxes_norm: np.ndarray, colors: Optional[Sequence] = None) -> np.ndarray:
    """boxes_norm [K,4] xywh in [0,1] of the source frame."""
    import cv2

    out = img.copy()
    h, w = out.shape[:2]
    palette = colors or [(255, 60, 60), (60, 255, 60), (60, 120, 255), (255, 220, 0), (255, 0, 255), (0, 255, 255)]
    th = max(2, round(max(h, w) / 160))  # stays visible after the contact sheet downscales to 320 px tiles
    for k, b in enumerate(np.asarray(boxes_norm).reshape(-1, 4)):
        if not np.isfinite(b).all():
            continue
        x0, y0 = int(round(b[0] * w)), int(round(b[1] * h))
        x1, y1 = int(round((b[0] + b[2]) * w)), int(round((b[1] + b[3]) * h))
        cv2.rectangle(out, (x0, y0), (max(x0, x1 - 1), max(y0, y1 - 1)), palette[k % len(palette)], th)
    return out


def overlay_map(img: np.ndarray, m: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    import cv2

    heat = cv2.resize((np.clip(m, 0, 1) * 255).astype(np.uint8), (img.shape[1], img.shape[0]))
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)[:, :, ::-1]
    return (img * (1 - alpha) + heat * alpha).astype(np.uint8)


def overlay_mask(img: np.ndarray, m: np.ndarray, color=(255, 0, 0)) -> np.ndarray:
    import cv2

    mk = cv2.resize(m.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    out = img.copy()
    out[mk] = (0.5 * out[mk] + 0.5 * np.asarray(color)).astype(np.uint8)
    return out


def grid(tiles: list[np.ndarray], cols: int = 4, tile_w: int = 320) -> np.ndarray:
    import cv2

    rs = []
    for t in tiles:
        s = tile_w / t.shape[1]
        rs.append(cv2.resize(t, (tile_w, max(1, int(round(t.shape[0] * s))))))
    th = max(t.shape[0] for t in rs)
    rs = [np.pad(t, ((0, th - t.shape[0]), (0, 0), (0, 0))) for t in rs]
    while len(rs) % cols:
        rs.append(np.zeros_like(rs[0]))
    rows = [np.concatenate(rs[i:i + cols], 1) for i in range(0, len(rs), cols)]
    return np.concatenate(rows, 0)


def save(img: np.ndarray, path: str | Path) -> None:
    import cv2

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])


def timeline_strip(times: np.ndarray, value: np.ndarray, valid: np.ndarray, width: int = 1280,
                   height: int = 60, marks: Optional[np.ndarray] = None) -> np.ndarray:
    """Bar plot of a 1D label over real time; invalid positions drawn grey."""
    import cv2

    img = np.full((height, width, 3), 30, np.uint8)
    if len(times) == 0:
        return img
    t0, t1 = float(times[0]), float(times[-1]) + 1e-6
    vmax = float(np.nanmax(np.where(valid, value, np.nan))) if valid.any() else 1.0
    vmax = vmax if vmax > 0 else 1.0
    for t, v, ok in zip(times, value, valid):
        x = int((t - t0) / (t1 - t0) * (width - 1))
        if ok:
            hh = int(v / vmax * (height - 14))
            cv2.line(img, (x, height - 1), (x, height - 1 - hh), (80, 200, 255), 1)
        else:
            cv2.line(img, (x, height - 1), (x, height - 6), (110, 110, 110), 1)
    if marks is not None:
        for t in marks:
            x = int((t - t0) / (t1 - t0) * (width - 1))
            cv2.line(img, (x, 0), (x, 10), (255, 80, 80), 2)
    cv2.putText(img, f"{t0:.1f}s", (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    cv2.putText(img, f"{t1:.1f}s", (width - 50, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    return img


label = _label
