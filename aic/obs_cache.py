"""Per-frame spatial observation cache (B0 YuNet path + COCO detections).

`ObservedFacePath` replicates `SpatialPath('true_face_smooth')` step by step
but also records every observation it used: reset flag, gradient-saliency
centre, all YuNet faces (small-image pixels) and the raw pre-EMA point.  The
B0 crop it emits must equal the frozen B0 package bit for bit, so trajectory
experiments can reuse the exact same observations without re-decoding.
"""
from pathlib import Path
import numpy as np
from .spatial import gradient_saliency_center, place_crop


class ObservedFacePath:
    def __init__(self, ratio, detector_path, alpha=.25):
        import cv2
        if not Path(detector_path).is_file(): raise ValueError('face weights required')
        self.ratio, self.alpha = ratio, alpha
        self.previous_image = self.center = self.face = None
        self.detector = cv2.FaceDetectorYN.create(str(detector_path), "", (320, 320), .8, .3, 5000)

    def step(self, rgb):
        import cv2
        h, w = rgb.shape[:2]; thumb = cv2.resize(rgb, (64, 36)).astype(np.float32) / 255
        reset = self.previous_image is None or float(np.abs(thumb - self.previous_image).mean()) > .25
        self.previous_image = thumb
        if reset: self.center = None; self.face = None
        scale = min(1, 320 / w); small = cv2.resize(rgb, (int(round(w * scale)), int(round(h * scale))))
        cx, cy, conf = gradient_saliency_center(small)
        weight = np.clip((conf - .08) / .92, 0, 1)
        point = np.array([.5 + weight * (cx - .5), .5 + weight * (cy - .5)])
        self.detector.setInputSize((small.shape[1], small.shape[0]))
        faces = self.detector.detect(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))[1]
        faces = np.zeros((0, 15), np.float32) if faces is None else faces.astype(np.float32)
        chosen = -1
        if len(faces):
            centers = (faces[:, :2] + faces[:, 2:4] / 2) / np.array([small.shape[1], small.shape[0]])
            importance = faces[:, 2] * faces[:, 3] * faces[:, -1]
            if self.face is not None:
                importance = importance / (1 + 8 * np.linalg.norm(centers - self.face, axis=1))
            chosen = int(np.argmax(importance)); point = centers[chosen]; self.face = point
        raw = np.asarray(point, np.float64).copy()
        if self.center is not None: point = self.alpha * point + (1 - self.alpha) * self.center
        self.center = np.clip(point, 0, 1)
        crop = place_crop(w, h, self.ratio, self.center[0] * w, self.center[1] * h)
        return crop, {'reset': reset, 'sal': (cx, cy, conf), 'faces': faces, 'chosen': chosen, 'raw': raw,
                      'small_wh': (small.shape[1], small.shape[0])}


def ragged(rows, width, dtype=np.float32):
    offsets = np.zeros(len(rows) + 1, np.int64); offsets[1:] = np.cumsum([len(r) for r in rows])
    data = np.concatenate([np.asarray(r, dtype).reshape(-1, width) for r in rows]) if offsets[-1] else np.zeros((0, width), dtype)
    return data, offsets


def save_cache(dest, W, H, ratio, crops, obs, dets):
    faces, face_off = ragged([o['faces'] for o in obs], 15)
    det_rows = [np.column_stack([b, s, l]) if len(b) else np.zeros((0, 6)) for b, s, l in dets]
    det, det_off = ragged(det_rows, 6)
    np.savez_compressed(dest, W=W, H=H, ratio=np.asarray(ratio, np.float64), b0=np.asarray(crops, np.float64),
                        reset=np.array([o['reset'] for o in obs]), sal=np.array([o['sal'] for o in obs], np.float64),
                        raw=np.array([o['raw'] for o in obs]), chosen=np.array([o['chosen'] for o in obs]),
                        small_wh=np.array(obs[0]['small_wh']), faces=faces, face_off=face_off, det=det, det_off=det_off)
