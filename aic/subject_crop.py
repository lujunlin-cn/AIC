"""Subject-aware, variable-scale crop path (exploratory; no AIC crop-scale GT).

Observations: COCO detector boxes (person weighted highest) plus YuNet faces.
The target window encloses the primary subject group with a fixed margin; its
width is clamped to [min_scale, 1] x the largest legal window.  Center and
scale are EMA-smoothed and reset on the same shot-change rule as SpatialPath.
All parameters are fixed a priori, not fitted to any crop GT.
"""
from pathlib import Path
import numpy as np
from .spatial import gradient_saliency_center, legal_widths

PERSON = 1
# COCO ids treated as possible subjects besides person: animals, vehicles, sports gear.
SUBJECT_CLASSES = {PERSON, 2, 3, 4, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 37, 38, 39, 40, 41, 42, 43}


class BatchDetector:
    """torchvision Faster R-CNN MobileNetV3-FPN (COCO), batched on one device."""
    def __init__(self, weights_path, device="cuda", min_size=320, max_size=640, score=.5):
        import torch
        from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
        self.torch, self.device, self.score = torch, device, score
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None, num_classes=91,
                                                  min_size=min_size, max_size=max_size)
        model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        self.model = model.to(device).eval()
        self.parameters = sum(p.numel() for p in model.parameters())

    def __call__(self, frames):
        torch = self.torch
        with torch.inference_mode():
            batch = [torch.from_numpy(f).to(self.device).permute(2, 0, 1).float().div_(255) for f in frames]
            out = self.model(batch)
        result = []
        for o in out:
            keep = o["scores"] >= self.score
            result.append((o["boxes"][keep].cpu().numpy(), o["scores"][keep].cpu().numpy(), o["labels"][keep].cpu().numpy()))
        return result


class SubjectPath:
    def __init__(self, ratio, face_path, min_scale=.6, margin=1.25, alpha=.25, scale_alpha=.1):
        import cv2
        self.rw, self.rh = float(ratio[0]), float(ratio[1]); self.ratio = ratio
        self.min_scale, self.margin, self.alpha, self.scale_alpha = min_scale, margin, alpha, scale_alpha
        if not face_path or not Path(face_path).is_file(): raise ValueError("face weights required")
        self.face = cv2.FaceDetectorYN.create(str(face_path), "", (320, 320), .8, .3, 5000)
        self.previous_image = None; self.center = None; self.scale = None; self.subject = None
        self.resets = self.observations = self.subject_frames = 0

    def _faces(self, rgb):
        import cv2
        h, w = rgb.shape[:2]; s = min(1, 320 / w)
        small = cv2.resize(rgb, (int(round(w * s)), int(round(h * s))))
        self.face.setInputSize((small.shape[1], small.shape[0]))
        faces = self.face.detect(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))[1]
        if faces is None or not len(faces): return np.zeros((0, 4)), np.zeros(0)
        boxes = faces[:, :4] / s; boxes[:, 2:] += boxes[:, :2]
        return boxes, faces[:, -1]

    def _subject(self, w, h, dets, faces):
        """Return subject group box [x0,y0,x1,y1] in pixels, or None."""
        boxes, scores, labels = dets
        cand = [];
        for b, sc, lb in zip(boxes, scores, labels):
            if int(lb) not in SUBJECT_CLASSES: continue
            area = max(0., (b[2] - b[0]) * (b[3] - b[1])) / (w * h)
            if area < .002 or area > .9: continue
            cand.append((np.sqrt(area) * sc * (1. if int(lb) == PERSON else .6), b))
        fb, fs = faces
        for b, sc in zip(fb, fs):
            area = max(0., (b[2] - b[0]) * (b[3] - b[1])) / (w * h)
            # A face is a strong subject cue; expand to a head-and-shoulders box.
            bw, bh = b[2] - b[0], b[3] - b[1]
            cand.append((np.sqrt(area) * sc * 1.5, np.array([b[0] - bw, b[1] - .5 * bh, b[2] + bw, b[3] + 2 * bh])))
        if not cand: return None
        if self.subject is not None:
            # Temporal association: prefer candidates near the previous subject.
            pc = (self.subject[:2] + self.subject[2:]) / 2
            cand = [(s / (1 + 4 * np.linalg.norm(((b[:2] + b[2:]) / 2 - pc) / [w, h])), b) for s, b in cand]
        cand.sort(key=lambda x: -x[0]); top = cand[0][0]
        group = [b for s, b in cand[:3] if s >= .6 * top]
        g = np.array([min(b[0] for b in group), min(b[1] for b in group), max(b[2] for b in group), max(b[3] for b in group)])
        return np.clip(g, 0, [w, h, w, h])

    def step(self, rgb, dets):
        import cv2
        h, w = rgb.shape[:2]
        thumb = cv2.resize(rgb, (64, 36)).astype(np.float32) / 255
        reset = self.previous_image is None or float(np.abs(thumb - self.previous_image).mean()) > .25
        self.previous_image = thumb
        if reset: self.center = self.scale = self.subject = None; self.resets += 1
        wmax = legal_widths(w, h, self.ratio, (1.,))[0]; hmax = wmax * self.rh / self.rw
        sub = self._subject(w, h, dets, self._faces(rgb))
        if sub is None:
            s = min(1, 320 / w); small = cv2.resize(rgb, (int(round(w * s)), int(round(h * s))))
            cx, cy, conf = gradient_saliency_center(small); k = np.clip((conf - .08) / .92, 0, 1)
            point = np.array([.5 + k * (cx - .5), .5 + k * (cy - .5)]); scale = 1.
        else:
            self.subject = sub; self.subject_frames += 1
            bw, bh = (sub[2] - sub[0]) * self.margin, (sub[3] - sub[1]) * self.margin
            need = max(bw, bh * self.rw / self.rh)
            scale = float(np.clip(need / wmax, self.min_scale, 1.))
            cy = (sub[1] + sub[3]) / 2
            win_h = scale * hmax
            if sub[3] - sub[1] > win_h:
                # Subject taller than the window: keep the top (head) with small headroom.
                cy = sub[1] - .05 * win_h + win_h / 2
            point = np.array([(sub[0] + sub[2]) / 2 / w, cy / h])
        if self.center is not None:
            point = self.alpha * point + (1 - self.alpha) * self.center
            scale = self.scale_alpha * scale + (1 - self.scale_alpha) * self.scale
        self.center, self.scale = np.clip(point, 0, 1), scale; self.observations += 1
        return self.crop(w, h, self.scale), self.crop(w, h, 1.), reset

    def crop(self, w, h, scale):
        wmax = legal_widths(w, h, self.ratio, (1.,))[0]
        cw = wmax * scale; ch = cw * self.rh / self.rw
        if ch > h: cw = np.nextafter(cw, 0); ch = cw * self.rh / self.rw
        x = min(max(self.center[0] * w - cw / 2, 0.), float(w) - cw)
        y = min(max(self.center[1] * h - ch / 2, 0.), float(h) - ch)
        return [float(x), float(y), float(cw)]


class ScalePath(SubjectPath):
    """Variable scale nested inside a given base window (the B0 face path).

    Only subjects whose center lies inside the base window are considered, so
    the output window is always a sub-window of the base crop; with no subject
    (or a subject needing the full window) the base crop is returned exactly.
    """
    def step(self, rgb, dets, base):
        import cv2
        h, w = rgb.shape[:2]
        thumb = cv2.resize(rgb, (64, 36)).astype(np.float32) / 255
        reset = self.previous_image is None or float(np.abs(thumb - self.previous_image).mean()) > .25
        self.previous_image = thumb
        if reset: self.center = self.scale = self.subject = None; self.resets += 1
        bx, by, bw = map(float, base); bh = bw * self.rh / self.rw
        boxes, scores, labels = dets; fb, fs = self._faces(rgb)
        inside = lambda b: bx <= (b[0] + b[2]) / 2 <= bx + bw and by <= (b[1] + b[3]) / 2 <= by + bh
        keep = [i for i, b in enumerate(boxes) if inside(b)]; fkeep = [i for i, b in enumerate(fb) if inside(b)]
        sub = self._subject(w, h, (boxes[keep], scores[keep], labels[keep]), (fb[fkeep], fs[fkeep]))
        if sub is None:
            scale, point = 1., np.array([bx + bw / 2, by + bh / 2])
        else:
            sub = np.clip(sub, [bx, by, bx, by], [bx + bw, by + bh, bx + bw, by + bh]); self.subject = sub; self.subject_frames += 1
            need = max((sub[2] - sub[0]) * self.margin, (sub[3] - sub[1]) * self.margin * self.rw / self.rh)
            scale = float(np.clip(need / bw, self.min_scale, 1.))
            win_h = scale * bh; cy = (sub[1] + sub[3]) / 2
            if sub[3] - sub[1] > win_h: cy = sub[1] - .05 * win_h + win_h / 2
            point = np.array([(sub[0] + sub[2]) / 2, cy])
        if self.center is not None:
            point = self.alpha * point + (1 - self.alpha) * self.center
            scale = self.scale_alpha * scale + (1 - self.scale_alpha) * self.scale
        self.center, self.scale = point, scale; self.observations += 1
        if scale >= .995: return [bx, by, bw], reset
        cw = bw * scale; ch = cw * self.rh / self.rw
        x = min(max(point[0] - cw / 2, bx), bx + bw - cw); y = min(max(point[1] - ch / 2, by), by + bh - ch)
        return [float(x), float(y), float(cw)], reset
