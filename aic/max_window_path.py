"""Max-window trajectory policies over cached observations (P1a/P1b/P2).

The crop size is always the largest legal window; only the free axis moves
(x when the window is full height, y when full width).  All policies consume
the same cached per-frame observation `raw` (B0's pre-EMA point: chosen YuNet
face centre, else confidence-weighted gradient saliency) and B0's shot resets,
so the only changed factor is the path.

`dp_path` solves, per shot, a 1-D Viterbi over a grid of legal window offsets:
    cost = sum_t |o_t - c_t| * w_t  +  lam * sum_t |c_t - c_{t-1}|  (+ jump)
where o_t is the observation clipped to its reachable range and w_t is 1 for
face frames, `sal_w` for saliency-only frames.  L1 motion cost gives
piecewise-constant (static camera) paths with sharp re-framing when the
subject really moves, unlike EMA which always lags.
"""
import numpy as np
from .contract import parse_ratio


def geometry(W, H, ratio):
    rw, rh = parse_ratio(ratio)
    w = min(float(W), float(H) * rw / rh); h = w * rh / rw
    axis = 0 if w < W - 1e-9 else (1 if h < H - 1e-9 else None)
    return w, h, axis


def to_crops(W, H, ratio, free_offsets):
    """free_offsets: top-left along the free axis (pixels); returns [x,y,w] list."""
    w, h, axis = geometry(W, H, ratio)
    out = []
    for o in free_offsets:
        if axis == 0: out.append([float(o), 0.0, w])
        elif axis == 1: out.append([0.0, float(o), w])
        else: out.append([0.0, 0.0, w])
    return out


def centre_to_offset(c, W, H, ratio):
    w, h, axis = geometry(W, H, ratio)
    if axis is None: return np.zeros_like(c)
    L, s = (W, w) if axis == 0 else (H, h)
    return np.clip(c * L - s / 2, 0, L - s)


def shots(reset):
    idx = list(np.flatnonzero(reset)) + [len(reset)]
    if idx[0] != 0: idx = [0] + idx
    return [(a, b) for a, b in zip(idx[:-1], idx[1:]) if b > a]


def dp_path(obs_off, weight, reset, span, lam=4.0, grid=64):
    """obs_off: observed free-axis offset per frame (pixels); span: max offset.

    lam is the motion cost in units of 'pixels of observation error per pixel
    moved' scaled to the free-axis span; it is fixed on dev sources.
    """
    n = len(obs_off); out = np.zeros(n)
    if span <= 1e-9: return out
    g = np.linspace(0, span, grid); move = lam * np.abs(g[:, None] - g[None, :]) / span
    for a, b in shots(reset):
        o = obs_off[a:b] / span; wt = weight[a:b]; gn = g / span
        cost = wt[0] * np.abs(gn - o[0]); back = np.zeros((b - a, grid), np.int32)
        for t in range(1, b - a):
            tot = cost[:, None] + move            # prev x next
            back[t] = np.argmin(tot, axis=0); cost = tot[back[t], np.arange(grid)] + wt[t] * np.abs(gn - o[t])
        k = int(np.argmin(cost)); path = np.zeros(b - a, np.int64)
        for t in range(b - a - 1, -1, -1):
            path[t] = k; k = back[t][k] if t else k
        out[a:b] = g[path]
    return out


def shot_static(obs_off, weight, reset):
    """One fixed window per shot: weighted median of observations."""
    out = np.zeros(len(obs_off))
    for a, b in shots(reset):
        o, w = obs_off[a:b], weight[a:b]; order = np.argsort(o); cw = np.cumsum(w[order])
        out[a:b] = o[order][np.searchsorted(cw, cw[-1] / 2)]
    return out


def cache_arrays(z):
    """Observation arrays shared by every policy (from an OBS cache npz)."""
    W, H = int(z['W']), int(z['H']); ratio = z['ratio'].tolist(); raw = z['raw']; reset = z['reset'].astype(bool)
    has_face = z['chosen'] >= 0
    return W, H, ratio, raw, reset, has_face


def ema_offsets(obs_centre, reset, W, H, ratio, alpha=.25):
    """B0's EMA on the normalised free-axis centre, then clipped placement."""
    out = np.zeros(len(obs_centre)); prev = None
    for t, c in enumerate(obs_centre):
        if reset[t]: prev = None
        p = c if prev is None else alpha * c + (1 - alpha) * prev
        prev = float(np.clip(p, 0, 1)); out[t] = prev
    return centre_to_offset(out, W, H, ratio)


# COCO category priors (fixed a priori, not fitted): people dominate, animals
# next, sports objects / vehicles weaker, everything else weak.
def class_prior(label):
    label = int(label)
    if label == 1: return 1.0
    if 16 <= label <= 25: return .8
    if 34 <= label <= 43: return .5
    if 2 <= label <= 9: return .5
    return .3


def frame_boxes(z, t, det_score=.5, face_weight=1.5):
    """(extent_lo, extent_hi, weight) along the free axis for frame t, original pixels."""
    W, H = int(z['W']), int(z['H']); _, _, axis = geometry(W, H, z['ratio'].tolist())
    d = z['det'][z['det_off'][t]:z['det_off'][t + 1]]; d = d[d[:, 4] >= det_score]
    sw, sh = z['small_wh']; f = z['faces'][z['face_off'][t]:z['face_off'][t + 1]]
    boxes = []
    for x1, y1, x2, y2, s, l in d:
        a = (x2 - x1) * (y2 - y1) / (W * H)
        if not .002 <= a <= .9: continue
        boxes.append((x1, y1, x2, y2, class_prior(l) * s * np.sqrt(a)))
    for fx, fy, fw, fh in f[:, :4] if len(f) else []:
        x1, y1, x2, y2 = fx * W / sw, fy * H / sh, (fx + fw) * W / sw, (fy + fh) * H / sh
        a = (x2 - x1) * (y2 - y1) / (W * H); boxes.append((x1, y1, x2, y2, face_weight * np.sqrt(max(a, 1e-6))))
    if axis is None or not boxes: return np.zeros((0, 3))
    b = np.asarray(boxes)
    return b[:, [0, 2, 4]] if axis == 0 else b[:, [1, 3, 4]]


def coverage_centre(z, t, b0_raw_c, keep_margin=1.1, min_mass=.02, grid=64):
    """Free-axis centre (normalised) maximising weighted box coverage of the max window.

    Falls back to B0's raw observation when there is little subject mass or the
    best window does not clearly beat the window placed at B0's observation.
    """
    W, H = int(z['W']), int(z['H']); w, h, axis = geometry(W, H, z['ratio'].tolist())
    if axis is None: return b0_raw_c, 'full'
    L, s = (W, w) if axis == 0 else (H, h); ext = frame_boxes(z, t)
    if len(ext) == 0 or ext[:, 2].sum() < min_mass: return b0_raw_c, 'no_subject'
    lo, hi, wt = ext.T; length = np.maximum(hi - lo, 1.)
    def score(o): return float((wt * np.clip(np.minimum(hi, o + s) - np.maximum(lo, o), 0, None) / length).sum())
    o_b0 = float(np.clip(b0_raw_c * L - s / 2, 0, L - s)); grid_o = np.append(np.linspace(0, L - s, grid), o_b0)
    sc = np.array([score(o) for o in grid_o]); best = int(np.argmax(sc))
    if sc[best] <= keep_margin * sc[-1] + 1e-9: return b0_raw_c, 'keep_b0'
    # centre of the plateau of best offsets (ties are common for small boxes)
    top = grid_o[sc >= sc[best] - 1e-9]; o = float((top.min() + top.max()) / 2)
    return (o + s / 2) / L, 'coverage'


def region_fit_centre(subject, context, axis, win, mode='fit'):
    """Deterministic region -> normalised free-axis window centre (T1).

    subject/context: [x1,y1,x2,y2] normalised; win: window length / frame length.
    fit: keep the whole context if it fits, as close to the subject centre as
    possible; else keep the whole subject, as close to the context centre as
    possible; else centre on the subject.  'subject'/'context' = box centre only
    (diagnostic mappings).
    """
    s_lo, s_hi = subject[axis], subject[axis + 2]; c_lo, c_hi = context[axis], context[axis + 2]
    if mode == 'subject': return (s_lo + s_hi) / 2
    if mode == 'context': return (c_lo + c_hi) / 2
    for (lo, hi), pref in (((c_lo, c_hi), (s_lo + s_hi) / 2), ((s_lo, s_hi), (c_lo + c_hi) / 2)):
        if hi - lo <= win + 1e-9:
            o = float(np.clip(np.clip(pref - win / 2, hi - win, lo), 0, 1 - win)); return o + win / 2
    return (s_lo + s_hi) / 2


def region_points(regions, axis, win, mode='fit'):
    """Turn region replies into the point format consumed by qwen_centres."""
    out = []
    for r in regions:
        if r is None: out.append(None); continue
        c = region_fit_centre(r[0], r[1], axis, win, mode); p = [None, None, bool(r[2])]; p[axis] = c; out.append(p)
    return out


def qwen_centres_interp(points, keyframes, reset, b0_raw_c, axis):
    """qwen_centres, but between two valid keyframes of the same shot the centre
    moves linearly from one point to the next instead of holding the first.

    Spans ending at an invalid keyframe, at a shot cut or at the video end keep
    the hold behaviour, so every frame whose span has no valid right end is
    identical to qwen_centres.
    """
    out, src = qwen_centres(points, keyframes, reset, b0_raw_c, axis)
    shot = np.cumsum(np.asarray(reset, bool)); kf = sorted(zip(keyframes, points))
    ok = lambda p: p is not None and not p[2]
    for (k0, p0), (k1, p1) in zip(kf[:-1], kf[1:]):
        if k1 <= k0 + 1 or not (ok(p0) and ok(p1)) or k1 >= len(reset) or shot[k0] != shot[k1]: continue
        t = np.arange(k0, k1); out[k0:k1] = p0[axis] + (p1[axis] - p0[axis]) * (t - k0) / (k1 - k0)
    return out, src


def qwen_centres(points, keyframes, reset, b0_raw_c, axis):
    """Hold each keyframe's subject point within its shot until the next keyframe.

    Failed/uncertain keyframes fall back to B0's raw observation for that span.
    """
    n = len(reset); out = np.array(b0_raw_c, float).copy(); src = np.zeros(n, '<U10'); src[:] = 'b0'
    kf = sorted(zip(keyframes, points)); j = -1; cur = None
    for t in range(n):
        if reset[t]: cur = None
        while j + 1 < len(kf) and kf[j + 1][0] <= t:
            j += 1; p = kf[j][1]; cur = None if (p is None or p[2]) else p[axis]
        if cur is not None: out[t] = cur; src[t] = 'qwen'
    return out, src
