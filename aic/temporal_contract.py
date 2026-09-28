"""Temporal supervision contract v1 (P3): one anchor/label convention everywhere.

Convention (fixes the +0.99 s offset found in TIME_MAPPING_CHECK_V1):
  * an anchor's score describes the instant `t_anchor` (its clip is centred there);
  * a 2 s label clip k covers [2k, 2k+2) and is represented by its centre 2k+1;
  * training target for an anchor = linear interpolation of clip labels at
    the anchor time using clip CENTRES (not floor(t/2));
  * inference expands anchor scores to frames by interpolation at frame PTS
    (unchanged `expand_scores`), so train and inference now agree.

Label states (never collapse unlabelled into negatives):
  POS      reliable highlight (human score >= pos_thr)
  NEG      reliable non-highlight (human score <= neg_thr, same annotator pool)
  QUERY_IRRELEVANT  only negative for that query; masked for generic highlight
  UNLABELLED        masked
  TEACHER           weak label, separate channel, only after audit
"""
import numpy as np

POS, NEG, IGNORE = 1, 0, -1


def clip_centres(n_clips, clip_s=2.0):
    return np.arange(n_clips) * clip_s + clip_s / 2


def anchor_targets(anchor_t, clip_labels, clip_mask, clip_s=2.0):
    """Interpolated targets + mask at anchor times using clip centres.

    An anchor is supervised only if both neighbouring clip centres used by the
    interpolation are labelled (or the nearest one, at the ends).
    """
    lab = np.asarray(clip_labels, float); m = np.asarray(clip_mask, bool); c = clip_centres(len(lab), clip_s)
    t = np.asarray(anchor_t, float)
    y = np.interp(t, c, lab)
    j = np.clip(np.searchsorted(c, t), 1, len(c) - 1); i = j - 1
    w = np.clip((t - c[i]) / (c[j] - c[i]), 0, 1)
    ok = np.where(w <= 0, m[i], np.where(w >= 1, m[j], m[i] & m[j]))
    return y, ok


def legacy_targets(anchor_t, clip_labels, clip_mask, clip_s=2.0):
    """The V0 convention kept only for comparison: floor(t/2) bin lookup."""
    lab = np.asarray(clip_labels, float); m = np.asarray(clip_mask, bool)
    b = np.minimum((np.asarray(anchor_t) / clip_s).astype(int), len(lab) - 1)
    return lab[b], m[b]


def label_states(clip_labels, clip_mask, pos_thr=.75, neg_thr=None):
    """Per-clip state. QVH native: mask=False means unrated (IGNORE), not NEG.

    QVH ratings exist only for query-relevant clips, so there is no reliable
    generic NEG in this source unless neg_thr is given explicitly (then low
    rated clips inside the mask become NEG; still query-conditioned).
    """
    lab = np.asarray(clip_labels, float); m = np.asarray(clip_mask, bool)
    s = np.full(len(lab), IGNORE)
    s[m & (lab >= pos_thr)] = POS
    if neg_thr is not None: s[m & (lab <= neg_thr)] = NEG
    return s
