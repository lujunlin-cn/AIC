"""Holdout / exposure ledger and the project split.

The ledger records, per source-video group, every experiment that has looked
at it and in which role.  Membership is read from the experiments' own output
files (never from naming conventions or memory).  A ledger version is frozen
once (``_registry/holdout_exposure_<v>.json``); ``holdout_exposure_current.json``
points at the version that splits and releases use.

Roles
  fit                       labels were used to fit parameters (training data)
  selection                 used to choose a model / policy / threshold (dev)
  diagnostic                looked at for analysis
  confirmation              used for a confirmation / promotion decision
  in_progress_unregistered  a running experiment uses it; role not registered yet

Project split (``project_split``) per group, on top of the working
``aic_split``; ``official_split`` is never changed:
  quarantine                aic_split quarantine (in-house eval overlap)
  holdout_confirm_reserved  fresh confirmation reserve frozen with the ledger
                            (never used for selection/confirmation; may have
                            fit-only exposure, recorded per group)
  holdout_confirm_exposed   any confirmation use
  holdout_dev_exposed       any selection / diagnostic / unregistered use
  train / val / test_nolabel  otherwise the working aic_split
A video used for validation, and every cross-dataset copy in its group, never
re-enters training.  Fit-only exposure keeps ``train``.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

from .common import REGISTRY_DIR, now

EXP = Path("/data/aic/experiments")
CURRENT = REGISTRY_DIR / "holdout_exposure_current.json"
ROLE_ORDER = ("confirmation", "selection", "diagnostic", "in_progress_unregistered", "fit")
BLOCKING = {"confirmation": "holdout_confirm_exposed", "selection": "holdout_dev_exposed",
            "diagnostic": "holdout_dev_exposed", "in_progress_unregistered": "holdout_dev_exposed"}


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _rv(v) -> str:
    return f"dhf1k:{int(str(v).split('.')[0]):03d}"


def _yth(video_id: str) -> str:
    return "yt:" + video_id.split("__", 1)[1]


def collect(exp_root: Path = EXP, qvh_bounded: Optional[Path] = None) -> dict:
    """Read membership from experiment outputs; returns the ledger dict."""
    rows, evidence = [], {}

    def add(exp: str, role: str, split_name: str, dataset: str, groups: Iterable[str], path: Path,
            note: Optional[str] = None) -> None:
        path = Path(path)
        evidence[str(path)] = _sha(path) if path.is_file() else \
            "dir:" + hashlib.sha256("\n".join(sorted(q.name for q in path.iterdir())).encode()).hexdigest()
        for g in sorted(set(groups)):
            rows.append({"group_id": g, "dataset": dataset, "experiment": exp, "role": role,
                         "split_name": split_name, "evidence": str(path), "note": note})

    # round-1 spatial benchmarks on RetargetVid 001-020 (dev) and 021-030 (confirm)
    p = exp_root / "SPATIAL_SUBJECT_002" / "metrics.json"
    if p.exists():
        vids = {r["video_id"] for r in json.loads(p.read_text()).get("per_video", [])}
        add("SPATIAL_SUBJECT_002", "selection", "exposed20", "RetargetVid", map(_rv, vids), p)
    for exp in ("MAX_WINDOW_PATH_P1A_V1", "MAX_WINDOW_CAND_P1B_P2_V1", "MAX_WINDOW_CAND_P1B_P2_V2_PYAV15"):
        p = exp_root / exp / "metrics.json"
        if not p.exists():
            continue
        by = defaultdict(set)
        for r in json.loads(p.read_text()).get("per_video", []):
            by[r.get("split")].add(r["video_id"])
        for sp, vids in by.items():
            add(exp, "confirmation" if sp == "confirm" else "selection", str(sp), "RetargetVid", map(_rv, vids), p)
    # teacher v2 diagnostics: used / dev2 / confirm2, read from the per-keyframe output
    role_of = {"used": "diagnostic", "dev2": "selection", "confirm2": "confirmation"}
    for exp in ("T0_T2_TEACHER_DIAG_V1", "T1_T2_TEACHER_DIAG_V2"):
        p = exp_root / exp / "keyframes.jsonl"
        if not p.exists():
            continue
        by = defaultdict(set)
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                by[r["split"]].add(r["video_id"])
        for sp, vids in by.items():
            add(exp, role_of.get(sp, "diagnostic"), sp, "RetargetVid", map(_rv, vids), p)
    # T5 crop head: the unit tables it actually fitted / selected / confirmed on
    t5 = {"rv_train": "fit", "rv_dev": "selection", "rv_confirm2": "confirmation",
          "live_train": "fit", "live_dev": "selection", "live_confirm3": "confirmation"}
    for name, role in t5.items():
        p = exp_root / "T5_CROPHEAD_V3" / "tables" / f"{name}.pkl"
        if not p.exists():
            continue
        units = pickle.loads(p.read_bytes())
        ds = "RetargetVid" if name.startswith("rv") else "LIVE_YT_VC"
        groups = [_rv(u["vid"]) if ds == "RetargetVid" else f"liveytvc:{u['vid']}" for u in units]
        add("T5_CROPHEAD_V3", role, name, ds, groups, p)
    # V4 (running; no prereg found): membership from its video subset folders
    for sub in ("rv_subset", "rv_subset_c2"):
        d = exp_root / "V4_E1_OBS025" / sub
        if d.exists():
            vids = [q.stem for q in d.iterdir() if q.suffix.upper() == ".AVI"]
            add("V4_E1_OBS025", "in_progress_unregistered", sub, "RetargetVid", map(_rv, vids), d,
                note="running experiment without a registered role; treated as validation exposure")
    # temporal: YouTube Highlights val proxy (T3 packaging rule, T4 rejection)
    p = exp_root / "T3_YTH_VAL_PROXY_V1" / "records.jsonl"
    if p.exists():
        vids = [json.loads(l)["video_id"] for l in open(p) if l.strip()]
        for exp in ("T3_YTH_VAL_PROXY_V1", "T4_TEMP_BOUNDARY_V3"):
            add(exp, "selection", "yth_val_proxy", "YouTubeHighlights", map(_yth, vids), p,
                note="prereg 'dev_proxy'; decided T3 packaging and T4 rejection")
    # in-house bounded QVH protocol
    if qvh_bounded and Path(qvh_bounded).exists():
        d = json.loads(Path(qvh_bounded).read_text())
        for sp, role in (("train", "fit"), ("dev", "selection"), ("holdout", "confirmation")):
            add("QVH_NATIVE_BOUNDED_V1", role, sp, "QVHighlights(in-house)",
                [f"yt:{r['source_group']}" for r in d["splits"].get(sp, [])], Path(qvh_bounded))
    groups = defaultdict(set)
    for r in rows:
        groups[r["group_id"]].add(r["role"])
    return {"generated": now(), "rule": __doc__.split("Project split", 1)[1].strip(),
            "evidence_sha256": evidence, "exposures": rows,
            "groups": {g: sorted(v) for g, v in sorted(groups.items())}}


def current_ledger() -> Optional[dict]:
    if not CURRENT.exists():
        return None
    ver = json.loads(CURRENT.read_text())["version"]
    return json.loads((REGISTRY_DIR / f"holdout_exposure_{ver}.json").read_text())


def index_by_group(ledger: Optional[dict]) -> dict[str, list[dict]]:
    idx: dict[str, list[dict]] = defaultdict(list)
    for e in (ledger or {}).get("exposures", []):
        idx[e["group_id"]].append(e)
    return idx


RESERVE_SALT = "confirm_reserve_v1"


def reserve_bucket(group_id: str, salt: str = RESERVE_SALT) -> float:
    h = hashlib.sha256(f"{salt}:{group_id}".encode()).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)


def project_split(aic_split: Optional[str], aic_reason: Optional[str],
                  exposures: list[dict], reservation: Optional[dict] = None) -> tuple[Optional[str], str]:
    if aic_split == "quarantine":
        return "quarantine", aic_reason or "quarantine"
    if reservation:
        return "holdout_confirm_reserved", reservation["reason"]
    for role in ROLE_ORDER[:-1]:
        hit = sorted({f"{e['experiment']}:{e['split_name']}" for e in exposures if e["role"] == role})
        if hit:
            return BLOCKING[role], f"{role} exposure: {', '.join(hit)} (never train)"
    fit = sorted({f"{e['experiment']}:{e['split_name']}" for e in exposures if e["role"] == "fit"})
    reason = aic_reason or ""
    if fit:
        reason += f"; fit-only exposure {', '.join(fit)}"
    return aic_split, reason.strip("; ")


def worst_role(exposures: list[dict]) -> Optional[str]:
    roles = {e["role"] for e in exposures}
    return next((r for r in ROLE_ORDER if r in roles), None)
