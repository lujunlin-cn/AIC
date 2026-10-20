#!/usr/bin/env python3
"""Audit the Valen crop training contract and candidate oracle gap.

This script only reads public native-crop records and the generated Valen JSONL.
It does not read official test labels or predictions.  It is intentionally
independent of the Valen model so that a bad media/label contract is caught
before interpreting a checkpoint score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


def read_jsonl(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def softmax(values, temperature):
    z = [math.exp((float(v) - max(values)) / temperature) for v in values]
    s = sum(z)
    return [x / s for x in z]


def source_domain(source):
    if str(source).startswith("rv:"):
        return "RetargetVid"
    if str(source).startswith("live:"):
        return "LIVE"
    return str(source).split(":", 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--prepared", nargs="+", required=True)
    ap.add_argument("--temperature", type=float, default=0.12)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    records = read_jsonl(Path(args.records))
    # A source frame can occur once per target aspect ratio, so key alone is
    # not unique.  The ratio is part of the row identity.
    by_identity = {(r["key"], tuple(float(x) for x in r["ratio"])): r for r in records}
    prepared = []
    for path in args.prepared:
        rows = read_jsonl(Path(path))
        prepared.extend((path, r) for r in rows)

    errors = Counter()
    prepared_metrics = defaultdict(lambda: defaultdict(list))
    split_sources = defaultdict(set)
    image_checks = 0
    candidate_rows = 0

    for path, row in prepared:
        meta = row.get("meta", {})
        key = meta.get("row_key")
        ratio_key = tuple(float(x) for x in meta.get("ratio", []))
        original = by_identity.get((key, ratio_key))
        split = meta.get("split") or Path(path).stem
        source = meta.get("source")
        split_sources[split].add(source)
        if original is None:
            errors["missing_original_key"] += 1
            continue
        if source != original.get("source"):
            errors["meta_original_mismatch"] += 1
        candidates = list(original.get("candidates4", []))
        if len(candidates) != 4 or len(set(candidates)) != 4:
            errors["bad_candidate_set"] += 1
        if original.get("head_index") not in candidates:
            errors["head_not_in_candidate4"] += 1
        if 64 not in candidates:
            errors["center64_not_in_candidate4"] += 1
        labels = list("ABCD")
        try:
            content = row["request"]["state"]["messages"][0]["content"]
            criteria = row["request"]["questions"]["crop"]["criteria"]
            target = row["targets"]["crop"]["probabilities"]
            source_url = content[1]["image_url"]["url"]
            if Path(source_url).resolve() != Path(original["image"]).resolve():
                errors["source_image_mismatch"] += 1
            if set(criteria) != set(labels) or set(target) != set(labels):
                errors["option_key_mismatch"] += 1
            probs = softmax([original["iou"][j] for j in candidates], args.temperature)
            for label, expected in zip(labels, probs):
                if abs(float(target[label]) - expected) > 2e-6:
                    errors["target_label_mismatch"] += 1
            if abs(sum(float(target[x]) for x in labels) - 1.0) > 2e-6:
                errors["target_not_normalized"] += 1
            for j, label in zip(candidates, labels):
                text_item = content[2 + 2 * labels.index(label)]
                image_item = content[3 + 2 * labels.index(label)]
                if label not in text_item.get("text", ""):
                    errors["description_key_mismatch"] += 1
                crop_path = Path(image_item["image_url"]["url"])
                if crop_path.stem != str(j):
                    errors["candidate_filename_index_mismatch"] += 1
                if not crop_path.is_file():
                    errors["missing_candidate_image"] += 1
                    continue
                with Image.open(crop_path) as im:
                    w, h = im.size
                with Image.open(source_url) as source_image:
                    source_w, source_h = source_image.size
                sx = source_w / float(original["W"])
                # Recompute the exact integer PIL crop dimensions used by the
                # preparation script, based on the original native box.
                b = original["boxes"][j]
                expected_w = round(b[2] * sx) - round(b[0] * sx)
                sy = source_h / float(original["H"])
                expected_h = round(b[3] * sy) - round(b[1] * sy)
                if (w, h) != (expected_w, expected_h):
                    errors["candidate_crop_size_mismatch"] += 1
                if abs((w / h) - (original["ratio"][0] / original["ratio"][1])) > 0.01:
                    errors["candidate_ratio_mismatch"] += 1
                image_checks += 1
        except (KeyError, IndexError, TypeError, ValueError, OSError):
            errors["malformed_request"] += 1

        iou = [float(x) for x in original["iou"]]
        head = float(iou[int(original["head_index"])])
        center = float(iou[64])
        oracle129 = max(iou)
        oracle4 = max(iou[j] for j in candidates)
        candidate_rows += 1
        domain = source_domain(source)
        for name, value in (("head", head), ("center", center),
                            ("oracle129", oracle129), ("oracle4", oracle4),
                            ("candidate_recall_gap", oracle129 - oracle4)):
            prepared_metrics[(split, domain)][name].append(value)

    metrics = {}
    for (split, domain), values in prepared_metrics.items():
        sources = defaultdict(list)
        # Reconstruct source grouping from prepared rows for source-macro means.
        for path, row in prepared:
            meta = row.get("meta", {})
            if (meta.get("split") or Path(path).stem, source_domain(meta.get("source"))) != (split, domain):
                continue
            original = by_identity.get((meta.get("row_key"), tuple(float(x) for x in meta.get("ratio", []))))
            if original is None:
                continue
            src = meta.get("source")
            iou = [float(x) for x in original["iou"]]
            candidates = list(original["candidates4"])
            values_for_source = {
                "head": iou[int(original["head_index"])],
                "center": iou[64],
                "oracle129": max(iou),
                "oracle4": max(iou[j] for j in candidates),
            }
            sources[src].append(values_for_source)
        source_means = {}
        for src, rows in sources.items():
            source_means[src] = {k: mean([r[k] for r in rows]) for k in rows[0]}
        metrics[f"{split}:{domain}"] = {
            "sources": len(source_means),
            "rows": sum(len(v) for v in sources.values()),
            "source_macro": {k: mean([v[k] for v in source_means.values()]) for k in source_means[next(iter(source_means))]} if source_means else {},
            "row_mean": {k: mean(v) for k, v in values.items()},
        }

    result = {
        "records": len(records),
        "record_sources": len(set(r.get("source") for r in records)),
        "prepared_rows": len(prepared),
        "prepared_sources": {k: len(v) for k, v in split_sources.items()},
        "candidate_rows_checked": candidate_rows,
        "candidate_images_checked": image_checks,
        "errors": dict(errors),
        "contract_pass": not errors,
        "metrics": metrics,
        "source_sha256": hashlib.sha256(Path(args.records).read_bytes()).hexdigest(),
    }
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
