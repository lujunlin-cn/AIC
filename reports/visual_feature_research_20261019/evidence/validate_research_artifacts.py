#!/usr/bin/env python3
"""Validate this research deliverable without downloading or running models.

Run from any directory. By default the JSON record is printed, not written.
To preserve it:
  python evidence/validate_research_artifacts.py > evidence/artifact_validation.json
The validation record excludes itself from its own inventory/hash calculation.
"""

import ast
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evidence" / "artifact_validation.json"
EXPECTED = {
    "visual_feature_research.md",
    "open_models_registry.md",
    "aic_transfer_learning_plan.md",
    "minimal_falsification_experiments.md",
    "research_verdict.md",
}


def main():
    errors = []
    files = sorted(p for p in ROOT.rglob("*")
                   if p.is_file() and p != OUTPUT and "__pycache__" not in p.parts
                   and p.suffix != ".pyc")
    texts = {}
    inventory = []
    for path in files:
        data = path.read_bytes()
        name = str(path.relative_to(ROOT))
        try:
            texts[path] = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            errors.append({"file": name, "check": "utf8", "error": str(exc)})
        inventory.append({
            "path": name,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        })

    md_files = sorted(p for p in files if p.suffix == ".md")
    if {str(p.relative_to(ROOT)) for p in md_files} != EXPECTED:
        errors.append({"check": "five_core_markdown_files", "expected": sorted(EXPECTED)})

    json_count = py_count = links_checked = external_links = 0
    for path, source in texts.items():
        name = str(path.relative_to(ROOT))
        if path.suffix == ".json":
            json_count += 1
            try:
                json.loads(source)
            except (ValueError, TypeError) as exc:
                errors.append({"file": name, "check": "json_parse", "error": str(exc)})
        if path.suffix == ".py":
            py_count += 1
            try:
                ast.parse(source, filename=name)
            except SyntaxError as exc:
                errors.append({"file": name, "check": "python_ast", "error": str(exc)})
        if path.suffix == ".md":
            for target in re.findall(r"\[[^\]\n]*\]\(([^\s)]+)\)", source):
                if re.match(r"(?:https?://|mailto:|#)", target):
                    external_links += int(target.startswith(("https://", "http://")))
                    continue
                links_checked += 1
                candidate = (path.parent / unquote(target.split("#", 1)[0])).resolve()
                if not candidate.exists():
                    errors.append({"file": name, "check": "local_link", "target": target})
            if re.search(r"\bturn\d+(?:search|view|fetch)\d+\b", source):
                errors.append({"file": name, "check": "nonportable_web_reference"})

            # Tables in these files use leading/trailing pipes. Escaped pipes do
            # not count as separators, including inside inline code/math.
            table_width = None
            fence = None
            for lineno, line in enumerate(source.splitlines(), 1):
                if line.startswith(("```", "~~~")):
                    marker = line[:3]
                    fence = None if fence == marker else marker
                if fence:
                    continue
                if line.startswith("|"):
                    cells = len(re.findall(r"(?<!\\)\|", line)) - 1
                    if table_width is None:
                        table_width = cells
                    elif cells != table_width:
                        errors.append({"file": name, "line": lineno,
                                       "check": "table_width", "expected": table_width,
                                       "observed": cells})
                else:
                    table_width = None
            if fence:
                errors.append({"file": name, "check": "unclosed_code_fence"})

    registry = texts[ROOT / "open_models_registry.md"]
    entries = list(re.finditer(r"^### (M\d{2}) .*?$", registry, flags=re.M))
    fields = {}
    for i, match in enumerate(entries):
        end = entries[i + 1].start() if i + 1 < len(entries) else len(registry)
        field_ids = re.findall(r"^\| (\d{2}) ", registry[match.end():end], flags=re.M)
        fields[match.group(1)] = field_ids
        if field_ids != [f"{n:02d}" for n in range(1, 18)]:
            errors.append({"check": "registry_17_fields", "entry": match.group(1),
                           "observed": field_ids})
    if [m.group(1) for m in entries] != [f"M{n:02d}" for n in range(1, 27)]:
        errors.append({"check": "registry_26_entries"})

    # Print only locations, never matched values, if an accidental credential
    # or signed download capability is present in a publication artifact.
    patterns = {
        "github_credential": r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})",
        "aws_access_id": r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
        "private_key": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        "signed_url": r"(?i)(?:x-amz-signature|x-goog-signature)=[0-9a-f]{24,}",
        "bearer_credential": r"(?i)authorization:\s*bearer\s+[A-Za-z0-9._-]{24,}",
    }
    for path, source in texts.items():
        for label, pattern in patterns.items():
            if re.search(pattern, source):
                errors.append({"file": str(path.relative_to(ROOT)), "check": label})

    provenance = json.loads(texts[ROOT / "evidence" / "research_provenance.json"])
    for item in provenance["actual_evidence"]:
        for field in ("script", "result"):
            if field in item and not (ROOT / "evidence" / item[field]).is_file():
                errors.append({"check": "provenance_file", "target": item[field]})
        for filename in item.get("results", []):
            if not (ROOT / "evidence" / filename).is_file():
                errors.append({"check": "provenance_file", "target": filename})
        if "script_sha256" in item:
            actual = hashlib.sha256((ROOT / "evidence" / item["script"]).read_bytes()).hexdigest()
            if actual != item["script_sha256"]:
                errors.append({"check": "provenance_script_hash", "target": item["script"]})

    # Arithmetic checks are accounting checks, not loaded-model verification.
    counts = {
        "v9_historical_tower_plus_source_head_plus_historical_yunet": 85_843_200 + 1_511_425 + 53_104,
        "planned_independent_graph_cap": 85_843_200 + 1_511_425 + 53_104 + 1_000_000,
        "planned_independent_graph_plus_u2netp_cap": 85_843_200 + 1_511_425 + 53_104 + 1_000_000 + 1_131_181,
        "historical_large_graph": 8_767_123_696 + 85_843_200 + 1_511_425 + 53_104,
        "historical_large_graph_plus_full_hlclip_class": 8_854_531_425 + 151_564_278,
    }
    expected_counts = [87_407_729, 88_407_729, 89_538_910, 8_854_531_425, 9_006_095_703]
    if list(counts.values()) != expected_counts:
        errors.append({"check": "parameter_arithmetic"})

    record = {
        "timestamp_shanghai": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "scope": "Artifact integrity only; no new model execution, task evaluation or official checker.",
        "base_commit": provenance["base_commit"],
        "core_markdown_count": len(md_files),
        "registry_entries": len(entries),
        "registry_fields_per_entry": {name: len(ids) for name, ids in fields.items()},
        "json_parse_count": json_count,
        "python_ast_count": py_count,
        "local_links_checked": links_checked,
        "external_link_occurrences": external_links,
        "external_links_retrieved_by_this_script": False,
        "parameter_arithmetic": counts,
        "file_count_excluding_this_record": len(files),
        "inventory_exclusions": ["evidence/artifact_validation.json", "__pycache__", "*.pyc"],
        "inventory": inventory,
        "errors": errors,
        "passed": not errors,
    }
    print(json.dumps(record, indent=2, ensure_ascii=False))
    return int(bool(errors))


if __name__ == "__main__":
    sys.exit(main())
