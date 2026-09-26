#!/usr/bin/env python3
"""Run frozen release candidates on an intake-verified evaluation index."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="release manifest")
    parser.add_argument("--candidate", action="append",
                        help="candidate ID; repeat, defaults to all candidates")
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--index", required=True, help="intake-enriched JSONL")
    parser.add_argument("--intake-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stage", choices=["preliminary", "final"], default="final")
    parser.add_argument("--run-id", help="fresh run directory name")
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    args.manifest = Path(args.manifest).resolve()
    args.weights_dir = Path(args.weights_dir).resolve()
    args.index = Path(args.index).resolve()
    args.intake_manifest = Path(args.intake_manifest).resolve()
    args.output_dir = Path(args.output_dir).resolve()
    release = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    intake = json.loads(Path(args.intake_manifest).read_text(encoding="utf-8"))
    if intake.get("index_sha256") != hashlib.sha256(Path(args.index).read_bytes()).hexdigest():
        raise SystemExit("intake manifest does not match --index")
    candidates = args.candidate or list(release["candidates"])
    unknown = sorted(set(candidates) - set(release["candidates"]))
    if unknown:
        raise SystemExit(f"unknown candidate(s): {unknown}")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("eval_%Y%m%dT%H%M%SZ")
    run_dir = Path(args.output_dir) / run_id
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {run_dir}")
    run_dir.mkdir(parents=True)
    records = []
    for candidate in candidates:
        output = run_dir / f"{candidate}.jsonl"
        report = run_dir / f"{candidate}.report.json"
        command = [sys.executable, "-m", "aic.release", "--manifest", args.manifest,
                   "--candidate", candidate, "--weights-dir", args.weights_dir,
                   "--index", args.index, "--output", str(output), "--report", str(report),
                   "--device", args.device, "--stage", args.stage]
        (run_dir / f"{candidate}.command.txt").write_text(
            " ".join(command) + "\n", encoding="utf-8")
        result = subprocess.run(command, check=False, cwd=repo_root)
        record = {"candidate": candidate, "returncode": result.returncode,
                  "output": str(output), "report": str(report),
                  "command": command}
        records.append(record)
        if result.returncode != 0:
            (run_dir / "batch.json").write_text(json.dumps({"run_id": run_id,
                "release_manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                "intake_manifest": intake, "records": records}, indent=2) + "\n")
            raise SystemExit(f"candidate failed: {candidate}")
    summary = {"run_id": run_id, "release_manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
               "intake_manifest": intake, "candidates": candidates, "records": records,
               "official_f_video": None, "competition_score": None}
    (run_dir / "batch.json").write_text(json.dumps(summary, ensure_ascii=False,
                                                    indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
