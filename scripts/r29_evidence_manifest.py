"""Bind the R29 report, executable code and allowlisted public inputs by SHA256."""
import argparse
import datetime
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports/r29/evidence_manifest.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def entry(path):
    return {"bytes": path.stat().st_size, "sha256": digest(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        saved = json.loads(DEFAULT_OUTPUT.read_text())
        count = 0
        for group in ("files", "public_input_files"):
            for name, expected in saved[group].items():
                assert entry(ROOT / name) == expected, name
                count += 1
        print(json.dumps({"status": "PASS", "verified_files": count}))
        return
    if DEFAULT_OUTPUT.exists():
        raise FileExistsError("Preserve original manifest; use --verify")

    files = set((ROOT / "reports/r29").rglob("*"))
    files.update((ROOT / "scripts").glob("r29*"))
    files.update((ROOT / "configs").glob("R29*"))
    files.update(ROOT / p for p in [
        "01_赛事规则与评分标准.md", "02_数据集_模型路线_特征工程研究.md",
        "AGENTS.md", "STATUS.md", "DECISIONS.md", "EXPERIMENTS.md", "TODO.md",
        "experiments/registry.jsonl", "configs/R28_SPATIAL_PUBLIC_PILOT.json",
        "reports/r22/r22_instrument_report.json", "reports/r26/public_metrics.json",
    ])
    files = {p for p in files if p.is_file() and "__pycache__" not in p.parts}

    public_root = ROOT / "artifacts/r29/public_evidence"
    bindings = json.loads((public_root / "bindings.json").read_text())
    assert len(bindings) == 40 and len({r["video_id"] for r in bindings}) == 40
    assert {r["split"] for r in bindings} == {"train", "dev"}
    declared = json.loads((public_root / "source_hashes.json").read_text())
    public_files = {}
    for name, expected_sha in declared.items():
        relative = Path(name)
        assert not relative.is_absolute() and ".." not in relative.parts
        path = public_root / relative
        assert digest(path) == expected_sha, name
        public_files[str(path.relative_to(ROOT))] = entry(path)
    for name in ["bindings.json", "source_hashes.json", "sampled_rv_gt.json"]:
        path = public_root / name
        public_files[str(path.relative_to(ROOT))] = entry(path)
    archive = ROOT / "artifacts/r29/exposed_public_evidence.tar.gz"
    public_files[str(archive.relative_to(ROOT))] = entry(archive)

    record = {
        "experiment_id": "R29_BREAKTHROUGH_20261016",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "parent_git": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scope": "R29 code/config/reports; public inputs excluded from Git; no self hash",
        "input_allowlist": {"temporal_sources": 40, "splits": ["train", "dev"],
                            "spatial_sources": ["031", "032", "033", "034"],
                            "protected_access": False, "official_test_access": False},
        "files": {str(p.relative_to(ROOT)): entry(p) for p in sorted(files)},
        "public_input_files": dict(sorted(public_files.items())),
        "resource_and_exit_evidence": [
            "reports/r29/temporal_probe_environment.json",
            "reports/r29/temporal_probe_resource.log", "reports/r29/decision_theory.json",
            "reports/r29/spatial_completion/spatial_exit_code.txt",
            "reports/r29/spatial_completion/spatial_continuation/environment_observed.json",
            "reports/r29/closeout_resource.json",
        ],
        "claim_boundary": "Implementation/development evidence only; no new candidate, independent sources or official score",
    }
    DEFAULT_OUTPUT.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"manifest": str(DEFAULT_OUTPUT.relative_to(ROOT)),
                      "files": len(record["files"]), "public_inputs": len(public_files)}))


if __name__ == "__main__":
    main()
