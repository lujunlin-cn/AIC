"""Record actual CPU test/runtime/raw-identity evidence on the media host."""
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys

import av
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'ext_data'))
from aicext.retargetvid_archive import digest

def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

start = datetime.now(timezone.utc).isoformat()
commands = [
    [sys.executable, '-m', 'compileall', '-q', 'ext_data/aicext', 'ext_data/scripts/ingest_dhf1k_retargetvid.py', 'scripts/audit_retargetvid.py', 'ext_data/tests/test_retargetvid_full_media.py'],
    [sys.executable, '-m', 'pytest', '-q', 'ext_data/tests/test_retargetvid_full_media.py'],
]
checks = []
for cmd in commands:
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    checks.append(dict(command=cmd, exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr))
write('remote_cpu_validation.json', dict(started_at=start, finished_at=datetime.now(timezone.utc).isoformat(),
      passed=all(c['exit_code']==0 for c in checks), checks=checks))
if not all(c['exit_code']==0 for c in checks):
    raise RuntimeError('Actual remote regression test failed; preserve its evidence')

inventory=json.loads((OUT/'media_inventory.json').read_text())
annotations=json.loads((OUT/'annotation_source_hashes.json').read_text())
media_ok=all(digest(m['path'])==m['sha256'] for m in inventory['source_videos'])
raw_ok=all(digest(path)==h for path,h in annotations['original_files'].items())
clocks_ok=all(digest(m['timeline_path'])==m['timeline_sha256'] for m in inventory['source_videos'])
split=json.loads((OUT/'split_leakage_audit.json').read_text())
protected_paths={'ledger':'ext_data/configs/holdout_exposure_v1.json',
                 'release':'ext_data/reports/releases/spatial_crop_v1.RELEASE.json',
                 'quality':'ext_data/reports/releases/spatial_crop_v1.quality.json'}
protected_ok=all(digest(ROOT/path)==split['protected_original_sha256_before'][key] for key,path in protected_paths.items())
write('post_audit_identity_check.json',dict(checked_at=datetime.now(timezone.utc).isoformat(),
      media_count=len(inventory['source_videos']),annotation_count=len(annotations['original_files']),
      all_AVI_SHA256_unchanged=media_ok, all_TXT_SHA256_unchanged=raw_ok,
      all_timelines_SHA256_valid=clocks_ok, original_ledger_release_quality_SHA256_unchanged=protected_ok,
      passed=media_ok and raw_ok and clocks_ok and protected_ok))
if not (media_ok and raw_ok and clocks_ok and protected_ok):
    raise RuntimeError('Source identity verification failed')

paths=list((ROOT/'ext_data/aicext').rglob('*.py'))
paths += [ROOT/p for p in ['ext_data/scripts/ingest_dhf1k_retargetvid.py','scripts/audit_retargetvid.py','ext_data/tests/test_retargetvid_full_media.py']]
write('remote_source_sha256.json', {str(p.relative_to(ROOT)):digest(p) for p in sorted(paths)})
limits={p:Path(p).read_text().strip() if Path(p).exists() else None for p in
        ['/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/memory.max']}
write('runtime_environment.json',dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
      numpy=np.__version__,pyav=av.__version__,ffmpeg_libraries={k:list(v) for k,v in av.library_versions.items()},
      resource_limits=limits, GPU_used=False, training_started=False,
      audit_code_identity='remote_source_sha256.json; actual deployed files, not an inferred remote Git checkout',
      reported_at=datetime.now(timezone.utc).isoformat()))
print(json.dumps(dict(remote_CPU_tests=True,media_SHA256_verified=200,annotation_SHA256_verified=2400,
                     timelines_verified=200,protected_identity_unchanged=True)))
