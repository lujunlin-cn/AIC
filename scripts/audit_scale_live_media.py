"""Verify the execution host's complete original LIVE video inventory by SHA256."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from aic.scale_parent import atomic_json, file_sha


def audit(manifest, video_root):
    expected=json.loads(manifest.read_text())['video_sha256']
    def check(item):
        source,wanted=item
        path=video_root/(source+'.mp4')
        if not path.is_file():return source,dict(status='missing')
        actual=file_sha(path)
        return source,dict(status='verified' if actual==wanted else 'hash_mismatch',
                           sha256=actual,expected_sha256=wanted,bytes=path.stat().st_size)
    with ThreadPoolExecutor(max_workers=4) as pool:
        inventory=dict(pool.map(check,expected.items()))
    missing=[s for s,v in inventory.items() if v['status']=='missing']
    mismatch=[s for s,v in inventory.items() if v['status']=='hash_mismatch']
    return dict(status='verified' if not missing and not mismatch else 'incomplete',
                source_root=str(video_root),manifest_sha256=file_sha(manifest),
                expected_sources=len(expected),verified_sources=len(expected)-len(missing)-len(mismatch),
                missing_sources=missing,hash_mismatch_sources=mismatch,sources=inventory,
                pts_frame_pixel_recovery='checked separately during P1 projection for every used source')


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--video-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=audit(a.manifest,a.video_root);atomic_json(a.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='sources'},indent=2))
    return 0 if result['status']=='verified' else 2


if __name__=='__main__':raise SystemExit(main())
