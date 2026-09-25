"""Post-run audit sidecars; never overwrite historical configs or launch claims."""
import argparse
import datetime
import hashlib
import json
import platform
import shlex
import subprocess
import sys
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source-commit',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--experiments-root',default='/data/aic/experiments')
    a=ap.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    repo=Path.cwd()
    source_files=sorted([*repo.glob('aic/*.py'),*repo.glob('scripts/*.py')])
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    source={str(p.relative_to(repo)):digest(p) for p in source_files}
    (out/'pip_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    (out/'nvidia_smi.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    roots=[];eroot=Path(a.experiments_root)
    for group in ['RG_NCV_001','RG_LINEAR_001']:
        roots.extend(p.parent for p in (eroot/group).glob('*/config.json'))
    for group in ['SUMME_OOD_001','SUMME_OOD_002','SUMME_OOD_003','SUMME_CV_OOD_001','SUMME_CV_OOD_002','SUMME_LINEAR_OOD_001']:
        roots.extend(p.parent for p in (eroot/group).glob('*/config.json'))
    for group in ['SPATIAL_GT_001','SPATIAL_GT_NATIVE_002','DENSE_E2E_001','ENGINEERING_RELEASE_001','ENGINEERING_RELEASE_002','ENGINEERING_RELEASE_003','SUMME_BASELINES_001','CROSS_DATASET_DUPLICATE_001']:
        if (eroot/group).is_dir():roots.append(eroot/group)
    runs=[]
    for run in roots:
        files={p.name:{'sha256':digest(p),'bytes':p.stat().st_size}
               for p in run.iterdir() if p.is_file() and p.suffix in ['.json','.jsonl','.txt','.sh']}
        cfg=run/'config.json'
        report={'run_directory':str(run),'retained_metadata':files,
                'config_git_commit':json.loads(cfg.read_text()).get('git_commit') if cfg.exists() else None,
                'recorded_after_run':True,'source_snapshot_commit':a.source_commit,
                'snapshot_directory':str(out),
                'limitation':'Captured after completion, not a contemporaneous launch snapshot. Historical config/command/environment preserved unchanged; code hashes describe verification-time files.'}
        sidecar=run/(out.name.lower()+'.json')
        if sidecar.exists():raise FileExistsError(sidecar)
        sidecar.write_text(json.dumps(report,indent=2)+'\n');runs.append(report)
    summary={'date':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':platform.node(),
             'python':sys.version,'interpreter':sys.executable,'source_snapshot_commit':a.source_commit,
             'remote_git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
             'note':'Remote working tree is individually synced and differs from its old Git HEAD; use explicit hashes and recorded implementation commits.',
             'command':shlex.join([sys.executable,*sys.argv]),'source_sha256':source,'runs':runs}
    (out/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'run_count':len(runs),'source_files':len(source),'output':str(out)}))


if __name__=='__main__':main()
