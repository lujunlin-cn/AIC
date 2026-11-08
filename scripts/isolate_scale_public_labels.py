"""Quarantine entire Train sources with exact Val bytes; preserve original inputs."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path

from aic.scale_data import read_jsonl
from aic.scale_parent import atomic_json,file_sha


def isolate(rows):
    groups=defaultdict(list)
    for row in rows:
        if row['split'] not in {'train','val'}:raise ValueError('public Test rows forbidden')
        groups[row['image_sha256']].append(row)
    cross=[v for v in groups.values() if len({r['split'] for r in v})>1]
    quarantine={(r['dataset_source'],str(r['source_id'])) for v in cross for r in v if r['split']=='train'}
    clean=[dict(r) for r in rows if (r['dataset_source'],str(r['source_id'])) not in quarantine]
    for r in clean:
        if r['dataset_source']=='GNMC':r['independent_source_id']='image-sha256:'+r['image_sha256']
    return clean,dict(cross_split_identical_image_groups=len(cross),
                      quarantined_train_sources=[list(k) for k in sorted(quarantine)],
                      original_views=len(rows),removed_views=len(rows)-len(clean),clean_views=len(clean),
                      policy='keep Val, quarantine complete Train source with all views; preserve originals',
                      original_labels_changed=False)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rows,report=isolate(read_jsonl(a.labels))
    a.output.mkdir(parents=True,exist_ok=False)
    path=a.output/'labels.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report.update(labels_sha256=file_sha(path),original_labels_sha256=file_sha(a.labels))
    atomic_json(a.output/'isolation.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
