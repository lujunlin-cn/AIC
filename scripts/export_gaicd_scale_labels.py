"""Export Journal Train image inputs and original MOS candidates, never Test labels."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from aic.scale_parent import atomic_json, file_sha


def export(root, output):
    root=root.resolve()
    dataset=root/'datasets/GAICD'
    media={}
    for line in (dataset/'processed/media.jsonl').open():
        row=json.loads(line)
        if 'journal' in row.get('release','') and row.get('official_split')=='train':
            media[row['item_id']]=row
    if not media:
        raise ValueError('no Journal Train media inventory')
    output.mkdir(parents=True,exist_ok=False)
    labels=[]
    seen=set()
    for line in (dataset/'processed/annotations.jsonl').open():
        entry=json.loads(line)
        key=entry.get('media_uid','').split(':')[-1]
        if entry.get('variant')!='journal' or key not in media:
            continue
        ann=json.loads(Path(entry['annotation_path']).read_text())
        if str(ann['image'])!=key or key in seen:
            raise ValueError('Journal annotation key is duplicate or mismatched')
        seen.add(key)
        m=media[key]
        image=Path(m['media_path']).resolve()
        if not image.is_relative_to(root) or file_sha(image)!=m['sha256']:
            raise ValueError('GAICD input media SHA/path mismatch')
        if [ann['W'],ann['H']] != [m['width'],m['height']]:
            raise ValueError('Journal image dimensions mismatch')
        lengths={len(ann[k]) for k in ('boxes_xywh','mos','valid')}
        if len(lengths)!=1:
            raise ValueError('candidate score/geometry lengths differ')
        base=dict(dataset_source='GAICD',source_id=m['group_id'],image_id=key,
                  split='train',release='journal',image_width=ann['W'],image_height=ann['H'],
                  image_sha256=m['sha256'],input_image=str(image.relative_to(root)),
                  position_source=None,media_size=[ann['W'],ann['H']],
                  annotation_sha256=file_sha(entry['annotation_path']),
                  candidates=[dict(crop_xywh=b,mos=score,valid=valid)
                              for b,score,valid in zip(ann['boxes_xywh'],ann['mos'],ann['valid'])])
        for ratio in ([9,16],[16,9]):
            labels.append(dict(base,target_ratio=ratio))
    if seen!=set(media):
        raise ValueError('incomplete Journal Train annotation/media join')
    path=output/'labels.jsonl'
    path.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in labels))
    result=dict(images=len(media),views=len(labels),labels_sha256=file_sha(path),
                label_path=str(path),test_labels_opened=False,release='Journal Train only')
    atomic_json(output/'manifest.json',result)
    return result


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(export(a.root,a.output),indent=2))


if __name__=='__main__':main()
