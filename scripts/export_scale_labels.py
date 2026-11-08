"""Export real LIVE/GNMC train/val labels for a later immutable P1 position join.

This emits public crop labels, never substitutes them for P1 centers, and never
opens GNMC Test labels. No image extraction or CUDA is required.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile
from PIL import Image
from aic.scale_data import gnmc_training_row


def export(snapshot, gnmc_zip, gnmc_archive_relative, split_path, output):
    output.mkdir(parents=True,exist_ok=False)
    split=json.loads(split_path.read_text());counts={};aggregate=hashlib.sha256()
    with (output/'labels.jsonl').open('w') as f:
        def emit(row):
            key=row['dataset_source']+':'+row['split']+':'+':'.join(map(str,row['target_ratio']))
            counts[key]=counts.get(key,0)+1
            payload=json.dumps(row,sort_keys=True)+'\n';f.write(payload);aggregate.update(payload.encode())
        for split_name,file_name in [('train','train_a.json'),('val','dev_a.json')]:
            allowed=set(split[split_name]);seen=set();frame_counts={}
            for raw in json.loads((snapshot/'repro/liveytvc_full_retrain_20261023/data'/file_name).read_text()):
                if raw['group'] not in allowed:continue  # explicit, versioned quarantine only
                seen.add(raw['group']);frame_counts[raw['group']]=frame_counts.get(raw['group'],0)+1
                W,H=raw['original_size'];g=raw['human_raw_crop_original_pixels']
                box=[g['left'],g['top'],g['right']-g['left'],g['bottom']-g['top']]
                mw=max(32,round(W*640/max(W,H)/32)*32);mh=max(32,round(H*640/max(W,H)/32)*32)
                base={'dataset_source':'LIVE_YT_VC','source_id':raw['group'],'image_id':raw['id'],
                      'frame_id':raw['frame_id'],'split':split_name,'image_sha256':raw['image_sha256'],
                      'input_image':'repro/liveytvc_full_retrain_20261023/data/'+raw['image'],
                      'source_video_sha256':raw['source_video_sha256'],
                      'position_source':None,'coordinate_units':'original_image_pixels'}
                emit(dict(base,image_width=W,image_height=H,media_size=[mw,mh],target_ratio=[9,16],
                          gt_crop_xywh=box,transposed_geometry_augmentation=False))
                emit(dict(base,image_width=H,image_height=W,media_size=[mh,mw],target_ratio=[16,9],
                          gt_crop_xywh=[box[1],box[0],box[3],box[2]],transposed_geometry_augmentation=True))
            if seen != allowed or any(v!=30 for v in frame_counts.values()):
                raise ValueError('LIVE public-label coverage is incomplete; no silent source/frame skip')
        with ZipFile(gnmc_zip) as z:
            for split_name,folder in [('train','train'),('val','validation')]:
                for raw in json.loads(z.read('GNMC/json/'+folder+'.json')):
                    member='GNMC/'+folder+'/'+Path(raw['filename']).name;payload=z.read(member)
                    with Image.open(io.BytesIO(payload)) as image:W,H=image.size
                    for ratio,transpose in [((16,9),False),((9,16),True)]:
                        row=gnmc_training_row(raw,split=split_name,image_width=W,image_height=H,
                                              target_ratio=ratio,transpose=transpose)
                        row.update(input_image=member,input_archive=gnmc_archive_relative,
                                   image_sha256=hashlib.sha256(payload).hexdigest(),
                                   media_size=[H,W] if transpose else [W,H],position_source=None)
                        emit(row)
    result={'status':'public_labels_exported_NOT_P1_conditioned','counts':counts,
            'labels_sha256':aggregate.hexdigest(),'P1_positions_substituted':False,
            'gnmc_archive_deployment_path':gnmc_archive_relative,
            'archive_must_exist_on_execution_host':True,'test_labels_opened':False}
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--gnmc-zip',type=Path,required=True);p.add_argument('--gnmc-archive-relative',required=True)
    p.add_argument('--live-split',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(export(a.snapshot,a.gnmc_zip,a.gnmc_archive_relative,a.live_split,a.output),indent=2))
if __name__=='__main__':main()
