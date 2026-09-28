#!/usr/bin/env python3
"""Standalone post-unzip check of a mask/spatial combo package.

Reads the candidate ZIP and both scored parent ZIPs directly (not the packer's
intermediate JSONL), extracts each into a fresh temp dir, and checks:
174 rows vs index, video order == index order, frame keys == mask ZIP,
bboxes == spatial ZIP, ratio/int/bounds via the dependency-free checker.
Does not import aic.* or the packer.
"""
import argparse,hashlib,json,sys,tempfile,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from independent_submission_check import check,read_jsonl


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def unzip_rows(zp,tmp,tag):
    with zipfile.ZipFile(zp) as zf:
        if zf.namelist()!=['predictions.jsonl']:raise ValueError(f'{tag}: zip members {zf.namelist()}')
        d=Path(tmp)/tag;d.mkdir();zf.extractall(d)
    return d/'predictions.jsonl'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--index',required=True);ap.add_argument('--candidate',required=True)
    ap.add_argument('--mask-zip',required=True);ap.add_argument('--spatial-zip',required=True)
    ap.add_argument('--expect',nargs=3,metavar=('CAND','MASK','SPATIAL'),help='expected ZIP SHA-256s')
    a=ap.parse_args()
    shas={'candidate':sha(a.candidate),'mask':sha(a.mask_zip),'spatial':sha(a.spatial_zip)}
    if a.expect and list(shas.values())!=a.expect:raise ValueError(f'zip sha mismatch {shas}')
    order=[r['video_id'] for r in read_jsonl(a.index)]
    with tempfile.TemporaryDirectory() as tmp:
        cp=unzip_rows(a.candidate,tmp,'cand');mp=unzip_rows(a.mask_zip,tmp,'mask');spp=unzip_rows(a.spatial_zip,tmp,'spatial')
        indep=check(a.index,cp,None,require_size=False)
        cand=read_jsonl(cp);mask={r['video_id']:r for r in read_jsonl(mp)};spat={r['video_id']:r for r in read_jsonl(spp)}
        if [r['video_id'] for r in cand]!=order:raise ValueError('video order != index order')
        key_bad=[];bbox_bad=0;int_bad=0;empty=[];n=0
        for r in cand:
            vid=r['video_id'];f=[p['frame'] for p in r['predictions']]
            if f!=[p['frame'] for p in mask[vid]['predictions']]:key_bad.append(vid)
            sb={p['frame']:p['bboxes'] for p in spat[vid]['predictions']}
            bbox_bad+=sum(1 for p in r['predictions'] if p['bboxes']!=sb.get(p['frame']))
            int_bad+=sum(1 for p in r['predictions'] if not all(type(v) is int for v in p['bboxes']))
            if r['targetRatioWH']!=spat[vid]['targetRatioWH']:raise ValueError('ratio '+vid)
            if not f:empty.append(vid)
            n+=len(f)
        res={'valid':not key_bad and bbox_bad==0,'zip_sha256':shas,'predictions_sha256':sha(cp),'videos':len(cand),'predictions':n,
             'mask_prediction_count':sum(len(r['predictions']) for r in mask.values()),'frame_key_mismatch_videos':key_bad,
             'bbox_mismatch_frames':bbox_bad,'non_int_bbox_frames':int_bad,'empty_videos':empty,'independent':indep}
    print(json.dumps(res,indent=1))
    if not res['valid']:sys.exit(1)


if __name__=='__main__':main()
