#!/usr/bin/env python3
"""Standalone post-unzip check of an axis-split diagnostic package (R0).

Reads the candidate ZIP and both scored parent ZIPs directly, recomputes each
video's free axis from index width/height/targetRatioWH with its own
arithmetic (no aic.* import), and checks: video order == index, frame keys ==
base, rows == override on the chosen axis and == base elsewhere.
"""
import argparse,json,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from independent_submission_check import check,read_jsonl
from verify_combo_zip import sha,unzip_rows


def free_axis(W,H,ratio):
    rw,rh=map(float,ratio);w=min(float(W),float(H)*rw/rh);h=w*rh/rw
    return 0 if w<W-1e-9 else (1 if h<H-1e-9 else None)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--index',required=True);ap.add_argument('--candidate',required=True)
    ap.add_argument('--base-zip',required=True);ap.add_argument('--override-zip',required=True);ap.add_argument('--axis',type=int,required=True)
    ap.add_argument('--expect',nargs=3,metavar=('CAND','BASE','OVERRIDE'));a=ap.parse_args()
    shas={'candidate':sha(a.candidate),'base':sha(a.base_zip),'override':sha(a.override_zip)}
    if a.expect and list(shas.values())!=a.expect:raise ValueError(f'zip sha mismatch {shas}')
    idx=read_jsonl(a.index)
    with tempfile.TemporaryDirectory() as tmp:
        cp=unzip_rows(a.candidate,tmp,'cand');indep=check(a.index,cp,None,require_size=False);cand=read_jsonl(cp)
        base={r['video_id']:r for r in read_jsonl(unzip_rows(a.base_zip,tmp,'base'))};over={r['video_id']:r for r in read_jsonl(unzip_rows(a.override_zip,tmp,'over'))}
        if [r['video_id'] for r in cand]!=[r['video_id'] for r in idx]:raise ValueError('video order != index order')
        bad_keys=[];bad_rows=[];n={'override':0,'base':0}
        for r,m in zip(cand,idx):
            vid=r['video_id'];src='override' if free_axis(m['width'],m['height'],m['targetRatioWH'])==a.axis else 'base';n[src]+=1
            if [p['frame'] for p in r['predictions']]!=[p['frame'] for p in base[vid]['predictions']]:bad_keys.append(vid)
            if r['predictions']!=(over if src=='override' else base)[vid]['predictions']:bad_rows.append(vid)
        res={'valid':not bad_keys and not bad_rows,'zip_sha256':shas,'predictions_sha256':sha(cp),'videos':len(cand),'predictions':sum(len(r['predictions']) for r in cand),
             'videos_by_source':n,'frame_key_mismatch_vs_base':bad_keys,'rows_not_equal_declared_source':bad_rows,'empty_videos':[r['video_id'] for r in cand if not r['predictions']],'independent':indep}
    print(json.dumps(res,indent=1))
    if not res['valid']:sys.exit(1)


if __name__=='__main__':main()
