#!/usr/bin/env python3
"""Reuse aligned public R70 caches; never read official test media here."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aic.highlight_decisions import diverse_candidates
from r70_spatial_utility_pilot import Head, candidate_offsets, boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--t5-root", required=True)
    ap.add_argument("--rv-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    rows = [json.loads(l) for l in Path(a.rows).read_text().splitlines()]
    x = np.load(a.features, mmap_mode="r"); y = np.load(a.targets)
    assert len(x) == len(y) == len(rows)
    train_src = {r['source'] for r in rows if r['split']=='train'}
    dev_src = {r['source'] for r in rows if r['split']=='dev'}
    assert not train_src & dev_src
    ck = torch.load(a.head, map_location="cpu", weights_only=False)
    head = Head.make(torch).cuda().eval(); head.load_state_dict(ck['state_dict'])
    mu = torch.as_tensor(ck['mu'], device='cuda').float()
    sd = torch.as_tensor(ck['sd'], device='cuda').float()
    scores = np.zeros_like(y)
    started = time.perf_counter()
    with torch.inference_mode():
        for st in range(0,len(rows),64):
            z = torch.from_numpy(np.array(x[st:st+64])).cuda().float()
            scores[st:st+len(z)] = head((z-mu)/sd).cpu().numpy()
    records = []; grouped = {}
    for i,r in enumerate(rows):
        ds,vid,frame = r['key'].split(':')
        path = (Path(a.t5_root)/'keyframes_train'/vid/f'{frame}.png' if ds=='live'
                else Path(a.rv_root)/'d1'/vid/f'{frame}.png')
        if not path.is_file(): raise FileNotFoundError(path)
        offsets = candidate_offsets(r['W'],r['H'],r['ratio'])[0]
        bb = boxes(offsets,r['W'],r['H'],r['ratio'])
        c4 = diverse_candidates(scores[i],4); c8 = diverse_candidates(scores[i],8)
        record = {'row_index':i,'key':r['key'],'source':r['source'],'split':r['split'],
                  'name':r['name'],'ratio':r['ratio'],'W':r['W'],'H':r['H'],'image':str(path),
                  'head_index':int(scores[i].argmax()),'candidates4':c4,'candidates8':c8,
                  'boxes':bb.tolist(),'iou':y[i].tolist(),'gt':r['gt']}
        records.append(record); grouped.setdefault(r['source'],[]).append(i)
    # One row per dev source, hash chosen independently of scores and labels.
    pilot = []
    for s in sorted(dev_src):
        ids = grouped[s]
        pilot.append(min(ids,key=lambda i:hashlib.sha256((records[i]['key']+str(records[i]['ratio'])+':valen-s0').encode()).hexdigest()))
    write = lambda p,z: (out/p).write_text(json.dumps(z,indent=2)+'\n')
    (out/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    write('pilot_indices.json',pilot)
    def macro(v):
        return float(np.mean([np.mean(v[[i for i in ids if records[i]['split']=='dev']]) for s,ids in grouped.items() if s in dev_src]))
    metrics = {}
    ids = np.arange(len(rows)); metrics['head'] = macro(y[ids,scores.argmax(1)])
    metrics['center'] = macro(y[:,64]); metrics['oracle129'] = macro(y.max(1))
    for k in (4,8):
        metrics[f'oracle{k}'] = macro(np.array([max(r['iou'][j] for j in r[f'candidates{k}']) for r in records]))
        metrics[f'raw_top{k}_oracle'] = macro(np.take_along_axis(y,np.argsort(scores,axis=1)[:,-k:],axis=1).max(1))
    write('prepare.json',{'rows':len(rows),'train_sources':len(train_src),'dev_sources':len(dev_src),'pilot_rows':len(pilot),
                          'seconds':time.perf_counter()-started,'all_dev':metrics,'head':a.head,
                          'head_sha256':hashlib.sha256(Path(a.head).read_bytes()).hexdigest(),
                          'row_metadata_sha256':hashlib.sha256(Path(a.rows).read_bytes()).hexdigest()})
    print(json.dumps(json.loads((out/'prepare.json').read_text()),indent=2))


if __name__=='__main__': main()
