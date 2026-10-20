#!/usr/bin/env python3
"""Valen crop Choice with semantic IDs, source macro and order controls."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from aic.valen_adapter import ValenDecisionAdapter


def crop_request(row, k, root, shift=0, prompt_mode='generic'):
    candidates = row[f'candidates{k}']
    key = hashlib.sha256((row['key']+str(row['ratio'])).encode()).hexdigest()[:20]
    folder = Path(root)/key; folder.mkdir(parents=True,exist_ok=True)
    p0 = folder/'original.jpg'
    with Image.open(row['image']) as image:
        image=image.convert('RGB')
        # LIVE tables retain native dimensions while the public keyframe cache
        # is a 640x360 derivative.  Scale the legal native boxes into the
        # actual media pixels; labels and IoU remain in the native coordinates.
        sx, sy = image.width / float(row['W']), image.height / float(row['H'])
        if sx <= 0 or sy <= 0 or abs((image.width/image.height) - (row['W']/row['H'])) > 1e-3:
            raise ValueError((row['key'], image.size, row['W'], row['H']))
        if not p0.exists(): image.save(p0,quality=95)
        for j in candidates:
            p=folder/f'{j}.jpg'
            b=row['boxes'][j]
            scaled=(round(b[0]*sx), round(b[1]*sy), round(b[2]*sx), round(b[3]*sy))
            if not p.exists(): image.crop(scaled).save(p,quality=95)
    order = candidates[shift:]+candidates[:shift]
    if prompt_mode in {'no_original', 'minimal'}:
        content=[]
    else:
        content=[{'type':'text','text':
                  'This is the complete source frame. The images that follow are exact candidate crops from this same frame. '
                  'Each candidate already has the required output aspect ratio.'},
             {'type':'image_url','image_url':{'url':str(p0.resolve())}}]
    mapping={}
    for p,j in enumerate(order):
        label=chr(65+p); mapping[label]=j
        b = row['boxes'][j]
        cx = ((b[0] + b[2]) / 2) / float(row['W'])
        cy = ((b[1] + b[3]) / 2) / float(row['H'])
        if prompt_mode in {'geometry', 'precise'}:
            if cx < .34: side = 'near the left side'
            elif cx > .66: side = 'near the right side'
            else: side = 'near the horizontal center'
            if cy < .34: vert = 'near the top'
            elif cy > .66: vert = 'near the bottom'
            else: vert = 'near the vertical center'
            if prompt_mode == 'geometry':
                description = (f'Crop option {label}: a legal {row["ratio"][0]}:{row["ratio"][1]} '
                               f'window, center at normalized ({cx:.3f}, {cy:.3f}), '
                               f'{side}, {vert}. The image below is the actual crop. '
                               'The coordinates describe geometry only, not quality.')
            else:
                description = (f'Candidate crop {label}. It is a {row["ratio"][0]}:{row["ratio"][1]} crop '
                               f'whose normalized window centre is ({cx:.3f}, {cy:.3f}); '
                               f'it lies {side} and {vert}. This text gives geometry only. '
                               'Judge crop quality from the pixels below.')
        elif prompt_mode == 'explicit':
            description = (f'Candidate crop {label}. The image immediately after this description is the complete '
                           'candidate crop. Compare its visible subject, action, interactions and useful context '
                           'with the other candidate images. The option name carries no preference.')
        elif prompt_mode == 'minimal':
            description = f'Candidate crop {label}.'
        elif prompt_mode == 'numbered':
            description = f'Candidate {p + 1}. Exact candidate crop image; option names are arbitrary.'
        else:
            description = f'Crop option {label} image'
        content += [{'type':'text','text':f'Crop option {label}:'},
                    {'type':'image_url','image_url':{'url':str((folder/f'{j}.jpg').resolve())}}]
        # Replace the short label with the complete, deterministic candidate
        # description.  Keeping the label separate makes mapping auditable.
        content[-2]['text'] = description
    return {'state':{'messages':[{'role':'user','content':content}]},
            'questions':{'crop':{'type':'choice',
               'instructions':('Choose exactly one candidate crop for this frame. Maximize the expected official crop utility: '
                               'keep the important subject and action, preserve relevant interactions and context, '
                               'and avoid cutting off people, body parts, objects or essential text. '
                               'All candidates have the required aspect ratio. Compare the visible pixels; option names '
                               'and descriptions are not quality labels.' if prompt_mode != 'minimal' else
                               'Choose exactly one candidate crop. Compare only the visible pixels and preserve the important subject, action and context.'),
        'criteria':{key:(f'Candidate crop {key}; compare the visible pixels only.' if prompt_mode == 'explicit'
                          else f'Candidate crop {key}.' if prompt_mode == 'minimal'
                          else f'Crop option {key} image') for key in mapping}}}},mapping


def metrics(records, predictions):
    by={r['row_index']:r for r in records}
    groups=defaultdict(lambda:defaultdict(list))
    for p in predictions:
        if p.get('shift',0): continue
        r=by[p['row_index']]; k=p['k']; j=p['chosen_index']
        groups[(k,r['source'])]['valen'].append(r['iou'][j])
        for name,value in [('head',r['iou'][r['head_index']]),('center',r['iou'][64]),
                           ('oracle129',max(r['iou'])),('oracleK',max(r['iou'][j] for j in r[f'candidates{k}']))]:
            groups[(k,r['source'])][name].append(value)
    out={}
    for k in sorted({x[0] for x in groups}):
        for domain in ['rv','live','all']:
            src=sorted(s for kk,s in groups if kk==k and (domain=='all' or s.startswith(domain+':')))
            if not src: continue
            means={n:np.array([np.mean(groups[(k,s)][n]) for s in src]) for n in ['valen','head','center','oracle129','oracleK']}
            d=means['valen']-means['head']; rng=np.random.default_rng(20261020)
            boot=d[rng.integers(0,len(d),size=(4000,len(d)))].mean(1)
            out[f'k{k}_{domain}']={'sources':len(src),'rows':sum(len(groups[(k,s)]['valen']) for s in src),
                                 **{n:float(v.mean()) for n,v in means.items()},'delta_head':float(d.mean()),
                                 'ci95_head':np.quantile(boot,[.025,.975]).tolist(),
                                 'source_deltas':dict(zip(src,d.tolist()))}
    base={(p['row_index'],p['k']):p for p in predictions if not p.get('shift',0)}
    orders=[]
    for p in predictions:
        if not p.get('shift',0): continue
        b=base.get((p['row_index'],p['k']))
        if b:
            r=by[p['row_index']]
            orders.append({'same_choice':p['chosen_index']==b['chosen_index'],
                           'iou_delta':r['iou'][p['chosen_index']]-r['iou'][b['chosen_index']],
                           'prob_l1':sum(abs(p['candidate_probs'][key]-b['candidate_probs'][key]) for key in b['candidate_probs'])})
    out['order_control']={'n':len(orders),'same_choice':float(np.mean([p['same_choice'] for p in orders])) if orders else None,
                          'mean_iou_delta':float(np.mean([p['iou_delta'] for p in orders])) if orders else None,
                          'mean_probability_l1':float(np.mean([p['prob_l1'] for p in orders])) if orders else None}
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',required=True); ap.add_argument('--model',required=True); ap.add_argument('--checkpoint')
    ap.add_argument('--out',required=True)
    ap.add_argument('--prompt-mode',choices=['generic','geometry','precise','explicit','minimal','numbered','no_original'],default='generic')
    ap.add_argument('--ks',default='4,8'); ap.add_argument('--batch',type=int,default=4); ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--order-controls',type=int,default=32); ap.add_argument('--max-pixels',type=int,default=131072)
    ap.add_argument('--all-rows',action='store_true',help='evaluate every row instead of one deterministic pilot row per source')
    a=ap.parse_args(); data=Path(a.data); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    records=[json.loads(l) for l in (data/'records.jsonl').read_text().splitlines()]
    ids = list(range(len(records))) if a.all_rows else json.loads((data/'pilot_indices.json').read_text())
    # Interleave domains so a bounded smoke includes both public distributions.
    rv=[i for i in ids if records[i]['source'].startswith('rv:')]; live=[i for i in ids if records[i]['source'].startswith('live:')]
    ids=[i for pair in __import__('itertools').zip_longest(rv,live) for i in pair if i is not None]
    if a.limit: ids=ids[:a.limit]
    jobs=[(i,int(k),0) for k in a.ks.split(',') for i in ids]
    jobs += [(i,int(k),1) for k in a.ks.split(',') for i in ids[:a.order_controls]]
    pred_path=out/'predictions.jsonl'; predictions=[]
    if pred_path.exists(): predictions=[json.loads(l) for l in pred_path.read_text().splitlines()]
    done={(p['row_index'],p['k'],p['shift']) for p in predictions}; jobs=[j for j in jobs if j not in done]
    adapter=ValenDecisionAdapter(a.model,max_pixels=a.max_pixels,checkpoint=a.checkpoint)
    import torch
    t=time.perf_counter(); timings=[]
    with ThreadPoolExecutor(max_workers=4) as pool, pred_path.open('a') as stream:
        for st in range(0,len(jobs),a.batch):
            batch=jobs[st:st+a.batch]
            prepared=list(pool.map(lambda j:crop_request(records[j[0]],j[1],data/'media',j[2],a.prompt_mode),batch))
            results,timing=adapter.decide_batch([q[0] for q in prepared]); timings.append(timing)
            for (i,k,shift),(request,mapping),result in zip(batch,prepared,results):
                r=result['results']['crop']; p={'row_index':i,'source':records[i]['source'],'k':k,'shift':shift,
                    'chosen_index':mapping[r.selected_id], 'candidate_probs':{str(mapping[key]):v for key,v in r.candidate_probs.items()},
                    'raw':result['raw'],'tokens':result['tokens']}
                predictions.append(p); stream.write(json.dumps(p)+'\n')
            stream.flush()
            print(json.dumps({'done':st+len(batch),'total':len(jobs),'elapsed_s':time.perf_counter()-t,'batch':timing}),flush=True)
    report={'model':a.model,'checkpoint':a.checkpoint,'parameters':adapter.parameters,
            'checkpoint_tensors':getattr(adapter,'checkpoint_parameters',0),'prompt_mode':a.prompt_mode,'seconds':time.perf_counter()-t,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'metrics':metrics(records,predictions),'timings':timings,
            'official_test_used':False,'data_sha256':hashlib.sha256((data/'records.jsonl').read_bytes()).hexdigest()}
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='timings'},indent=2))


if __name__=='__main__': main()
