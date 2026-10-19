#!/usr/bin/env python3
"""Package one audited exact-budget temporal candidate."""
from __future__ import annotations
import argparse, hashlib, json, tempfile, zipfile
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def rows(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--candidate',type=Path,required=True); ap.add_argument('--b3-allframes',type=Path,required=True); ap.add_argument('--center-parent',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--name',required=True); ap.add_argument('--mechanism',required=True); ap.add_argument('--params',type=int,default=450283377); a=ap.parse_args()
    cand=rows(a.candidate); full=rows(a.b3_allframes); center=rows(a.center_parent); fm={str(x['video_id']):x for x in full}; cm={str(x['video_id']):x for x in center};
    if len(cand)!=426 or len(fm)!=426 or len(cm)!=426: raise ValueError('coverage')
    out=[]; total=0; details=[]
    for q in sorted(cand,key=lambda x:int(str(x['video_id']))):
        v=str(q['video_id']); n=len(fm[v]['predictions']); k=int(round(.85*n)); ids=[int(p['frame']) for p in q['predictions']]
        if len(ids)!=k or ids!=sorted(set(ids)): raise ValueError(f'{v}: budget/order {len(ids)} {k}')
        allp={int(p['frame']):p for p in fm[v]['predictions']}; qp={int(p['frame']):p for p in q['predictions']}
        if any(allp[i].get('bboxes')!=p.get('bboxes') for i,p in qp.items()): raise ValueError(f'{v}: box mismatch full B3')
        cp={int(p['frame']):p for p in cm[v]['predictions']}; changed=len(set(ids)^set(cp))//2
        z=dict(q); z['model_size_mb']=a.params/1e6; z['model_params_b']=a.params/1e9; out.append(z); total+=k; details.append({'video_id':v,'n_full':n,'n_keep':k,'changed_vs_center':changed})
    a.output.mkdir(parents=True,exist_ok=False); pred=a.output/'predictions.jsonl'
    with pred.open('w') as f:
        for z in out: f.write(json.dumps(z,ensure_ascii=False,separators=(',',':'))+'\n')
    zp=a.output/f'{a.name}.zip'
    with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z: z.write(pred,'predictions.jsonl')
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zp) as z:
            if z.namelist()!=['predictions.jsonl']: raise ValueError('zip members')
            z.extractall(td)
        if sha(Path(td)/'predictions.jsonl')!=sha(pred): raise ValueError('zip roundtrip')
    m={'submission_id':a.name,'status':'PACKAGED_NOT_UPLOADED','uploaded':False,'official_platform_score':None,'parent_submission':'R26_POS_CENTER_EXACT085_B3','mechanism':a.mechanism,'changed_factor':'temporal frame membership only; full-frame B3 spatial boxes copied','keep_rule':'round(0.85 * full original frame count)','video_count':len(out),'prediction_count':total,'total_parameters':a.params,'model_params_b':a.params/1e9,'model_size_mb':a.params/1e6,'predictions_sha256':sha(pred),'zip_sha256':sha(zp),'zip_bytes':zp.stat().st_size,'spatial_boxes_source':'R26/LFM B3 full-frame artifact','per_video':details}
    (a.output/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n'); (a.output/(zp.name+'.sha256')).write_text(sha(zp)+'  '+zp.name+'\n'); print(json.dumps({k:v for k,v in m.items() if k!='per_video'},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
