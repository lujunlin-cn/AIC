"""Frozen-score R26 budget/head diagnostics and single-factor submission controls.
No test labels, video decoding or training. Test outputs remain ignored.
"""
import argparse, hashlib, json, zipfile
from pathlib import Path
import numpy as np
from independent_submission_check import check
from aic.contract import load_jsonl, load_index, validate_submission


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p,x):
    Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def owners(samples,n):
    return np.searchsorted((np.asarray(samples[:-1])+samples[1:])//2,np.arange(n),side='left')


def exact_mask(samples,scores,n,k):
    own=owners(samples,n); keep=np.zeros(n,bool)
    for j in np.argsort(-np.asarray(scores),kind='stable'):
        cell=np.flatnonzero(own==j); need=k-int(keep.sum())
        if need<=0: break
        if len(cell)>need:
            cell=cell[np.lexsort((cell,np.abs(cell-samples[j])))[:need]]
        keep[cell]=True
    assert keep.sum()==k
    return keep


def ranks(a):
    a=np.asarray(a); ix=np.argsort(a,kind='stable'); vals=a[ix]
    starts=np.r_[0,np.flatnonzero(np.diff(vals))+1]; ends=np.r_[starts[1:],len(a)]
    out=np.empty(len(a),float)
    for s,e in zip(starts,ends): out[ix[s:e]]=(s+e-1)/2
    return out


def corr(a,b):
    a=np.asarray(a); b=np.asarray(b)
    return float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else None


def dist(v):
    v=np.asarray([x for x in v if x is not None],float)
    return dict(n=len(v),mean=float(v.mean()),p10=float(np.quantile(v,.1)),median=float(np.median(v)),p90=float(np.quantile(v,.9)),min=float(v.min()),max=float(v.max()))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--index',required=True); ap.add_argument('--scores',required=True); ap.add_argument('--spatial',required=True); ap.add_argument('--r25',required=True); ap.add_argument('--output',required=True); ap.add_argument('--report',required=True); a=ap.parse_args()
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True); report=Path(a.report); report.mkdir(parents=True,exist_ok=True)
    metas=load_jsonl(a.index); baseline={r['video_id']:r for r in load_jsonl(Path(a.r25)/'predictions.jsonl')}; spatial={r['video_id']:r for r in load_jsonl(a.spatial)}
    r25_manifest=json.loads((Path(a.r25)/'manifest.json').read_text())
    assert sha(a.index)==r25_manifest['run']['index_sha256']; assert sha(a.spatial)==r25_manifest['run']['spatial_sha256']
    assert len(metas)==len(baseline)==len(spatial)==426
    kinds={'R26_AHA_EXACT085_B3':8480646545,'R26_POS_CENTER_EXACT085_B3':450283377,'R26_R25MASK_CENTER':8030363168}
    allrows={k:[] for k in kinds}; maskstats={k:[] for k in ['R25_original']+list(kinds)}; heads=[]; budget=[]; changed=0; identity=0; curves=[[] for _ in range(10)]
    for m in metas:
        vid=m['video_id']; n=m['frame_count']; d=json.loads((Path(a.scores)/(vid+'.json')).read_text()); x=np.asarray(d['frame_indices']); v=np.asarray(d['values']); sc=np.asarray(d['scores']); k=round(.85*n)
        assert np.array_equal(sc,v[:,0]+v[:,1]); assert d['run_code_sha256']==r25_manifest['run']['source_code_sha256']
        own=owners(x,n); old=np.zeros(n,bool)
        for j in np.argsort(-sc,kind='stable'):
            old[own==j]=True
            if old.sum()>=k: break
        oldframes=[p['frame'] for p in baseline[vid]['predictions']]
        assert np.array_equal(np.flatnonzero(old),oldframes)
        exact=exact_mask(x,sc,n,k); assert np.all(~exact|old)
        center=np.zeros(n,bool); start=(n-k)//2; center[start:start+k]=True
        for label,mask in [('R25_original',old),('R26_AHA_EXACT085_B3',exact),('R26_POS_CENTER_EXACT085_B3',center),('R26_R25MASK_CENTER',old)]:
            chosen=np.flatnonzero(mask)
            maskstats[label].append(dict(keep=float(mask.mean()),thirds=[float(mask[i*n//3:(i+1)*n//3].mean()) for i in range(3)],runs=int(np.count_nonzero(mask & ~np.r_[False,mask[:-1]])),first=bool(mask[x[0]]),last=bool(mask[x[-1]])))
        byframe={p['frame']:p['bboxes'] for p in spatial[vid]['predictions']}; assert set(byframe)==set(range(n))
        rw,rh=m['targetRatioWH']; w=min(m['width'],m['height']*rw/rh); h=w*rh/rw
        box=[(m['width']-w)/2,(m['height']-h)/2,w]
        for kind,mask in [('R26_AHA_EXACT085_B3',exact),('R26_POS_CENTER_EXACT085_B3',center),('R26_R25MASK_CENTER',old)]:
            preds=[dict(frame=int(f),bboxes=box if kind=='R26_R25MASK_CENTER' else byframe[int(f)]) for f in np.flatnonzero(mask)]
            if kind=='R26_R25MASK_CENTER':
                assert [p['frame'] for p in preds]==oldframes
                changed+=sum(not np.allclose(p['bboxes'],byframe[p['frame']],rtol=0,atol=1e-7) for p in preds)
            if kind=='R26_AHA_EXACT085_B3':
                assert all(p['bboxes']==byframe[p['frame']] for p in preds); identity+=len(preds)
            allrows[kind].append(dict(video_id=vid,targetRatioWH=m['targetRatioWH'],model_size_mb=kinds[kind]/1e6,model_params_b=kinds[kind]/1e9,predictions=preds))
        t=x/max(n-1,1)
        rec=dict(info_std=float(v[:,0].std()),rel_std=float(v[:,1].std()),sum_std=float(sc.std()),info_rel_rho=corr(ranks(v[:,0]),ranks(v[:,1])),sum_time_rho=corr(ranks(sc),ranks(t)),info_time_rho=corr(ranks(v[:,0]),ranks(t)),rel_time_rho=corr(ranks(v[:,1]),ranks(t)),first_minus_rest=float(sc[0]-sc[1:].mean()) if len(sc)>1 else 0,unique_fraction=len(np.unique(sc))/len(sc),info_saturation=float(((v[:,0]<.01)|(v[:,0]>.99)).mean()),rel_saturation=float(((v[:,1]<.01)|(v[:,1]>.99)).mean()))
        heads.append(rec)
        for b in range(10):
            ix=np.flatnonzero((t>=b/10)&((t<(b+1)/10) if b<9 else (t<=1)))
            if len(ix): curves[b].append(v[ix].mean(0).tolist())
        budget.append(dict(n=n,old=int(old.sum()),exact=k,excess=int(old.sum())-k,seconds=n/m['fps'],xor_position=int(np.count_nonzero(exact!=center))))
    packages={}
    for name,rows in allrows.items():
        p=out/name; p.mkdir(exist_ok=True); pred=p/'predictions.jsonl'
        pred.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
        independent=check(a.index,pred); contract=validate_submission(rows,load_index(a.index),stage='semifinal',actual_model_params_b=kinds[name]/1e9,require_sorted=True); assert contract.valid,contract.to_dict()
        zpath=p/(name+'.zip')
        with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
            zi=zipfile.ZipInfo('predictions.jsonl',date_time=(1980,1,1,0,0,0)); zi.compress_type=zipfile.ZIP_DEFLATED; zi.external_attr=0o644<<16
            z.writestr(zi,pred.read_bytes(),compresslevel=9)
        with zipfile.ZipFile(zpath) as z:
            assert z.namelist()==['predictions.jsonl']; assert hashlib.sha256(z.read('predictions.jsonl')).hexdigest()==sha(pred)
        manifest=dict(submission_id=name,status='PACKAGED_NOT_UPLOADED',official_score=None,uploaded=False,total_parameters=kinds[name],aha_weight_bytes=(16060832112 if name!='R26_POS_CENTER_EXACT085_B3' else 0),spatial_weight_bytes_reported=(903762857 if name!='R26_R25MASK_CENTER' else 0),precision=('bf16 AHA; cached frozen B3 boxes where used'),parameter_limit=9000000000,model_size_mb_semantics='total inference parameters / 1e6, NOT file bytes',index_sha256=sha(a.index),r25_zip_sha256=r25_manifest['zip_sha256'],spatial_sha256=sha(a.spatial),code_sha256=sha(__file__),predictions_sha256=sha(pred),zip_sha256=sha(zpath),zip_bytes=zpath.stat().st_size,validation=independent,contract=contract.to_dict(),fixed_mask=(name=='R26_R25MASK_CENTER'),same_boxes_as_b3=(name!='R26_R25MASK_CENTER'))
        dump(p/'manifest.json',manifest); (p/(name+'.zip.sha256')).write_text(sha(zpath)+'  '+zpath.name+'\n'); packages[name]={k:v for k,v in manifest.items() if k!='contract'}
    result=dict(scope='Test aggregate diagnostics only; no GT and no accuracy attribution',videos=426,total_frames=sum(b['n'] for b in budget),old_predictions=sum(b['old'] for b in budget),exact_predictions=sum(b['exact'] for b in budget),excess=sum(b['excess'] for b in budget),excess_per_video=dist([b['excess'] for b in budget]),mask_distributions={name:dict(macro_keep=float(np.mean([x['keep'] for x in rr])),macro_keep_by_third=np.mean([x['thirds'] for x in rr],axis=0).tolist(),one_run_videos=sum(x['runs']==1 for x in rr),first_sample_kept=sum(x['first'] for x in rr),last_sample_kept=sum(x['last'] for x in rr)) for name,rr in maskstats.items()},head_diagnostics={k:dist([h[k] for h in heads]) for k in heads[0]},macro_head_time_deciles=[dict(bin=b,n_videos=len(c),info=float(np.mean(c,axis=0)[0]),rel=float(np.mean(c,axis=0)[1]),unc=float(np.mean(c,axis=0)[2])) for b,c in enumerate(curves)],center_spatial_changed_boxes=changed,exact_b3_identity_checked=identity,aha_exact_vs_center_temporal_xor=sum(b['xor_position'] for b in budget),packages=packages,limitations=['No official GT: distributions do not establish ranking error or spatial quality','Official platform scores are pending user upload','Position control eliminates AHA and spatial control eliminates B3; declared inference parameter totals therefore differ','Public VideoXum temporal F1 cannot predict official score'])
    dump(report/'diagnostics.json',result); print(json.dumps({k:v for k,v in result.items() if k not in ['packages','macro_head_time_deciles']},indent=2))

if __name__=='__main__': main()
