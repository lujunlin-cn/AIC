"""Public source-disjoint AHA vs fixed position controls; protected sources unopened."""
import argparse, json, hashlib
from pathlib import Path
import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def evaluate(pred,mask):
    k=int(pred.sum()); tp=(mask&pred).sum(1); gt=mask.sum(1)
    return dict(f=float(np.divide(2*tp,k+gt,out=np.zeros(10),where=(k+gt)>0).mean()),precision=float((tp/max(k,1)).mean()),recall=float(np.divide(tp,gt,out=np.zeros(10),where=gt>0).mean()),kept=k,frames=len(pred))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); a=ap.parse_args(); c=json.loads(Path(a.config).read_text()); root=Path(c['data_root']); out=Path(c['output'])
    report=json.loads((root/'report.json').read_text()); ids=report['qc_summary']['pass_ids_by_split']; axes={r['video_id']:r for r in report['axis']}; rows=[]
    for split in ['train','dev']:
        for vid in ids[split]:
            axis=axes[vid]; apath=root/axis['axis_array_relpath']; assert sha(apath)==axis['axis_array_sha256']
            z=np.load(apath); pts=z['timestamps']; mask=z['raw_summary_membership'].astype(bool)
            d=json.loads((out/'scores_native_shape'/(vid+'.json')).read_text()); ix=np.array(d['frame_indices']); assert mask.shape==(10,len(pts)); assert np.array_equal(pts[ix],d['actual_pts'])
            own=np.abs(pts[:,None]-pts[ix][None,:]).argmin(1)
            rows.append(dict(id=vid,split=split,pts=pts,mask=mask,ix=ix,own=own,values=np.array(d['values'])))
    # Each TRAIN source has equal total weight, no dev labels in this fit.
    gram=np.zeros((4,4)); target=np.zeros(4)
    for r in rows:
        if r['split']!='train': continue
        t=(r['pts']-r['pts'][0])/(r['pts'][-1]-r['pts'][0]); basis=np.stack([t**i for i in range(4)],1)
        gram+=basis.T@basis/len(t); target+=basis.T@r['mask'].mean(0)/len(t)
    coef=np.linalg.lstsq(gram,target,rcond=None)[0]; results={}; contrasts={}
    dev=[r for r in rows if r['split']=='dev']; keeps=[c['primary_keep']]+c['secondary_keep']
    for keep in keeps:
        arms={}
        for r in dev:
            n=len(r['pts']); k=round(keep*n); t=(r['pts']-r['pts'][0])/(r['pts'][-1]-r['pts'][0]); basis=np.stack([t**i for i in range(4)],1)
            masks={}
            for name,score in [('aha_sum',r['values'][:,0]+r['values'][:,1]),('aha_info',r['values'][:,0]),('aha_rel',r['values'][:,1])]:
                exact=np.zeros(n,bool); whole=np.zeros(n,bool)
                for j in np.argsort(-score,kind='stable'):
                    cell=np.flatnonzero(r['own']==j); need=k-int(exact.sum())
                    if need<=0: break
                    whole[cell]=True
                    if len(cell)>need: cell=cell[np.lexsort((cell,np.abs(r['pts'][cell]-r['pts'][r['ix'][j]])))[:need]]
                    exact[cell]=True
                assert exact.sum()==k; masks[name]=exact
                if name=='aha_sum': masks['aha_sum_wholecell']=whole
            for name,order in [('position_prefix',np.arange(n)),('position_suffix',np.arange(n-1,-1,-1)),('position_center',np.argsort(np.abs(np.arange(n)-(n-1)/2),kind='stable')),('position_train_cubic',np.argsort(-(basis@coef),kind='stable'))]:
                pred=np.zeros(n,bool); pred[order[:k]]=True; masks[name]=pred
            randoms=[]
            for seed in range(32):
                rng=np.random.default_rng(seed); pred=np.zeros(n,bool); pred[rng.permutation(n)[:k]]=True; randoms.append(evaluate(pred,r['mask']))
            for name,pred in masks.items(): arms.setdefault(name,[]).append(dict(video_id=r['id'],**evaluate(pred,r['mask'])))
            arms.setdefault('fixed32_random',[]).append(dict(video_id=r['id'],**{key:float(np.mean([rr[key] for rr in randoms])) for key in randoms[0]}))
            arms.setdefault('select_all',[]).append(dict(video_id=r['id'],**evaluate(np.ones(n,bool),r['mask'])))
        results[str(keep)]={name:dict(f=float(np.mean([x['f'] for x in vals])),precision=float(np.mean([x['precision'] for x in vals])),recall=float(np.mean([x['recall'] for x in vals])),videos=vals) for name,vals in arms.items()}
        contrasts[str(keep)]={}
        for name in arms:
            if name=='aha_sum':continue
            d=np.array([x['f']-y['f'] for x,y in zip(arms['aha_sum'],arms[name])]); rng=np.random.default_rng(c['bootstrap_seed']); bs=rng.choice(d,(c['bootstrap_replicates'],len(d)),replace=True).mean(1)
            contrasts[str(keep)]['aha_sum_minus_'+name]=dict(mean=float(d.mean()),ci95=np.quantile(bs,[.025,.975]).tolist(),sources=len(d))
    probe=json.loads((out/'train_protocol_probe.json').read_text()); static=np.array(probe['static_first_frame_repeated']); seq=np.array(probe['chunk1']); chunk=np.array(probe['chunk8'])
    probe_summary=dict(video_id=probe['video_id'],max_abs_all_heads_chunk8_vs1=float(np.abs(seq-chunk).max()),max_abs_score_chunk8_vs1=float(np.abs(seq[:,:2].sum(1)-chunk[:,:2].sum(1)).max()),static_first_score=float(static[0,:2].sum()),static_last_score=float(static[-1,:2].sum()),static_score_range=float(np.ptp(static[:,:2].sum(1))),note='16 identical TRAIN frames, full causal context; changing score with identical image establishes context sensitivity, not task error')
    answer=dict(scope='Public VideoXum original-frame mean-annotator temporal F1 then source macro; NOT official score',config=c,train_ids=ids['train'],dev_ids=ids['dev'],protected_evaluated=False,position_coefficients_train_only=coef.tolist(),results=results,contrasts=contrasts,train_protocol_probe=probe_summary,limitations=['14 dev sources already observed in R24, no fresh lockbox claim','No spatial GT; no official score prediction','Generic frozen R25 query, not title-specific upstream TVSum prompts','No tuning on dev; head alternatives are predefined diagnostics','Bootstrap source sample n14; exploratory multiple contrasts, no multiplicity correction'])
    (out/'public_metrics.json').write_text(json.dumps(answer,indent=2,allow_nan=False)); print(json.dumps({k:{name:v['f'] for name,v in arms.items()} for k,arms in results.items()},indent=2)); print(json.dumps(probe_summary,indent=2))

if __name__=='__main__':main()
