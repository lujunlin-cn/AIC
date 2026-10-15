"""Fresh bound features and source-disjoint point/block comparison. No test assets."""
import argparse, hashlib, json, time, sys, gc
from pathlib import Path
import numpy as np
import torch
from torch import nn

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def dump(p,x):
    Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))

def evaluate(scores, rows, keep):
    result=[]
    for s,r in zip(scores,rows):
        # Stable block ranking; original-frame order resolves within-block ties.
        n=r['mask'].shape[1]; k=int(round(n*keep))
        order=np.argsort(-np.asarray(s)[r['block']],kind='stable')
        pred=np.zeros(n,bool); pred[order[:k]]=True
        tp=(r['mask'] & pred).sum(1); gt=r['mask'].sum(1)
        f=np.divide(2*tp,k+gt,out=np.zeros(10,float),where=(k+gt)>0)
        pr=tp/max(k,1); re=np.divide(tp,gt,out=np.zeros(10,float),where=gt>0)
        result.append(dict(video_id=r['id'],f=float(f.mean()),precision=float(pr.mean()),recall=float(re.mean()),kept=k,frames=n,gt_density=float(r['mask'].mean()),scores=np.asarray(s).tolist()))
    return dict(f=float(np.mean([v['f'] for v in result])),videos=result)

def paired(a,b):
    d=np.array([x['f']-y['f'] for x,y in zip(a['videos'],b['videos'])])
    rng=np.random.default_rng(20261015)
    bs=rng.choice(d,(10000,len(d)),replace=True).mean(1)
    return dict(mean=float(d.mean()),ci95=np.quantile(bs,[.025,.975]).tolist())

def main():
    import cv2
    from PIL import Image
    from transformers import CLIPVisionModel, CLIPImageProcessor
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); args=ap.parse_args()
    c=json.loads(Path(args.config).read_text()); out=Path(c['output']); out.mkdir(parents=True,exist_ok=True)
    dump(out/'config.json',c); torch.set_num_threads(4)
    root=Path(c['data_root']); report=json.loads((root/'report.json').read_text())
    splits=report['qc_summary']['pass_ids_by_split']; assert not set(splits['train']) & set(splits['dev'])
    axes={r['video_id']:r for r in report['axis']}; rows=[]; bindings=[]
    for split in ['train','dev']:
        for vid in splits[split]:
            a=axes[vid]; media=root/a['media_relpath']; ax=root/a['axis_array_relpath']
            assert sha(media)==a['media_sha256'] and sha(ax)==a['axis_array_sha256']
            z=np.load(ax); pts=z['timestamps']; mask=z['raw_summary_membership'].astype(bool)
            assert mask.shape==(10,len(pts))
            targets=np.linspace(pts[0],pts[-1],c['samples'])
            ix=np.abs(pts[:,None]-targets).argmin(0); assert len(set(ix.tolist()))==c['samples']
            block=np.abs(pts[:,None]-pts[ix]).argmin(1)
            cap=cv2.VideoCapture(str(media)); frames=[]; selected=set(ix.tolist()); j=0
            while True:
                ok,bgr=cap.read()
                if not ok: break
                if j in selected: frames.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
                j+=1
            cap.release(); assert j==len(pts) and len(frames)==c['samples']
            rows.append(dict(id=vid,split=split,mask=mask,block=block,frames=frames,
                point=mask[:,ix].mean(0),block_target=np.array([mask[:,block==i].mean() for i in range(c['samples'])])))
            bindings.append(dict(video_id=vid,split=split,media_sha256=a['media_sha256'],axis_sha256=a['axis_array_sha256'],sample_frame_ids=ix.tolist(),actual_pts=pts[ix].tolist()))
    dump(out/'bindings.json',bindings)
    sys.path.insert(0,'/root/autodl-tmp/vpb')
    from videoprism_torch import build_base
    wp=Path('/root/autodl-tmp/vpb/videoprism_base_f16r288.fp16.pt')
    model=build_base(ckpt_path=str(wp)).cuda().half().eval()
    started=time.time()
    with torch.inference_mode():
        for r in rows:
            x=np.stack([cv2.resize(f,(288,288),interpolation=cv2.INTER_AREA) for f in r['frames']])
            t,_=model(torch.from_numpy(x).cuda().half()[None]/255,return_spatial=True)
            r['vpb']=t.float().mean(2).squeeze(0).cpu().numpy()
            print('VPB',r['id'],r['vpb'].shape,flush=True)
    model_meta={'vpb':dict(weight_sha256=sha(wp),weight_bytes=wp.stat().st_size,preprocess='RGB /255, cv2 INTER_AREA 288 square',seconds=time.time()-started)}
    del model; gc.collect(); torch.cuda.empty_cache()
    cp=Path('/root/autodl-tmp/vpb/clip-vit-base-patch16')
    model=CLIPVisionModel.from_pretrained(cp,local_files_only=True).cuda().half().eval()
    proc=CLIPImageProcessor.from_pretrained(cp,local_files_only=True)
    started=time.time()
    with torch.inference_mode():
        for r in rows:
            x=proc(images=[Image.fromarray(f) for f in r['frames']],return_tensors='pt')['pixel_values'].cuda().half()
            r['clip']=model(pixel_values=x).pooler_output.float().cpu().numpy()
            assert r['clip'].shape==r['vpb'].shape==(c['samples'],768)
    files=[p for p in cp.iterdir() if p.suffix in ['.bin','.safetensors']]
    model_meta['clip']=dict(weights={p.name:sha(p) for p in files},weight_bytes=sum(p.stat().st_size for p in files),preprocess=proc.to_dict(),output='vision pooler_output768, no projection adapter',seconds=time.time()-started)
    del model; gc.collect(); torch.cuda.empty_cache()
    for r in rows:
        np.savez_compressed(out/(r['id']+'.npz'),vpb=r['vpb'],clip=r['clip'],point=r['point'],block=r['block_target'])
        del r['frames']
    dump(out/'feature_contract.json',model_meta)
    train=[r for r in rows if r['split']=='train']; dev=[r for r in rows if r['split']=='dev']
    results={}; keeps=[c['primary_keep']]+c['secondary_keep']
    for backbone in ['vpb','clip']:
        x=np.stack([r[backbone] for r in train]); mu=x.mean((0,1)); sd=x.std((0,1)).clip(.0001)
        tx=torch.tensor((x-mu)/sd,device='cuda'); dx=torch.tensor((np.stack([r[backbone] for r in dev])-mu)/sd,device='cuda')
        for target in ['point','block']:
            y=torch.tensor(np.stack([r['point' if target=='point' else 'block_target'] for r in train]),device='cuda',dtype=torch.float32)
            for seed in c['seeds']:
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                head=nn.Sequential(nn.Linear(768,512),nn.GELU(),nn.Linear(512,1)).cuda()
                opt=torch.optim.AdamW(head.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
                losses=[]; start=time.time()
                for epoch in range(c['epochs']):
                    head.train(); opt.zero_grad(); loss=nn.functional.binary_cross_entropy_with_logits(head(tx).squeeze(-1),y)
                    loss.backward(); opt.step(); losses.append(float(loss))
                head.eval()
                with torch.inference_mode(): scores=head(dx).squeeze(-1).sigmoid().cpu().numpy()
                name=f'{backbone}_{target}_s{seed}'; path=out/(name+'.pt')
                torch.save(dict(head=head.cpu().state_dict(),mu=mu,sd=sd,config=c),path)
                results[name]={str(k):evaluate(scores,dev,k) for k in keeps}
                results[name]['training']=dict(seconds=time.time()-start,losses=losses,parameters=sum(p.numel() for p in head.parameters()),checkpoint_bytes=path.stat().st_size,sha256=sha(path))
                print(name,{k:v['f'] for k,v in results[name].items() if k!='training'},flush=True)
    # Position-only polynomial regression uses training block labels only.
    pos=np.linspace(0,1,c['samples']); basis=np.stack([pos**i for i in range(4)],1)
    coef=np.linalg.lstsq(np.tile(basis,(len(train),1)),np.stack([r['block_target'] for r in train]).reshape(-1),rcond=None)[0]
    controls={}
    for k in keeps:
        controls[str(k)]={'position_only':evaluate(np.tile(basis@coef,(len(dev),1)),dev,k),
                          'equal_score_fixed_block':evaluate(np.zeros((len(dev),c['samples'])),dev,k),
                          'select_all':evaluate(np.zeros((len(dev),c['samples'])),dev,1)}
        runs=[]
        for seed in range(32):
            rng=np.random.default_rng(seed); vals=[]
            for r in dev:
                n=r['mask'].shape[1]; kk=round(n*k); pred=np.zeros(n,bool); pred[rng.permutation(n)[:kk]]=True
                tp=(r['mask']&pred).sum(1); vals.append(float((2*tp/(kk+r['mask'].sum(1))).mean()))
            runs.append(vals)
        controls[str(k)]['fixed32_random']={'f':float(np.mean(runs)),'per_source_f':np.mean(runs,axis=0).tolist(),'seed_macro_f':np.mean(runs,axis=1).tolist()}
    contrasts={}
    for seed in c['seeds']:
        contrasts[f'block_minus_point_s{seed}']=paired(results[f'vpb_block_s{seed}'][str(keeps[0])],results[f'vpb_point_s{seed}'][str(keeps[0])])
        contrasts[f'clip_minus_vpb_block_s{seed}']=paired(results[f'clip_block_s{seed}'][str(keeps[0])],results[f'vpb_block_s{seed}'][str(keeps[0])])
    diffs=[contrasts[f'block_minus_point_s{s}']['mean'] for s in c['seeds']]
    control_max=max(v['f'] for v in controls[str(keeps[0])].values())
    gate=np.mean(diffs)>=.005 and min(diffs)>=0 and all(results[f'vpb_block_s{s}'][str(keeps[0])]['f']>control_max for s in c['seeds'])
    dump(out/'metrics.json',dict(scope='Selected source-disjoint public VideoXum dev temporal only; not official score; no globally fresh claim',train_ids=[r['id'] for r in train],dev_ids=[r['id'] for r in dev],results=results,controls=controls,contrasts=contrasts,block_gate_pass=bool(gate),protected_evaluated=False,limitations=['No crop GT or spatial metric','No reliable domain annotations','No dedicated no-highlight dataset','12 samples across whole video','No epoch selection; fixed300'],peak_vram_bytes=torch.cuda.max_memory_allocated()))

if __name__=='__main__': main()
