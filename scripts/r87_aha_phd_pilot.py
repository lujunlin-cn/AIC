#!/usr/bin/env python3
"""AHA frozen-base target-domain pilot on source-disjoint PHD2 GIF fragments.

This is a diagnostic/adaptation experiment, not an official-test runner.  AHA
produces relevance/informativeness/uncertainty per observed frame; a tiny head
is then trained on PHD2 GIF interval labels with source-level holdout.  Known
unselected PHD2 time is treated as weak negative only for the ranking probe and
is reported as such.
"""
from __future__ import annotations
import argparse, json, math, os, random, time
from collections import defaultdict
from pathlib import Path
import numpy as np


def interval_labels(ann, times):
    out=[]
    for t in times:
        out.append(float(any(a-1e-6 <= t <= b+1e-6 for a,b in ann)))
    return np.asarray(out,np.float32)

def load_labels(path):
    return {x['video_id']: x for x in (json.loads(l) for l in Path(path).read_text().splitlines() if l.strip())}

def make_frames(paths):
    from PIL import Image
    arr=[]
    for p in paths:
        im=Image.open(p).convert('RGB')
        # Processor accepts a tensor; use a uniform canvas and preserve aspect.
        w,h=im.size; s=384; sc=min(s/w,s/h); nw,nh=max(1,round(w*sc)),max(1,round(h*sc))
        im=im.resize((nw,nh)); canvas=Image.new('RGB',(s,s),(0,0,0)); canvas.paste(im,((s-nw)//2,(s-nh)//2))
        arr.append(np.asarray(canvas).transpose(2,0,1))
    import torch
    return torch.from_numpy(np.stack(arr)).contiguous()

def rank_ap(sc,y):
    sc=np.asarray(sc,float).reshape(-1); y=np.asarray(y,float).reshape(-1); pos=y.sum()
    if pos<=0:return float('nan')
    order=np.argsort(-sc,kind='mergesort'); yy=y[order]; prec=np.cumsum(yy)/np.arange(1,len(yy)+1)
    return float((prec*yy).sum()/pos)

def source_metrics(rows, scores, kfrac=.85):
    by=defaultdict(list)
    for r,s in zip(rows,scores):
        y=np.asarray(r['y'],float); k=max(1,int(round(kfrac*len(y)))); idx=np.argsort(-s,kind='mergesort')[:k]
        by[r['src']].append((float(y[idx].sum()/max(y.sum(),1)),float(y[idx].mean()),float(y.mean()),float(rank_ap(s,y))))
    a=np.asarray([np.mean(v,0) for v in by.values()])
    return {'sources':len(a),'recall_at_keep':float(a[:,0].mean()),'precision_at_keep':float(a[:,1].mean()),'base_rate':float(a[:,2].mean()),'ap_source_macro':float(a[:,3].mean())}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--pool',type=Path,required=True); ap.add_argument('--labels',type=Path,required=True); ap.add_argument('--val-sources',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--limit-train',type=int,default=96); ap.add_argument('--limit-val',type=int,default=48); ap.add_argument('--seed',type=int,default=20261019)
    a=ap.parse_args(); random.seed(a.seed); np.random.seed(a.seed)
    pool=a.pool; labels=load_labels(a.labels); valset=set(json.loads(a.val_sources.read_text()))
    allrows=[]
    for line in (pool/'index_v2.jsonl').read_text().splitlines():
        if not line.strip():continue
        r=json.loads(line); fdir=pool/'frames'/r['video_id']; lab=labels.get(r['video_id']);
        if lab is None: continue
        file_to_label={x['file']:float(x.get('soft',x.get('hard',0))) for x in lab['frames']}
        files=sorted([p for p in fdir.glob('*.jpg') if p.name in file_to_label],key=lambda p:float(p.stem))
        if len(files)<8:continue
        times=[float(p.stem) for p in files[:8]]
        allrows.append({'vid':r['video_id'],'src':r['src'],'files':[str(x) for x in files[:8]],'times':times,'y':np.asarray([file_to_label[x.name] for x in files[:8]],np.float32),'rot':bool(r.get('rotated'))})
    rng=np.random.default_rng(a.seed); bysrc=defaultdict(list)
    for r in allrows:bysrc[r['src']].append(r)
    train=[r for r in allrows if r['src'] not in valset]; val=[r for r in allrows if r['src'] in valset]
    rng.shuffle(train); rng.shuffle(val); train=train[:a.limit_train]; val=val[:a.limit_val]
    if not train or not val: raise SystemExit(f'empty split train={len(train)} val={len(val)}')
    print(json.dumps({'all_rows':len(allrows),'train':len(train),'val':len(val),'train_sources':len(set(r['src'] for r in train)),'val_sources':len(set(r['src'] for r in val))}),flush=True)

    import torch
    import torch
    from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower, SigLipVisionModel, SigLipVisionConfig
    def _local_load(self, device_map=None):
        if self.is_loaded: return
        self.vision_tower=SigLipVisionModel(SigLipVisionConfig())
        del self.vision_tower.vision_model.encoder.layers[-1:]
        self.vision_tower.vision_model.head=torch.nn.Identity()
        self.vision_tower.requires_grad_(False); self.is_loaded=True
    SigLipVisionTower.load_model=_local_load
    from models.live_llava.video_head_live_llava_qwen import build_video_head_live_llava_qwen
    base='/root/autodl-tmp/pretrained/llava-onevision-qwen2-7b-ov'; adapter='/root/autodl-tmp/pretrained/aha_weights'
    model, tokenizer=build_video_head_live_llava_qwen(is_training=False,llm_pretrained=base,lora_pretrained=adapter,vision_pretrained='google/siglip-large-patch16-384',frame_resolution=384,frame_num_tokens=49,frame_token_cls=False,frame_token_pooled=[7,7],video_pooling_stride=4,attn_implementation='eager',torch_dtype=torch.float16,set_vision_inside=False,quantization=False)
    model=model.to('cuda').eval(); vt=model.get_vision_tower(); processor=vt.image_processor
    hidden=model.config.hidden_size; start_ids=tokenizer.apply_chat_template([{'role':'system','content':'A multimodal AI assistant is helping users with some activities. Below is their conversation, interleaved with the list of video frames received by the assistant.'}],return_tensors='pt').to('cuda')
    from test.sink_cache import SinkCache
    print('AHA_LOADED',flush=True)
    outrows=[]
    for i,r in enumerate(train+val):
        t0=time.time(); frames=make_frames(r['files'])
        with torch.no_grad():
            pv=processor.preprocess([x.permute(1,2,0).numpy() for x in frames],return_tensors='pt')['pixel_values'].to('cuda',dtype=torch.float16)
            embeds=model.visual_embed(pv).split(model.config.frame_num_tokens)
        past=SinkCache(window_length=2048,num_sink_tokens=32); scores=[]
        for fe in embeds:
            # A single EOS token keeps the 4.49 eager path numerically stable
            # on the cloned environment; the upstream demo used an empty
            # embedding, which yields NaNs for some fragments here.
            ids=start_ids if past.get_seq_length() == 0 else torch.tensor([[tokenizer.eos_token_id]],device='cuda',dtype=torch.long)
            ie=model.get_input_embeddings()(ids).view(1,-1,hidden)
            inp=torch.cat([ie,fe.view(1,-1,hidden).to('cuda')],dim=1)
            with torch.no_grad(): out=model(inputs_embeds=inp,use_cache=True,past_key_values=past,return_dict=True)
            past=out.past_key_values
            rel=float(out.relevance_logits[0,-1].reshape(-1)[0]); info=float(out.informative_logits[0,-1].softmax(-1)[1]); unc=float(out.uncertainty[0,-1].reshape(-1)[0])
            scores.append([rel,info,unc])
        outrows.append({'vid':r['vid'],'src':r['src'],'split':'val' if r in val else 'train','y':r['y'].tolist(),'scores':scores,'times':r['times'],'seconds':time.time()-t0})
        print(json.dumps({'done':i+1,'total':len(train)+len(val),'vid':r['vid'],'split':outrows[-1]['split'],'seconds':round(outrows[-1]['seconds'],2)}),flush=True)
    a.out.mkdir(parents=True,exist_ok=True); (a.out/'raw_scores.json').write_text(json.dumps({'rows':outrows,'seed':a.seed,'official_test_labels_used':False},indent=1))
    finite_fraction=float(np.isfinite(np.asarray([q['scores'] for q in outrows],dtype=float)).mean())
    if finite_fraction < 1.0:
        report={'status':'blocked_by_runtime_nan','finite_fraction':finite_fraction,
                'rows':len(outrows),'train_rows':len(train),'val_rows':len(val),
                'message':'consecutive-frame AHA outputs are non-finite; no ranking or adaptation metric is valid',
                'official_test_labels_used':False}
        (a.out/'metrics.json').write_text(json.dumps(report,indent=1)+'\n')
        print(json.dumps(report),flush=True); return
    tr=[x for x in outrows if x['split']=='train']; va=[x for x in outrows if x['split']=='val']
    def base(rows,j): return [np.asarray(x['scores'])[:,j] for x in rows]
    report={'config':{'seed':a.seed,'train_rows':len(tr),'val_rows':len(va),'AHA_base_frozen':True,'weak_interval_labels':True},'val':{}}
    for name,j in [('relevance',0),('informative',1),('uncertainty',2)]:
        report['val'][name]={'ap_row':float(np.nanmean([rank_ap(s,x['y']) for s,x in zip(base(va,j),va)])),'source_macro':source_metrics(va,base(va,j))}
    # Tiny ridge/logistic-style linear head on 3 AHA scores + normalized time and finite differences.
    def feat(x):
        z=np.asarray(x['scores'],np.float32); t=np.linspace(0,1,len(z),dtype=np.float32)[:,None]; d=np.vstack([np.zeros((1,z.shape[1]),np.float32),np.diff(z,axis=0)]); return np.concatenate([z,t,d],1)
    Xtr=np.concatenate([feat(x) for x in tr]); Ytr=np.concatenate([np.asarray(x['y'],np.float32) for x in tr]); Xv=np.concatenate([feat(x) for x in va]); Yv=np.concatenate([np.asarray(x['y'],np.float32) for x in va])
    mu=Xtr.mean(0); sd=Xtr.std(0); sd[sd<1e-5]=1; Xtr=(Xtr-mu)/sd; Xv=(Xv-mu)/sd
    from sklearn.linear_model import Ridge
    head=Ridge(alpha=10.0).fit(Xtr,Ytr); pred=[head.predict(feat(x)*0 + (feat(x)-mu)/sd) for x in va]
    report['val']['adapted_linear']={'ap_row':float(np.nanmean([rank_ap(s,x['y']) for s,x in zip(pred,va)])),'source_macro':source_metrics(va,pred)}
    report['head']={'type':'ridge','alpha':10.0,'features':'AHA relevance, informative, log uncertainty, normalized time, first differences','coef_shape':list(head.coef_.shape)}
    (a.out/'metrics.json').write_text(json.dumps(report,indent=1)); print(json.dumps(report,indent=1),flush=True)
if __name__=='__main__':main()
