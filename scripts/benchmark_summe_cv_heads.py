#!/usr/bin/env python3
"""OOD replication across every TVSum CV head; never select a head using SumMe."""
import argparse,hashlib,json,time,sys,subprocess,shlex,os
from pathlib import Path
import numpy as np,torch
from scipy.io import loadmat
from aic.models import load_inference_model
from aic.features import _letterbox
from aic.video import probe_video
from aic.annotated_video import iter_annotated_cfr_frames
from aic.temporal_metrics import ranking_report,summary_mask
from scripts.benchmark_representations import load_head

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--model',required=True)
    a=ap.parse_args();cfg=json.loads(Path(a.config).read_text());job=cfg['models'][a.model]
    out=Path(cfg['output'])/a.model;out.mkdir(parents=True,exist_ok=False)
    cache=Path(cfg['features'])/a.model;cache.mkdir(parents=True,exist_ok=False)
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (out/'command.txt').write_text(shlex.join([sys.executable,*sys.argv])+'\n')
    (out/'environment.txt').write_text(sys.version+'\n'+str(torch.__version__)+'\nCUDA_VISIBLE_DEVICES='+os.environ.get('CUDA_VISIBLE_DEVICES','')+'\n'+subprocess.check_output(['nvidia-smi'],text=True))
    torch.set_num_threads(4);model,meta=load_inference_model(job['bundle'],'cuda:0')
    # Restore the exact FP32 initialization used for TVSum feature extraction.
    state=torch.load(job['backbone_weights'],map_location='cpu',weights_only=True)
    if 'state_dict' in state:state=state['state_dict']
    if a.model=='ViT_B':state={k:v for k,v in state.items() if not k.startswith('heads.')}
    model.backbone.load_state_dict(state,strict=True);model.eval();data=[];audit=[]
    for p in sorted((Path(cfg['root'])/'videos').glob('*.mp4')):
        info=probe_video(p);g=loadmat(Path(cfg['root'])/'GT'/(p.stem+'.mat'));n=int(g['nFrames'].item())
        audit.append({'video_id':p.stem,'decoded_frames':info.frame_count,'gt_frames':n,'included':info.frame_count==n,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        if info.frame_count!=n:continue
        samples=list(iter_annotated_cfr_frames(p,annotation_fps=float(g['FPS'].item()),annotation_count=n))
        with torch.inference_mode():
            features=torch.cat([model.encode_frames(_letterbox(np.stack([x[2] for x in samples[j:j+16]])).to('cuda:0')).float().cpu() for j in range(0,len(samples),16)])
        indices=np.array([x[0] for x in samples]);truth=g['user_score']>0;rel=g['gt_score'].ravel()
        np.savez_compressed(cache/(p.stem+'.npz'),features=features.numpy(),frame_indices=indices,relevance=rel,human=truth)
        data.append((p.stem,features.to('cuda:0')[None],indices,truth,rel))
    del model,state;torch.cuda.empty_cache();rows=[];start=time.perf_counter()
    checkpoints=sorted(Path(cfg['cv_root']).glob('RG_NCV_'+a.model+'_*/best.pt'))
    if len(checkpoints)!=30:raise ValueError('expected all 30 paired CV heads')
    for ck in checkpoints:
        head,shift,_=load_head(ck,'cuda:0');assert not shift
        selection=json.loads((ck.parent/'selection.json').read_text());per=[]
        with torch.inference_mode():
            for vid,x,idx,truth,rel in data:
                score=head(x)[0].sigmoid().cpu().numpy();full=np.interp(np.arange(len(rel)),idx,score)
                mask=summary_mask((full-full.min())/np.ptp(full) if np.ptp(full) else np.zeros_like(full))
                den=truth.sum(axis=0)+mask.sum();f=np.divide(2*(truth&mask[:,None]).sum(axis=0),den,out=np.zeros(truth.shape[1]),where=den>0)
                per.append({'video_id':vid,'native_mean_user_f1':float(f.mean()),'native_max_user_f1':float(f.max()),**ranking_report(full,rel),
                    'threshold':selection['chosen_threshold'],'prediction_rate':float((full>=selection['chosen_threshold']).mean())})
        rec={'run_id':ck.parent.name,'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest(),'per_video':per}
        rows.append(rec);(out/(ck.parent.name+'.json')).write_text(json.dumps(rec,indent=2)+'\n')
    report={'protocol':'SUMME_PAIRED_CV_HEADS_OOD_V1','model':a.model,'config':cfg,'manifest':audit,'runs':rows,
        'selection':'all 30 TVSum-inner-selected heads; no OOD head, threshold or postprocess selection; no ensemble',
        'decode_caveat':'annotation-ordinal CFR due to broken mirror PTS; two frame-count mismatches excluded',
        'backbone_sha256':hashlib.sha256(Path(job['backbone_weights']).read_bytes()).hexdigest(),
        'head_evaluation_seconds':time.perf_counter()-start,'official_f_video':None,'competition_score':None}
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n');print(a.model,'COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()
