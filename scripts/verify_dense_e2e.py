#!/usr/bin/env python3
"""Verify deployed dense spatial mode equals fixed-frame benchmark crops."""
import argparse,json,time,hashlib,sys,shlex,subprocess,os
from pathlib import Path
import numpy as np,torch
from aic.video import probe_video
from aic.inference import run_inference

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);a=ap.parse_args()
    cfg=json.loads(Path(a.config).read_text());out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    import cv2
    cv2.setNumThreads(2)
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (out/'command.txt').write_text(shlex.join([sys.executable,*sys.argv])+'\n')
    (out/'environment.txt').write_text(sys.version+'\n'+str(torch.__version__)+'\nCUDA_VISIBLE_DEVICES='+os.environ.get('CUDA_VISIBLE_DEVICES','')+'\n'+subprocess.check_output(['nvidia-smi'],text=True))
    p=Path(cfg['video']);info=probe_video(p)
    index=out/'index.jsonl';index.write_text(json.dumps({'video_id':p.stem,'video_path':str(p),
        'width':info.width,'height':info.height,'frame_count':info.frame_count,'targetRatioWH':[1,3]})+'\n')
    results=[]
    device=cfg.get('device','cuda:0')
    use_cuda=str(device).startswith('cuda')
    if use_cuda:torch.cuda.init()
    for job in cfg['jobs']:
        t=time.perf_counter()
        if use_cuda:torch.cuda.reset_peak_memory_stats()
        output=out/(job['id']+'.jsonl')
        report=run_inference(index,output,model_path=job['bundle'],device=device,stage='final',
            threshold=job['threshold'],spatial_mode=job['spatial_mode'],spatial_protocol='dense_v1',detector_path=cfg['detector'])
        gt=np.loadtxt(Path(cfg['spatial_benchmark'])/job['spatial_mode']/(p.stem+'_1-3.txt'),delimiter=',')
        row=json.loads(output.read_text());error=[]
        for x in row['predictions']:
            left,top,width=x['bboxes'];box=np.rint([left,top,left+width,top+width*3])
            error.append(float(np.abs(box-gt[x['frame']]).max()))
        if not error:raise ValueError('use an explicit integration-only threshold to exercise crops')
        if max(error)!=0:raise ValueError('benchmark/deployment spatial mismatch')
        results.append({'job':job,'result':report,'elapsed_seconds':time.perf_counter()-t,
                        'peak_vram_bytes':torch.cuda.max_memory_allocated() if use_cuda else 0,'benchmark_crop_max_error':max(error),
                        'bundle_sha256':hashlib.sha256(Path(job['bundle']).read_bytes()).hexdigest()})
        print(job['id'],'VALID',len(error),flush=True)
    (out/'metrics.json').write_text(json.dumps({'protocol':'DENSE_SPATIAL_E2E_V1','config':cfg,'results':results,
        'note':'threshold 0 exercises all frames, only integration proof; never used for model-performance selection',
        'official_f_video':None,'competition_score':None},indent=2)+'\n')
if __name__=='__main__':main()
