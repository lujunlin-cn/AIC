"""Score recovered fixed-frame predictions, preserving source paths and hashes."""
import argparse, ast, hashlib, json
from pathlib import Path
import av, cv2, numpy as np
from scripts.benchmark_spatial import iou
from scripts.analyze_nested_cv import paired_bootstrap


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True)
    a=ap.parse_args(); cfg=json.loads(Path(a.config).read_text())
    rows=[]; inputs=[]; max_error=0.; cv2.setNumThreads(2)
    tree=ast.parse(Path(cfg['upstream']).read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='bb_intersection_over_union')
    ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),'upstream','exec'),ns)
    manifest=json.loads(Path(cfg['manifest']).read_text())
    for source in manifest:
        vid=source['video_id']; n=source['frame_count']; w,h=source['width'],source['height']
        resets=[]; previous=None
        video=Path(source['path'])
        if hashlib.sha256(video.read_bytes()).hexdigest()!=source['sha256']:raise ValueError('source SHA mismatch')
        with av.open(str(video)) as c:
            for f in c.decode(video=0):
                thumb=cv2.resize(f.to_ndarray(format='rgb24'),(64,36)).astype(np.float32)/255
                resets.append(previous is None or float(np.abs(thumb-previous).mean())>.25)
                previous=thumb
        if len(resets)!=n:raise ValueError('frame count mismatch')
        for ratio in ['1-3','3-1']:
            truth=np.maximum(0,np.asarray([np.loadtxt(Path(cfg['annotations'])/f'annotator_{i}'/f'{vid}_{ratio}.txt',delimiter=',') for i in range(1,7)]))
            if truth.shape!=(6,n,4):raise ValueError('GT shape mismatch')
            for mode,roots in cfg['prediction_roots'].items():
                files=[Path(root)/mode/f'{vid}_{ratio}.txt' for root in roots]
                files=[p for p in files if p.is_file()]
                if len(files)!=1:raise ValueError(f'exactly one source required: {mode}/{vid}/{ratio}: {files}')
                p=files[0];pred=np.loadtxt(p,delimiter=',')
                if pred.shape!=(n,4) or not np.isfinite(pred).all():raise ValueError('invalid predictions')
                legal=(pred[:,0]>=0)&(pred[:,1]>=0)&(pred[:,2]<=w)&(pred[:,3]<=h)&(pred[:,2]>pred[:,0])&(pred[:,3]>pred[:,1])
                if not legal.all():raise ValueError('illegal crop')
                overlap=iou(pred[None],truth)
                for j in [0,n//2,n-1]:
                    max_error=max(max_error,abs(float(overlap[0,j])-ns['bb_intersection_over_union'](pred[j],truth[0,j])))
                center=(pred[:,:2]+pred[:,2:])/2
                error=np.linalg.norm((truth[:,:,:2]+truth[:,:,2:])/2-center[None],axis=-1)/np.hypot(w,h)
                row={'video_id':vid,'ratio':ratio,'mode':mode,'iou':float(overlap.mean()),'center_error_normalized':float(error.mean()),'legality':float(legal.mean())}
                for order,name in [(1,'velocity'),(2,'acceleration'),(3,'jerk')]:
                    delta=np.diff(center/np.array([w,h]),n=order,axis=0)*source['fps']**order
                    valid=np.array([not any(resets[i+1:i+order+1]) for i in range(n-order)])
                    row[name]=float(np.linalg.norm(delta[valid],axis=1).mean()) if valid.any() else None
                rows.append(row);inputs.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    modes=list(cfg['prediction_roots']); ids=[m['video_id'] for m in manifest]
    per={mode:{vid:float(np.mean([r['iou'] for r in rows if r['mode']==mode and r['video_id']==vid])) for vid in ids} for mode in modes}
    pairs={}
    for model,control in [('true_face_group','true_face'),('true_face_group_smooth','true_face_smooth'),('true_face_group_smooth','center')]:
        pairs[model+'-'+control]=paired_bootstrap([per[control][v] for v in ids],[per[model][v] for v in ids])
    report={'protocol':'RETARGETVID_GROUP_DEVELOPMENT_COMPARISON_V1','config':cfg,'video_count':len(ids),
            'methods':{mode:{ratio:float(np.mean([r['iou'] for r in rows if r['mode']==mode and r['ratio']==ratio])) for ratio in ['1-3','3-1']} for mode in modes},
            'paired':pairs,'per_video':rows,'source_predictions':inputs,'upstream_iou_max_error':max_error,
            'exposure':'All 20 videos previously exposed; this is not a new independent test',
            'official_f_video':None,'competition_score':None}
    Path(cfg['output']).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'methods':report['methods'],'paired':pairs},indent=2))
if __name__=='__main__':main()
