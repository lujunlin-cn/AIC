#!/usr/bin/env python3
"""RetargetVid dense fixed-frame crop evaluation; upstream inclusive-pixel IoU."""
import argparse,ast,hashlib,json,time
from pathlib import Path
import av,cv2,numpy as np
from PIL import Image,ImageDraw
from aic.spatial_pipeline import SpatialPath
def iou(a,b):
    lo=np.maximum(a[...,:2],b[...,:2]);hi=np.minimum(a[...,2:],b[...,2:])
    inter=np.maximum(0,hi-lo+1).prod(axis=-1)
    return inter/((a[...,2:]-a[...,:2]+1).prod(axis=-1)+(b[...,2:]-b[...,:2]+1).prod(axis=-1)-inter)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--videos",required=True);ap.add_argument("--annotations",required=True)
    ap.add_argument("--output",required=True);ap.add_argument("--detector");ap.add_argument("--upstream",required=True)
    a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(2)
    # Isolate the exact upstream pure function, avoiding its CLI side effects.
    tree=ast.parse(Path(a.upstream).read_text());node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=="bb_intersection_over_union")
    ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),"upstream_iou","exec"),ns)
    methods=["center","saliency","subject_proxy","subject_proxy_smooth"]
    if a.detector:methods+=["true_face","true_face_smooth"]
    results=[];manifest=[];cross_errors=[];start=time.perf_counter()
    for p in sorted(Path(a.videos).glob("*.AVI")):
        vid=p.stem;annotations={}
        for ratio in ["1-3","3-1"]:
            annotations[ratio]=[np.loadtxt(Path(a.annotations)/f"annotator_{i}"/f"{vid}_{ratio}.txt",delimiter=",").astype(float) for i in range(1,7)]
            if len({len(x) for x in annotations[ratio]})!=1:raise ValueError("GT lengths differ")
        paths={(r,m):SpatialPath(m,[int(x) for x in r.split("-")],a.detector) for r in annotations for m in methods}
        preds={k:[] for k in paths};resets=[];thumbs=[];frame_count=0
        with av.open(str(p)) as c:
            fps=float(c.streams.video[0].average_rate)
            for frame in c.decode(video=0):
                rgb=frame.to_ndarray(format="rgb24");h,w=rgb.shape[:2]
                for k,path in paths.items():
                    box,reset=path.step(rgb);x,y,cw=box;rw,rh=[int(v) for v in k[0].split("-")]
                    # Upstream GT stores right/bottom boundary and evaluates +1 areas.
                    # Round continuous legal boundary coords directly, preserving that convention.
                    pred=np.rint([x,y,x+cw,y+cw*rh/rw]).astype(int);preds[k].append(pred)
                resets.append(reset)
                if frame_count in [0,60,150,300,450]:
                    im=Image.fromarray(rgb).resize((320,180));draw=ImageDraw.Draw(im)
                    for m,col in [("center","white"),("saliency","cyan"),("subject_proxy_smooth","orange")]:
                        b=preds[("1-3",m)][-1]*np.array([320/w,180/h,320/w,180/h]);draw.rectangle(b.tolist(),outline=col,width=2)
                    thumbs.append(im)
                frame_count+=1
        for ratio,gt in annotations.items():
            if len(gt[0])!=frame_count:raise ValueError(f"{vid}: raw/GT mismatch {frame_count}/{len(gt[0])}")
        manifest.append({"dataset":"DHF1K/RetargetVid","video_id":vid,"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"frame_count":frame_count,"fps":fps,"width":w,"height":h,"split":"public_benchmark_only"})
        for (ratio,mode),pred in preds.items():
            pred=np.asarray(pred);truth=np.asarray(annotations[ratio])
            # The upstream evaluator clamps all negative GT/pred coordinates
            # before calling its inclusive-pixel IoU function.
            truth=np.maximum(truth,0)
            overlap=iou(pred[None],truth);center=(pred[:,:2]+pred[:,2:])/2
            gtcenter=(truth[:,:,:2]+truth[:,:,2:])/2
            center_error=np.linalg.norm(gtcenter-center[None],axis=-1)/np.hypot(w,h)
            diffs={};trajectory=center/np.array([w,h])
            for order,name in [(1,"velocity"),(2,"acceleration"),(3,"jerk")]:
                d=np.diff(trajectory,n=order,axis=0)*fps**order
                valid=np.array([not any(resets[i+1:i+order+1]) for i in range(len(trajectory)-order)])
                diffs[name]=float(np.linalg.norm(d[valid],axis=1).mean()) if valid.any() else None
            illegal=(pred[:,0]<0)|(pred[:,1]<0)|(pred[:,2]>w)|(pred[:,3]>h)|(pred[:,2]<=pred[:,0])|(pred[:,3]<=pred[:,1])
            for frame in [0,frame_count//2,frame_count-1]:
                cross_errors.append(abs(float(overlap[0,frame])-ns["bb_intersection_over_union"](pred[frame],truth[0,frame])))
            rec={"video_id":vid,"ratio":ratio,"mode":mode,"iou":float(overlap.mean()),"center_error_normalized":float(center_error.mean()),"legality":float((~illegal).mean()),**diffs,"face_detection_rate":paths[(ratio,mode)].detections/frame_count,"shot_resets":paths[(ratio,mode)].resets}
            results.append(rec);dest=out/mode;dest.mkdir(exist_ok=True)
            np.savetxt(dest/f"{vid}_{ratio}.txt",pred,fmt="%d",delimiter=",")
        if thumbs:
            sheet=Image.new("RGB",(320*len(thumbs),180))
            for i,im in enumerate(thumbs):sheet.paste(im,(320*i,0))
            sheet.save(out/(vid+"_crops.jpg"))
        print("SPATIAL_DONE",vid,frame_count,flush=True)
    summary={}
    for mode in methods:
        summary[mode]={r:float(np.mean([x["iou"] for x in results if x["mode"]==mode and x["ratio"]==r])) for r in ["1-3","3-1"]}
    report={"protocol":"RETARGETVID_DENSE_NATIVE_IOU_V2","hypothesis":"Gradient/face observations and fixed EMA may improve human crop overlap over center","tuning":"none; first 20 source IDs fixed before GT measurement","video_count":len(manifest),"methods":summary,"per_video":results,"upstream_iou_max_error":max(cross_errors),"detector_bytes":Path(a.detector).stat().st_size if a.detector else 0,"elapsed_seconds":time.perf_counter()-start,"official_f_video":None,"competition_score":None}
    (out/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="per_video"},indent=2))
if __name__=="__main__":main()
