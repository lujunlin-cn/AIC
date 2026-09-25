#!/usr/bin/env python3
"""Deterministic raw-frame evidence for temporal/spatial failure analysis."""
import argparse,json
from pathlib import Path
import av,numpy as np
from PIL import Image,ImageDraw

def frames_at(path,indices):
    wanted=set(indices);frames={}
    with av.open(str(path)) as c:
        for i,f in enumerate(c.decode(video=0)):
            if i in wanted:frames[i]=f.to_ndarray(format='rgb24')
            if len(frames)==len(wanted):break
    return frames

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);a=ap.parse_args();cfg=json.loads(Path(a.config).read_text())
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    manifest={r['video_id']:r for r in [json.loads(x) for x in Path(cfg['tvsum_manifest']).read_text().splitlines()]}
    for vid in cfg['temporal_ids']:
        r=manifest[vid];idx=np.linspace(0,r['frame_count']-1,6,dtype=int);images=frames_at(r['path'],idx)
        sheet=Image.new('RGB',(360*3,230*2),'#eeeeee');draw=ImageDraw.Draw(sheet)
        for j,i in enumerate(idx):
            im=Image.fromarray(images[i]);im.thumbnail((360,202));x=(j%3)*360;y=(j//3)*230
            sheet.paste(im,(x,y));draw.text((x+5,y+205),f'{vid} frame={i} t={i/r["fps"]:.1f}s',fill='black')
        sheet.save(out/(vid+'_frames.jpg'),quality=90)
    result=json.loads(Path(cfg['spatial_results']).read_text());rows=result['per_video'];ids=sorted({r['video_id'] for r in rows})
    deltas={vid:np.mean([r['iou'] for r in rows if r['video_id']==vid and r['mode']=='true_face_smooth'])-np.mean([r['iou'] for r in rows if r['video_id']==vid and r['mode']=='center']) for vid in ids}
    chosen=sorted(ids,key=lambda v:deltas[v]);selected=chosen[:2]+chosen[-2:];info=[]
    colors={'center':'white','saliency':'cyan','subject_proxy_smooth':'orange','true_face_smooth':'red'}
    for vid in selected:
        gt=np.loadtxt(Path(cfg['spatial_annotations'])/'annotator_1'/(vid+'_1-3.txt'),delimiter=',');idx=np.linspace(0,len(gt)-1,4,dtype=int)
        images=frames_at(Path(cfg['spatial_videos'])/(vid+'.AVI'),idx);sheet=Image.new('RGB',(480*2,310*2),'#eeeeee');draw=ImageDraw.Draw(sheet)
        boxes={mode:np.loadtxt(Path(cfg['spatial_output'])/mode/(vid+'_1-3.txt'),delimiter=',') for mode in colors}
        for j,i in enumerate(idx):
            rgb=images[i];h,w=rgb.shape[:2];im=Image.fromarray(rgb).resize((480,270));paint=ImageDraw.Draw(im);scale=np.array([480/w,270/h]*2)
            paint.rectangle((gt[i]*scale).tolist(),outline='lime',width=3)
            for mode,color in colors.items():paint.rectangle((boxes[mode][i]*scale).tolist(),outline=color,width=2)
            x=(j%2)*480;y=(j//2)*310;sheet.paste(im,(x,y));draw.text((x+5,y+273),f'{vid} frame={i} human=green center=white face=red',fill='black')
            draw.text((x+5,y+288),'saliency=cyan proxy+EMA=orange',fill='black')
        sheet.save(out/(vid+'_human_crops.jpg'),quality=90);info.append({'video_id':vid,'paired_iou_delta':deltas[vid]})
    (out/'spatial_selected.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info))
if __name__=='__main__':main()
