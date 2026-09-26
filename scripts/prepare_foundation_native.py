"""Prepare only native QVH train/dev; reserved holdout is never materialized."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 protocol=json.loads(Path('splits/qvh_native_bounded_v1.json').read_text())
 annotations={}
 for line in (a.root/'qvh_native_annotations/train.jsonl').read_text().splitlines():
  r=json.loads(line);annotations.setdefault(r['vid'],[]).append(r)
 media={p.stem:p for p in (a.root/'video_highlight/train/source_videos').rglob('*.mp4')}
 a.output.mkdir(parents=True,exist_ok=False)
 records=[];files=[]
 for split in ['train','dev']:
  for r in protocol['splits'][split]:
   rows=annotations[r['vid']];duration=rows[0]['duration'];count=int(np.ceil(duration/2))
   sums=np.zeros(count);n=np.zeros(count)
   for q in rows:
    for i,s in zip(q['relevant_clip_ids'],q['saliency_scores']):sums[i]+=np.mean(s)/4.;n[i]+=1
   records.append({**r,'split':split,'video_path':'/data/aic/datasets/QVH_NATIVE_CANDIDATES/videos/'+media[r['vid']].name,'duration':duration,'labels':np.divide(sums,n,out=np.zeros_like(sums),where=n>0).tolist(),'mask':(n>0).tolist(),'queries':rows})
   files.append(str(media[r['vid']].relative_to(a.root/'video_highlight/train/source_videos')))
 (a.output/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
 # rsync with relative source directories then flatten remotely using explicit list
 (a.output/'files.txt').write_text('\n'.join(files)+'\n')
 print(len(records),sum(media[r['vid']].stat().st_size for r in records))
if __name__=='__main__':main()
