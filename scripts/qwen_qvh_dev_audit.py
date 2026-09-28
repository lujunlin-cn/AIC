"""P2 Qwen temporal audit on native QVH dev (labels, never AIC test).

Same teacher prompt/settings as the official SUB_Q run (1 fps anchors, 448 px,
block 32, vLLM greedy).  Selected frames are mapped to 2 s clips (a clip counts
as selected when >=50% of its frames are selected) and scored with the same
set-F1 proxies as scripts/all_select_diagnosis.py, against all-select and a
volume-matched uniform-stride control (does any gain come only from output
volume?).
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from aic.qwen_teacher import QwenTeacher,build_anchors,extract_anchor_frames,segments_to_frames
from aic.video import frame_timeline,probe_video
from scripts.all_select_diagnosis import f_set,boot


def clips_from_frames(selected,timeline,n):
 hit=np.zeros(n);tot=np.zeros(n);sel=set(selected)
 for s in timeline:
  c=min(int(s.time_seconds/2),n-1);tot[c]+=1;hit[c]+=s.index in sel
 return (hit/np.maximum(tot,1))>=.5


def main():
 p=argparse.ArgumentParser();p.add_argument('--records',required=True);p.add_argument('--model',required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--ratio',default='9,16');p.add_argument('--tensor-parallel-size',type=int,default=4);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=True);cache=a.output/'selections';cache.mkdir(exist_ok=True)
 recs=[r for r in map(json.loads,open(a.records)) if r['split']=='dev'];ratio=[int(x) for x in a.ratio.split(',')]
 teacher=QwenTeacher(model_path=a.model,backend='vllm',image_size=448,block_size=32,max_new_tokens=512,tensor_parallel_size=a.tensor_parallel_size,gpu_memory_utilization=.85,max_num_seqs=16)
 rows=[];start=time.time()
 for r in recs:
  cf=cache/(r['vid']+'.json');labels=np.asarray(r['labels']);mask=np.asarray(r['mask'],bool);n=len(labels)
  timeline=frame_timeline(r['video_path'])
  if cf.is_file():c=json.loads(cf.read_text())
  else:
   t0=time.time();info=probe_video(r['video_path']);anchors=build_anchors(r['video_path'],1.)
   frames=extract_anchor_frames(r['video_path'],anchors,size=448);segs,status,raw=teacher.select_segments(frames,anchors,ratio,1.,info.duration)
   c={'vid':r['vid'],'status':status,'segments':[s.__dict__ for s in segs],'selected_frames':segments_to_frames(segs,anchors,timeline),'raw_reply':raw,'seconds':time.time()-t0}
   cf.write_text(json.dumps(c)+'\n')
  q=clips_from_frames(c['selected_frames'],timeline,n);k=int(q.sum())
  uni=np.zeros(n,bool)
  if k:uni[np.round(np.linspace(0,n-1,k)).astype(int)]=True
  row={'vid':r['vid'],'status':c['status'],'clips':n,'qwen_clips':k,'frame_ratio':len(c['selected_frames'])/len(timeline)}
  for g,gt in (('hi',(labels>=.75)&mask),('rel',mask)):
   row[f'qwen_{g}']=f_set(q,gt);row[f'all_{g}']=f_set(np.ones(n,bool),gt);row[f'uniform_{g}']=f_set(uni,gt)
   row[f'lift_{g}']=float(gt[q].mean()/gt.mean()) if k and gt.any() else None
  rows.append(row);print(json.dumps(row),flush=True)
 summ={'videos':len(rows),'empty_videos':sum(r['qwen_clips']==0 for r in rows),'status':{s:sum(r['status']==s for r in rows) for s in {r['status'] for r in rows}},
  'selected_clip_ratio':float(np.mean([r['qwen_clips']/r['clips'] for r in rows])),'elapsed_seconds':time.time()-start}
 for g in ('hi','rel'):
  for m in ('qwen','all','uniform'):summ[f'{m}_{g}']=float(np.mean([r[f'{m}_{g}'] for r in rows]))
  summ[f'qwen_vs_all_{g}']=boot([r[f'qwen_{g}'] for r in rows],[r[f'all_{g}'] for r in rows])
  summ[f'qwen_vs_uniform_{g}']=boot([r[f'qwen_{g}'] for r in rows],[r[f'uniform_{g}'] for r in rows])
  lifts=[r[f'lift_{g}'] for r in rows if r[f'lift_{g}'] is not None];summ[f'lift_{g}_mean']=float(np.mean(lifts)) if lifts else None
 # admission (guide P2): Qwen may enter distillation only if it beats all-select AND the volume-matched control on dev
 summ['admit_distillation']=bool(summ['qwen_vs_all_hi']['ci95'][0]>0 and summ['qwen_vs_uniform_hi']['ci95'][0]>0)
 (a.output/'metrics.json').write_text(json.dumps({'protocol':'QVH native dev, Qwen teacher selection -> 2 s clips, set-F1 proxy','test_content_used':False,'target_ratio_prompt':ratio,'summary':summ,'per_video':rows,'official_f_video':None},indent=2)+'\n')
 print(json.dumps(summ),flush=True)
if __name__=='__main__':main()
