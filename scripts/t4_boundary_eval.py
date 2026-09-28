"""T4 public proxy: does boundary refinement beat the coarse TEMP mask on YTH val?

Same weak labels and F proxy as scripts/t3_yth_eval.py (1 = reused by the
uploader's edit, -1 = not reused, not a human negative).  Compared: coarse
TEMP-style mask (T3 replies) vs refined mask (T4 zone replies).  Reported:
paired F delta (video bootstrap), restored/newly removed frames, and on
labelled flipped frames the share whose label agrees with the flip
(restored -> label 1, newly removed -> label -1) against the base rate.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.qwen_temporal_removal import keep_frames
from scripts.qwen_temporal_boundary import refine_mask
from scripts.t3_yth_eval import frame_labels,f_proxy


def boot(v):
 b=np.random.default_rng(20260928).choice(v,(5000,len(v))).mean(1);return [float(np.percentile(b,2.5)),float(np.percentile(b,97.5))]


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--records',required=True);ap.add_argument('--anchors',type=Path,required=True);ap.add_argument('--removal',type=Path,required=True)
 ap.add_argument('--frames-dir',type=Path,required=True);ap.add_argument('--zones',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 rows=[];status={};moves=[]
 for r in map(json.loads,open(a.records)):
  vid=r['video_id'];rf=a.removal/f'{vid}.json'
  if not rf.exists():continue
  d=json.loads(rf.read_text());meta=json.loads((a.anchors/vid/'meta.json').read_text());fc=np.array(meta['frame_cell']);n=len(fc)
  zf=a.zones/f'{vid}.json';zinfo=json.loads((a.frames_dir/vid/'zones.json').read_text())
  zs=json.loads(zf.read_text())['zones'] if zf.exists() else []
  for z in zs:status[z['status']]=status.get(z['status'],0)+1;moves+=[z['moved_subcells']] if z.get('moved_subcells') is not None else []
  kept,_=keep_frames(fc,d['removed_cells']);coarse=np.zeros(n,bool);coarse[kept]=True
  refined,info=refine_mask(zinfo['times'],fc,d['removed_cells'],zs) if zs else (coarse.copy(),{'refine_guard':False,'restored':0,'newly_removed':0})
  lab,_=frame_labels(r['segments'],n);m=~np.isnan(lab);rest=refined&~coarse&m;newr=coarse&~refined&m
  rows.append({'video_id':vid,'frames':n,'zones':len(zs),'labelled':int(m.sum()),'pos_rate':float((lab[m]==1).mean()) if m.any() else None,
   'F_coarse':float(f_proxy(coarse,lab)),'F_refined':float(f_proxy(refined,lab)),'F_keep_all':float(f_proxy(np.ones(n,bool),lab)),**info,
   'restored_labelled':int(rest.sum()),'restored_pos':int((rest&(lab==1)).sum()),'newly_removed_labelled':int(newr.sum()),'newly_removed_neg':int((newr&(lab==-1)).sum()),
   'coarse_keep':float(coarse.mean()),'refined_keep':float(refined.mean())})
 R=[x for x in rows if x['labelled']>0 and x['pos_rate'] is not None];Z=[x for x in R if x['zones']]
 dv=np.array([x['F_refined']-x['F_coarse'] for x in R]);dz=np.array([x['F_refined']-x['F_coarse'] for x in Z])
 lab_all=sum(x['labelled'] for x in R);pos_base=sum(x['labelled']*x['pos_rate'] for x in R)/max(1,lab_all)
 rl=sum(x['restored_labelled'] for x in R);nl=sum(x['newly_removed_labelled'] for x in R)
 out={'videos':len(R),'videos_with_zones':len(Z),'zone_status':status,'moved_subcells_hist':{str(k):moves.count(k) for k in sorted(set(moves))},
  'F_keep_all':float(np.mean([x['F_keep_all'] for x in R])),'F_coarse':float(np.mean([x['F_coarse'] for x in R])),'F_refined':float(np.mean([x['F_refined'] for x in R])),
  'refined_minus_coarse_all':{'mean':float(dv.mean()),'ci95':boot(dv),'better':int((dv>1e-6).sum()),'worse':int((dv<-1e-6).sum())},
  'refined_minus_coarse_zone_videos':{'mean':float(dz.mean()),'ci95':boot(dz)} if len(Z) else None,
  'frames_restored':int(sum(x['restored'] for x in R)),'frames_newly_removed':int(sum(x['newly_removed'] for x in R)),'refine_guard':sum(x['refine_guard'] for x in R),
  'restored_labelled':rl,'restored_pos_share':sum(x['restored_pos'] for x in R)/max(1,rl),'newly_removed_labelled':nl,'newly_removed_neg_share':sum(x['newly_removed_neg'] for x in R)/max(1,nl),
  'base_pos_share':float(pos_base),'base_neg_share':float(1-pos_base),
  'label_semantics':'weak: -1 = not reused by the uploader edit, not a human negative; 100-frame segments at 47-frame stride cannot resolve 0.25 s boundaries','official_f_video':None,'per_video':rows}
 a.output.mkdir(parents=True,exist_ok=True);(a.output/'metrics.json').write_text(json.dumps(out,indent=1)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='per_video'},indent=1))
if __name__=='__main__':main()
