"""T3 public proxy: does 32B removal-only judgement remove non-highlight frames?

YouTube Highlights val (<=120 s).  Labels are edited-video matching, not human
highlight/negative judgements: 1 = clip reused by the uploader's edit, -1 = not
reused (NOT a human negative), 0 = borderline.  Frame label: 1 if any covering
clip is 1; -1 if every covering clip is -1; otherwise ignored; uncovered frames
ignored.  Frame index = annotation frame (media frame_count mismatch videos are
reported, not dropped).

Proxy score per video with perfect boxes (IoU 1 on positive frames, 0 elsewhere):
  F = 2 |K n P| / (|K| + |P|)   over labelled frames (K kept, P positive)
Compared with keep-all and with a same-count random removal of cells (seeded,
20 draws) - the control for "fewer outputs" versus "better choice".
"""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.qwen_temporal_removal import keep_frames


def frame_labels(seg_path,n):
 s=json.loads(Path(seg_path).read_text())['segments_frames'];lab=np.full(n,np.nan);anyp=np.zeros(n,bool);cov=np.zeros(n,bool);alln=np.ones(n,bool)
 for x in s:
  a,b=int(x['start_frame']),min(n,int(x['end_frame']))
  if a>=n:continue
  cov[a:b]=True;l=x['match_label']
  if l==1:anyp[a:b]=True
  if l!=-1:alln[a:b]=False
 lab[cov&anyp]=1;lab[cov&~anyp&alln]=-1
 return lab,int(max(x['end_frame'] for x in s))


def f_proxy(keep,lab):
 m=~np.isnan(lab);k=keep[m];p=lab[m]==1
 return 2*(k&p).sum()/max(1,k.sum()+p.sum())


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--records',required=True);ap.add_argument('--removal',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 rng=np.random.default_rng(20260928);rows=[]
 for r in map(json.loads,open(a.records)):
  f=a.removal/f"{r['video_id']}.json"
  if not f.exists():continue
  d=json.loads(f.read_text());n=d['n_frames'];meta=json.loads((a.removal.parent/'anchors'/r['video_id']/'meta.json').read_text());fc=np.array(meta['frame_cell'])
  lab,last=frame_labels(r['segments'],n);keep=np.zeros(n,bool);keep[keep_frames(fc,d['removed_cells'])[0]]=True
  allk=np.ones(n,bool);cells=np.unique(fc);k=len(d['removed_cells']) if not d['guard_tripped'] else 0;rand=[]
  for _ in range(20):
   rc=rng.choice(cells,k,replace=False) if k else [];kk=np.zeros(n,bool);kk[keep_frames(fc,rc)[0]]=True;rand.append(f_proxy(kk,lab))
  m=~np.isnan(lab);rem=~keep&m
  rows.append({'video_id':r['video_id'],'domain':r['video_id'].split('__')[0],'frames':n,'labelled':int(m.sum()),'pos_rate':float((lab[m]==1).mean()) if m.any() else None,
   'keep_rate':float(keep.mean()),'guard':d['guard_tripped'],'removed_frames_labelled':int(rem.sum()),'removed_pos':int((rem&(lab==1)).sum()),'removed_neg':int((rem&(lab==-1)).sum()),
   'F_keep_all':float(f_proxy(allk,lab)),'F_removal':float(f_proxy(keep,lab)),'F_random_same_count':float(np.mean(rand)),'frame_mismatch':bool(last>n*1.02)})
 R=[x for x in rows if x['labelled']>0 and x['pos_rate'] is not None];dv=lambda k1,k2:np.array([x[k1]-x[k2] for x in R])
 def boot(v):
  b=np.random.default_rng(20260928).choice(v,(5000,len(v))).mean(1);return [float(np.percentile(b,2.5)),float(np.percentile(b,97.5))]
 rem=sum(x['removed_frames_labelled'] for x in R);neg_all=sum(x['labelled']*(1-x['pos_rate']) for x in R)/max(1,sum(x['labelled'] for x in R))
 out={'videos':len(R),'protocol':__doc__.strip().splitlines()[0],'keep_rate_mean':float(np.mean([x['keep_rate'] for x in R])),'videos_with_removal':sum(1 for x in R if x['keep_rate']<1),
  'guard_tripped':sum(x['guard'] for x in R),'F_keep_all':float(np.mean([x['F_keep_all'] for x in R])),'F_removal':float(np.mean([x['F_removal'] for x in R])),'F_random':float(np.mean([x['F_random_same_count'] for x in R])),
  'removal_minus_keep_all':{'mean':float(dv('F_removal','F_keep_all').mean()),'ci95':boot(dv('F_removal','F_keep_all')),'better':int((dv('F_removal','F_keep_all')>1e-6).sum()),'worse':int((dv('F_removal','F_keep_all')<-1e-6).sum())},
  'removal_minus_random':{'mean':float(dv('F_removal','F_random_same_count').mean()),'ci95':boot(dv('F_removal','F_random_same_count'))},
  'removed_labelled_frames':rem,'removed_neg_share':sum(x['removed_neg'] for x in R)/max(1,rem),'removed_pos_share':sum(x['removed_pos'] for x in R)/max(1,rem),'base_neg_share':float(neg_all),
  'by_domain':{dm:{'videos':sum(1 for x in R if x['domain']==dm),'delta':float(np.mean([x['F_removal']-x['F_keep_all'] for x in R if x['domain']==dm])),'keep':float(np.mean([x['keep_rate'] for x in R if x['domain']==dm]))} for dm in sorted({x['domain'] for x in R})},
  'label_semantics':'weak: -1 = not reused by the uploader edit, not a human negative','official_f_video':None,'per_video':rows}
 a.output.mkdir(parents=True,exist_ok=True);(a.output/'metrics.json').write_text(json.dumps(out,indent=1)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='per_video'},indent=1))
if __name__=='__main__':main()
