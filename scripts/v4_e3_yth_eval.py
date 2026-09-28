"""V4 E3 public check on YouTube Highlights: segment roles vs TEMP removal.

Prereg configs/V4_E3_EVENT_PREREG.json.  Labels (per frame of the annotation
timeline; uncovered frames are ignored, as in scripts/t3_yth_eval.py):
  human  mturk soft votes (AMT workers, relative within a video): positive if
         the max vote of the covering clips > 0, negative if every covering
         clip has 0 votes.  Human, but relative (no absolute negative).
  weak   match_label (uploader-edit reuse; -1 = not reused, NOT a human
         negative) - reported only, scripts.t3_yth_eval.frame_labels.
Proxy per video with perfect boxes: F = 2|K n P| / (|K| + |P|) over labelled
frames.  Masks: KEEP_ALL, TEMP (removal replies), EVENT (roles on top of
TEMP), RANDOM (same number of removed cells as EVENT, 20 seeded draws).
Also: human vote share of restored / newly dropped frames vs the video base.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.qwen_temporal_removal import keep_frames
from scripts.t3_yth_eval import frame_labels,f_proxy
from scripts.qwen_event_segments import drop_only_mask


def human_labels(seg_path,n):
 s=json.loads(Path(seg_path).read_text())['segments_frames'];cov=np.zeros(n,bool);mx=np.zeros(n)
 for x in s:
  a,b=int(x['start_frame']),min(n,int(x['end_frame']))
  if a>=n or x.get('mturk_votes') is None:continue
  cov[a:b]=True;mx[a:b]=np.maximum(mx[a:b],float(x['mturk_votes']))
 lab=np.full(n,np.nan);lab[cov]=np.where(mx[cov]>0,1.,-1.);return lab


def boot(v,seed=20260928):
 v=np.asarray(v,float);b=np.random.default_rng(seed).choice(v,(5000,len(v))).mean(1);return [float(np.percentile(b,2.5)),float(np.percentile(b,97.5))]


def mask(fc,cells):
 k=np.zeros(len(fc),bool);k[keep_frames(fc,cells)[0]]=True;return k


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--records',required=True);ap.add_argument('--anchors',type=Path,required=True);ap.add_argument('--temp',type=Path,required=True)
 ap.add_argument('--event',type=Path,required=True);ap.add_argument('--mode',choices=('event','drop_only'),default='event');ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 rng=np.random.default_rng(20260928);rows=[]
 for r in map(json.loads,open(a.records)):
  vid=r['video_id'];fe=a.event/f'{vid}.json'
  if not fe.exists():continue
  ev=json.loads(fe.read_text());tp=json.loads((a.temp/f'{vid}.json').read_text());meta=json.loads((a.anchors/vid/'meta.json').read_text());fc=np.array(meta['frame_cell']);n=len(fc)
  if ev['temp_removed_cells']!=tp['removed_cells']:raise ValueError('TEMP mismatch '+vid)
  hl=human_labels(r['segments'],n);wl,_=frame_labels(r['segments'],n)
  if a.mode=='drop_only':
   rc,_=drop_only_mask(fc,tp['removed_cells'],{int(s):r for s,r in ev['roles'].items()},{int(s):c for s,c in ev['segment_cells'].items()})
   eff=set(ev['temp_effective_removed_cells']);ev={**ev,'removed_cells':rc,'restored_cells':sorted(eff-set(rc)),'added_removed_cells':sorted(set(rc)-eff)}
  kt=mask(fc,tp['removed_cells']);ke=mask(fc,ev['removed_cells']);ka=np.ones(n,bool);cells=np.unique(fc);k=len(ev['removed_cells'])
  row={'video_id':vid,'domain':vid.split('__')[0],'frames':n,'keep_temp':float(kt.mean()),'keep_event':float(ke.mean()),'guard':ev['guard_tripped'],
   'restored_cells':len(ev['restored_cells']),'added_cells':len(ev['added_removed_cells']),'roles':{x:list(ev['roles'].values()).count(x) for x in ('MAIN','SETUP','RESULT','IDLE','JUNK','UNSURE')},
   'status':[b['status'] for b in ev['blocks']]}
  for tag,lab in (('human',hl),('weak',wl)):
   m=~np.isnan(lab)
   if not m.any():row[tag]=None;continue
   rand=[]
   for _ in range(20):
    rc=rng.choice(cells,min(k,len(cells)),replace=False) if k else [];rand.append(f_proxy(mask(fc,list(rc)),lab))
   restored=ke&~kt&m;dropped=~ke&kt&m
   row[tag]={'labelled':int(m.sum()),'pos_rate':float((lab[m]==1).mean()),'F_keep_all':float(f_proxy(ka,lab)),'F_temp':float(f_proxy(kt,lab)),'F_event':float(f_proxy(ke,lab)),'F_random':float(np.mean(rand)),
    'restored_labelled':int(restored.sum()),'restored_pos':int((restored&(lab==1)).sum()),'dropped_labelled':int(dropped.sum()),'dropped_pos':int((dropped&(lab==1)).sum())}
  rows.append(row)
 out={'prereg':'configs/V4_E3_EVENT_PREREG.json' if a.mode=='event' else 'configs/V4_E3B_DROP_ONLY_PREREG.json','mode':a.mode,'videos':len(rows),'official_f_video':None}
 for tag in ('human','weak'):
  R=[x for x in rows if x[tag]];d=lambda p,q:np.array([x[tag][p]-x[tag][q] for x in R])
  s={'videos':len(R),**{f'F_{k}':float(np.mean([x[tag][f'F_{k}'] for x in R])) for k in ('keep_all','temp','event','random')}}
  for p,q in (('F_event','F_temp'),('F_event','F_keep_all'),('F_temp','F_keep_all'),('F_event','F_random')):
   v=d(p,q);s[f'{p}-{q}']={'mean':float(v.mean()),'ci95':boot(v),'better':int((v>1e-6).sum()),'worse':int((v<-1e-6).sum())}
  rl=sum(x[tag]['restored_labelled'] for x in R);dl=sum(x[tag]['dropped_labelled'] for x in R);lab=sum(x[tag]['labelled'] for x in R)
  s['restored_pos_share']=sum(x[tag]['restored_pos'] for x in R)/max(1,rl);s['restored_labelled']=rl
  s['dropped_pos_share']=sum(x[tag]['dropped_pos'] for x in R)/max(1,dl);s['dropped_labelled']=dl
  s['base_pos_share']=sum(x[tag]['pos_rate']*x[tag]['labelled'] for x in R)/max(1,lab)
  s['by_domain']={dm:{'videos':sum(1 for x in R if x['domain']==dm),'delta_event_temp':float(np.mean([x[tag]['F_event']-x[tag]['F_temp'] for x in R if x['domain']==dm]))} for dm in sorted({x['domain'] for x in R})}
  out[tag]=s
 out['keep_temp']=float(np.mean([x['keep_temp'] for x in rows]));out['keep_event']=float(np.mean([x['keep_event'] for x in rows]))
 out['videos_changed']=sum(1 for x in rows if x['restored_cells'] or x['added_cells']);out['guard']=sum(x['guard'] for x in rows)
 out['roles']={k:sum(x['roles'][k] for x in rows) for k in ('MAIN','SETUP','RESULT','IDLE','JUNK','UNSURE')}
 st=[s for x in rows for s in x['status']];out['block_status']={k:st.count(k) for k in sorted(set(st))}
 out['per_video']=rows
 a.output.mkdir(parents=True,exist_ok=True);(a.output/'metrics.json').write_text(json.dumps(out,indent=1)+'\n')
 print(json.dumps({k:v for k,v in out.items() if k!='per_video'},indent=1))
if __name__=='__main__':main()
