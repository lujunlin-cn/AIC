"""Sample the PM-400 / AVE-PM portrait-reframe annotation pilot (guide §data).

Real portrait sources -> 16:9 windows.  The PM source-video-ID is the source
group (clips cut from one source never straddle pilot splits).  AVE-PM event
onset/offset, BGM and the 86-class label travel only as context fields; they
are never crop GT and never a confirmation set.  The Douyin direct links are
dead (tested 2026-09-29: 12 links x 3 header sets, all 502), so media comes
from the community AVE-PM cache (Google Drive 68.07 GB, background download);
PM400-to-AVEPM.csv covers exactly the clips inside that cache.

Output: pilot/sampling_manifest_v1.json  (320 sources / 240 clip references,
pilot_split by sha256("portrait_pilot_v1:"+video_id), status fields honest
about pending media and pending annotation).
"""
import csv,hashlib,json,random
from collections import defaultdict
from pathlib import Path

SRC=Path('/data/aic/external_datasets/PM400/raw/links')
OUT=Path('/data/aic/external_datasets/PM400/pilot')
N_SRC=320;N_CLIPS=240;SEED=20260929
SPLIT_BUCKETS=(2/3,5/6)


def split_of(vid):
 h=int(hashlib.sha256(('portrait_pilot_v1:'+vid).encode()).hexdigest(),16)%10**6/10**6
 return 'train' if h<SPLIT_BUCKETS[0] else ('dev' if h<SPLIT_BUCKETS[1] else 'confirm')


def main():
 rows=list(csv.DictReader(open(SRC/'PM400-to-AVEPM.csv')))
 names={}
 for l in open(SRC/'class_name_mapping.csv'):
  if ',' in l:names[int(l.rsplit(',',1)[1])]=l.rsplit(',',1)[0].strip()
 by_src=defaultdict(list)
 for r in rows:by_src[r['video_id']].append(r)
 cands=[(v,cl) for v,cl in by_src.items() if max(float(c['duration']) for c in cl)>=11.0]
 rng=random.Random(SEED);rng.shuffle(cands)
 by_cat=defaultdict(list)
 for v,cl in cands:by_cat[int(cl[0]['category'])].append((v,cl))
 for lst in by_cat.values():rng.shuffle(lst)
 cats=sorted(by_cat,key=lambda c:-len(by_cat[c]))  # round-robin, biggest classes first
 picks=[]
 while len(picks)<N_SRC:
  before=len(picks)
  for c in cats:
   if by_cat[c] and len(picks)<N_SRC:picks.append(by_cat[c].pop())
  if len(picks)==before:break
 out=[]
 for vid,cl in picks:
  cl=sorted(cl,key=lambda c:int(c['sample_id']));take=cl[:1]
  out.append({'pm_video_id':vid,'pilot_split':split_of(vid),'category':int(cl[0]['category']),
   'category_name':names.get(int(cl[0]['category']),'?'),'source_duration_s':float(cl[0]['duration']),
   'clip_context':None if len(out)>=N_CLIPS else {'sample_id':int(take[0]['sample_id']),'start_s':float(take[0]['start']),
    'end_s':float(take[0]['end']),'event_start_s':float(take[0]['event_start']),'event_end_s':float(take[0]['event_end']),
    'have_bgm':int(take[0]['haveBGM']),'event_label':int(take[0]['label']),
    'context_only_not_crop_gt':True},
   'media_status':'pending_download','annotation_status':'pending'})
 OUT.mkdir(parents=True,exist_ok=True)
 man={'name':'portrait_reframe_pilot_v1','created':'2026-09-29','seed':SEED,
  'license':'CC BY-NC-SA 4.0, non-commercial research only (bytedance/Portrait-Mode-Video DATA.md); usage registered here',
  'source_groups':'pm_video_id; clips from one source never straddle pilot splits',
  'split_rule':'sha256("portrait_pilot_v1:"+pm_video_id)/1e6 < 2/3 train, < 5/6 dev, else confirm',
  'media':{'direct_links_tested':'12 links x 3 header sets on 2026-09-29, all HTTP 502 (dead)',
   'active_channel':'community AVE-PM cache, Google Drive id 1zXOeVE9xaMprd1O_Xec65Wn7B5VWelLT, 68067427328 B, background download started 2026-09-29'},
  'gt_policy':'no crop GT exists in PM-400/AVE-PM; human 16:9 window annotation per ANNOTATION_SCHEMA is the only future GT; event/BGM/class fields are context only',
  'counts':{'sources':len(out),'with_clip_context':sum(1 for x in out if x['clip_context']),
   'splits':{s:sum(1 for x in out if x['pilot_split']==s) for s in ('train','dev','confirm')},
   'categories_covered':len({x['category'] for x in out})},
  'sources':out}
 (OUT/'sampling_manifest_v1.json').write_text(json.dumps(man,indent=1)+'\n')
 print(json.dumps(man['counts']))


if __name__=='__main__':main()
