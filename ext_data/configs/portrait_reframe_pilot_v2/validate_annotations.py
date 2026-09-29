"""Validate portrait_reframe_pilot_v2 annotation JSONL files; QC report only.

Never merges annotator rows (merge rules are frozen separately after
annotation completes, before any scoring).  Usage:
  python validate_annotations.py --annotations ann.jsonl [--annotations more.jsonl ...]
      [--manifest sampling_manifest_v1.json] --output qc_report.json
Exit code 0 iff no structural error; QC stats ride along in the report.
"""
import argparse,json,math,sys
from collections import defaultdict
from pathlib import Path


def err(row,msg,errors):
 errors.append({'pm_video_id':row.get('pm_video_id'),'frame_idx':row.get('frame_idx'),
  'annotator_id':row.get('annotator_id'),'msg':msg,'line':row.get('_line')})


def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--annotations',nargs='+',type=Path,required=True)
 ap.add_argument('--manifest',type=Path,default=Path(__file__).parent/'sampling_manifest_v2.json')
 ap.add_argument('--output',type=Path,default=Path('qc_report.json'))
 a=ap.parse_args()
 man=json.loads(a.manifest.read_text())
 known={s['pm_video_id'] for s in man['sources']}
 temporal_subsets={s['pm_video_id'] for s in man['sources'] if s.get('temporal_subset')}
 errors=[];frame_rows=[];temporal_rows=[]
 for path in a.annotations:
  for ln,line in enumerate(path.read_text().splitlines(),1):
   if not line.strip():continue
   try:row=json.loads(line)
   except json.JSONDecodeError as e:errors.append({'line':ln,'msg':f'bad json: {e}'});continue
   row['_line']=ln
   if row.get('schema')!='portrait_reframe_pilot_v2':err(row,'schema field must be portrait_reframe_pilot_v2',errors);continue
   if row.get('pm_video_id') not in known:err(row,'pm_video_id not in frozen manifest',errors);continue
   if row.get('kind')=='temporal':
    if row.get('pm_video_id') not in temporal_subsets:err(row,'temporal row for non-subset source',errors)
    if row.get('label') not in ('keep','trim','unsure'):err(row,'label must be keep|trim|unsure',errors)
    s,e2=row.get('span_start_sec'),row.get('span_end_sec')
    if not all(isinstance(v,(int,float)) and math.isfinite(v) and 0<=v for v in (s,e2)) or not e2>s:
     err(row,'span must be finite numbers with end>start>=0',errors)
    temporal_rows.append(row);continue
   for f in ('frame_idx','t_sec','source_w','source_h','window_w','window_h_cont','window_y','window_bottom_cont','annotator_id'):
    if f not in row:err(row,f'missing field {f}',errors)
   if errors and errors[-1].get('line')==ln:continue
   W,H=row['source_w'],row['source_h'];h=float(row['window_h_cont'])
   if row['window_w']!=W:err(row,'window_w must equal source_w (no scale/rotate)',errors)
   if abs(h-W*9/16)>1e-6:err(row,f'window_h_cont must equal W*9/16 exactly ({W*9/16})',errors)
   y=row['window_y']
   if not isinstance(y,int) or not 0<=y<=H-h:err(row,f'window_y must be integer in [0,{H-h}]',errors)
   if abs(row['window_bottom_cont']-(y+h))>1e-6:err(row,'window_bottom_cont != window_y + h_cont',errors)
   if row.get('confidence') not in ('sure','unsure'):err(row,'confidence must be sure|unsure',errors)
   if not isinstance(row.get('skip_transition'),bool):err(row,'skip_transition must be bool',errors)
   frame_rows.append(row)
 # per-source stats
 per=defaultdict(lambda:{'annotators':set(),'rows':0,'skip':0,'unsure':0})
 for r in frame_rows:
  p=per[r['pm_video_id']];p['annotators'].add(r['annotator_id']);p['rows']+=1
  p['skip']+=bool(r.get('skip_transition'));p['unsure']+=r.get('confidence')=='unsure'
 agree=[];flagged=[]
 bysrc=defaultdict(lambda:defaultdict(list))
 for r in frame_rows:bysrc[r['pm_video_id']][(r['frame_idx'],r['annotator_round'] if 'annotator_round' in r else 1)].append(r)
 for vid,frames in bysrc.items():
  for (fi,rd),rows in frames.items():
   if len(rows)>=2:
    ys=[r['window_y'] for r in rows];h=rows[0]['window_h_cont']
    d=max(ys)-min(ys);agree.append(d/max(h,1e-9))
    if d>0.1*h:flagged.append({'pm_video_id':vid,'frame_idx':fi,'spread_px':d,'threshold_px':0.1*h,'annotators':[r['annotator_id'] for r in rows]})
 conflicts=0
 byt=defaultdict(list)
 for r in temporal_rows:byt[(r['pm_video_id'],r['annotator_id'])].append(r)
 for k,rows in byt.items():
  for i,x in enumerate(rows):
   for y in rows[i+1:]:
    if x['label']!=y['label'] and x.get('span_start_sec',0)<y.get('span_end_sec',0) and y.get('span_start_sec',0)<x.get('span_end_sec',0):conflicts+=1
 report={'schema':'portrait_reframe_pilot_v2','files':[str(p) for p in a.annotations],
  'structural_errors':errors,'error_count':len(errors),
  'frame_rows':len(frame_rows),'temporal_rows':len(temporal_rows),
  'sources_touched':len(per),
  'per_source':{k:{'annotators':sorted(v['annotators']),'rows':v['rows'],'skip_transition':v['skip'],'unsure':v['unsure']} for k,v in sorted(per.items())},
  'dual_annotation':{'frame_pairs_compared':len(agree),
   'spread_over_0.1h_count':len(flagged),
   'spread_median':sorted(agree)[len(agree)//2] if agree else None,
   'flagged':flagged[:200]},
  'temporal_conflicts':conflicts,
  'note':'rows are never merged here; merge rules frozen separately before scoring'}
 a.output.write_text(json.dumps(report,indent=1,ensure_ascii=False)+'\n')
 print(json.dumps({'errors':len(errors),'frame_rows':len(frame_rows),'temporal_rows':len(temporal_rows),
  'sources_touched':len(per),'dual_pairs':len(agree),'flagged_over_0.1h':len(flagged)}))
 sys.exit(0 if not errors else 1)
if __name__=='__main__':main()
