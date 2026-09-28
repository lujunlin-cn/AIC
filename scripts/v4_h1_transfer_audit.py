"""V4 H1 audit: where does the T5 point calibration help on public crops, and
how far is the official set from those strata?

H1 (frozen T5 probe, a/b per free axis): c = clip(a*p + (1-a)*.5 + b*win, 0, 1),
identical in scripts/t5_crophead_probe.h1_centres (fit/eval) and
scripts/crophead_release.calibrate (official).  This script only reads the T5
tables (RetargetVid 6-annotator crops, LIVE-YT-VC sparse single-annotator boxes)
and the official obs cache / 1 s Qwen points; nothing is fitted.

Strata
  video level (full N0 pipeline, delta = H1 - H0 per video, macro average):
    set (rv 1:3 x-axis win .188 / rv 3:1 y-axis win .593 / live x-axis win .316),
    relative window bin, video face-rate bin
  keyframe level (window centred on the calibrated vs raw point, IoU at that
  keyframe; the held point also reaches no-face frames only):
    point position (edge = raw point outside the movable centre range, i.e. the
    N0 window is already clamped; middle otherwise), persons (>=.5 COCO person
    boxes), face at the keyframe
Split labels: *_train = H1 fit data; rv_dev/live_dev = T5 selection; rv_confirm2 /
live_confirm3 = T5 confirmation.  All four are now REUSED validation.
"""
import argparse,csv,json,math,pickle
from pathlib import Path
import numpy as np
from aic.max_window_path import geometry,centre_to_offset
from scripts.benchmark_spatial import iou
from scripts.teacher_diag_eval import win_boxes
from scripts.t5_crophead_probe import pipe,h1_centres,boot,FEATS

F=dict(zip(FEATS,range(len(FEATS))))
SPLITS=('rv_train','live_train','rv_dev','live_dev','rv_confirm2','live_confirm3')


def set_name(u):
 return f"{u['ds']}_{'x' if u['comp']==0 else 'y'}"


def win_bin(w):return 'win<.25' if w<.25 else ('win.25-.45' if w<.45 else 'win>=.45')


def face_bin(r):return 'face=0' if r==0 else ('face<.5' if r<.5 else 'face>=.5')


def pos_bin(pa,win):
 lo,hi=win/2,1-win/2
 if pa<lo or pa>hi:return 'edge'
 return 'centre' if abs(pa-.5)<.1 else 'middle'


def person_bin(n):return 'persons=0' if n==0 else ('persons=1' if n==1 else 'persons>=2')


def summary(d):
 d=np.asarray(d,float)
 if not len(d):return {}
 ci=boot(d) if len(d)>1 else [float(d[0])]*2
 return {'mean':float(d.mean()),'ci_lo':ci[0],'ci_hi':ci[1],'better':int((d>1e-4).sum()),'worse':int((d<-1e-4).sum())}


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tables',type=Path,default=Path('/data/aic/experiments/T5_CROPHEAD_V3/tables'))
 ap.add_argument('--probe',type=Path,default=Path('/data/aic/experiments/T5_CROPHEAD_V3/probe_seed0/metrics.json'))
 ap.add_argument('--index',type=Path,default=Path('/data/aic/official_test_20260926/intake/index.enriched.jsonl'))
 ap.add_argument('--official-cache',type=Path,default=Path('/data/aic/experiments/OBS_CACHE_OFFICIAL_V1/cache'))
 ap.add_argument('--official-points',type=Path,default=Path('/data/aic/experiments/QWEN_SUBJECT_POINT_OFFICIAL_V1/points'))
 ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 ab={int(k):tuple(v) for k,v in json.loads(a.probe.read_text())['H1'].items()}
 rows=[];fit_counts={}
 for sp in SPLITS:
  us=pickle.loads((a.tables/f'{sp}.pkl').read_bytes())
  fit_counts[sp]={'units':len(us),'videos':len({(u['ds'],u['vid']) for u in us}),'keyframes_with_gt':int(sum((~np.isnan(u['Y'][:,0])).sum() for u in us))}
  vid_rows=[];kf_rows=[]
  for u in us:
   win=u['s']/u['L'];fr=float(u['face'].mean())
   h1=h1_centres(u,ab) if len(u['J']) else np.zeros(0)
   d=pipe(u,h1)-pipe(u) if len(u['J']) else 0.
   vid_rows.append({'vid':(u['ds'],u['vid']),'set':set_name(u),'win':win_bin(win),'face':face_bin(fr),'d':d})
   m=~np.isnan(u['Y'][:,0])
   if not m.any():continue
   gpos={int(f):i for i,f in enumerate(u['gf'])};J=u['J'][m];pa=u['pa'][m];c1=h1[m]
   g=u['gt'][:,[gpos[u['keys'][j]] for j in J]]
   o0=centre_to_offset(pa,u['W'],u['H'],u['ratio']);o1=centre_to_offset(c1,u['W'],u['H'],u['ratio'])
   i0=iou(win_boxes(o0,u['W'],u['H'],u['ratio'])[None],g).mean(0);i1=iou(win_boxes(o1,u['W'],u['H'],u['ratio'])[None],g).mean(0)
   X=u['X'][m][:,0]
   for k in range(len(J)):
    kf_rows.append({'vid':(u['ds'],u['vid']),'set':set_name(u),'pos':pos_bin(pa[k],win),'persons':person_bin(int(round(math.expm1(X[k,F['n_person']])))),
     'kf_face':'kf_face' if X[k,F['face']]>0 else 'kf_noface','d':float(i1[k]-i0[k]),'shift':float(abs(c1[k]-pa[k])/win)})
  # video-level strata: per video macro (average ratios of the same RV video inside a stratum)
  for dim in ('set','win','face'):
   for val in sorted({r[dim] for r in vid_rows}):
    per={}
    for r in vid_rows:
     if r[dim]==val:per.setdefault(r['vid'],[]).append(r['d'])
    dv=[np.mean(v) for v in per.values()]
    rows.append({'split':sp,'level':'video_pipeline','dimension':dim,'stratum':val,'videos':len(dv),'keyframes':'',**summary(dv),'mean_shift_win':''})
  for s_ in sorted({r['set'] for r in kf_rows}):
   for dim in ('pos','persons','kf_face'):
    for val in sorted({r[dim] for r in kf_rows if r['set']==s_}):
     sel=[r for r in kf_rows if r['set']==s_ and r[dim]==val];per={}
     for r in sel:per.setdefault(r['vid'],[]).append(r['d'])
     dv=[np.mean(v) for v in per.values()]
     rows.append({'split':sp,'level':'keyframe_video_macro','dimension':f'{s_}:{dim}','stratum':val,'videos':len(dv),'keyframes':len(sel),**summary(dv),'mean_shift_win':float(np.mean([r['shift'] for r in sel]))})
 # official distribution over the same strata (no labels; 1 s points = the H1 release input)
 off=[];offkf=[]
 for r in map(json.loads,open(a.index)):
  vid=r['video_id'];z=np.load(a.official_cache/f'{vid}_t.npz');W,H=int(z['W']),int(z['H']);w,h,axis=geometry(W,H,r['targetRatioWH'])
  if axis is None:continue
  comp=0 if axis==0 else 1;win=(w/W) if comp==0 else (h/H);q=json.loads((a.official_points/f'{vid}.json').read_text());face=z['chosen']>=0
  off.append({'axis':comp,'win':win_bin(win),'face':face_bin(float(face.mean())),'src':f"{W}x{H}"})
  for t,p in zip(q['keyframes'],q['ratios']['t']['points']):
   if p is None or p[2]:continue
   aa,bb=ab[comp];c1=float(np.clip(aa*p[comp]+(1-aa)*.5+bb*win,0,1));d=z['det'][z['det_off'][t]:z['det_off'][t+1]];d=d[d[:,4]>=.5]
   offkf.append({'axis':comp,'pos':pos_bin(p[comp],win),'persons':person_bin(int((d[:,5]==1).sum()) if len(d) else 0),'kf_face':'kf_face' if face[t] else 'kf_noface','shift':abs(c1-p[comp])/win})
 for comp in (0,1):
  vs=[x for x in off if x['axis']==comp];ks=[x for x in offkf if x['axis']==comp];tag='official_x' if comp==0 else 'official_y'
  for dim in ('win','face'):
   for val in sorted({x[dim] for x in vs}):
    rows.append({'split':'official','level':'video_share','dimension':f'{tag}:{dim}','stratum':val,'videos':sum(1 for x in vs if x[dim]==val),'keyframes':'','mean':sum(1 for x in vs if x[dim]==val)/len(vs),'ci_lo':'','ci_hi':'','better':'','worse':'','mean_shift_win':''})
  for dim in ('pos','persons','kf_face'):
   for val in sorted({x[dim] for x in ks}):
    sel=[x for x in ks if x[dim]==val]
    rows.append({'split':'official','level':'keyframe_share','dimension':f'{tag}:{dim}','stratum':val,'videos':'','keyframes':len(sel),'mean':len(sel)/len(ks),'ci_lo':'','ci_hi':'','better':'','worse':'','mean_shift_win':float(np.mean([x['shift'] for x in sel]))})
 a.output.parent.mkdir(parents=True,exist_ok=True)
 keys=['split','level','dimension','stratum','videos','keyframes','mean','ci_lo','ci_hi','better','worse','mean_shift_win']
 with open(a.output,'w',newline='') as f:
  wr=csv.DictWriter(f,keys);wr.writeheader()
  for r in rows:wr.writerow({k:(round(r[k],5) if isinstance(r.get(k),float) else r.get(k,'')) for k in keys})
 geo={}
 for x in off:geo.setdefault(f"axis{x['axis']}:{x['win']}",0);geo[f"axis{x['axis']}:{x['win']}"]+=1
 print(json.dumps({'h1':ab,'fit_and_eval_counts':fit_counts,'official_geometry':geo,'rows':len(rows)},indent=1))
if __name__=='__main__':main()
