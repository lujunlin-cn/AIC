"""V6 E0: source-group level H3 pipeline metrics + full-prereg gate re-judging.

V5 pipeline_eval bootstrapped video x ratio units as if independent, but the
two RV units of one source video (axis0 + axis1) share footage: the sampling
unit was 'unit', not 'source'.  This script

  1. re-runs the per-unit paired pipeline IoUs (same code path as V5),
  2. SELF-CHECKS the unit-level aggregation against the saved V5 metrics.json
     (same flatten order, same bootstrap seed => CIs must match exactly),
  3. aggregates within source_group first (equal ratio weights, frozen here
     because the V5 prereg did not specify weights), then bootstraps sources,
  4. re-judges gate1 with BOTH prereg conditions (dataset-equal dev AND
     rv_dev-alone best(V,VQ)-G > 0) and gate2 at source level.

No retraining: heads, mu/sd come from the V5 h3_heads.pt snapshot.
Outputs: reports/v6_source_group_metrics.csv (per unit) + summary JSON.
"""
import argparse,hashlib,json,pickle
from pathlib import Path
import numpy as np

from scripts.v5_h3_visual_scorer import load_all,make_head,score_unit,keyframe_eval


def per_unit_pipeline(units,scores):
 from aic.max_window_path import qwen_centres_interp,ema_offsets
 from scripts.teacher_diag_eval import frame_iou
 rows=[]
 for i,u in enumerate(units):
  s=scores[i];pts=[list(p) if p is not None else None for p in u['pts']]
  for k,j in enumerate(u['J']):pts[j][u['comp']]=float((u['offs'][k,s[k].argmax()]+u['s']/2)/u['L'])
  def run(p):
   c,_=qwen_centres_interp(p,u['keys'],u['reset'],u['raw'],u['comp']);c=np.where(u['face'],u['raw'],c)
   off=ema_offsets(c,u['reset'],u['W'],u['H'],u['ratio'])[u['gf']]
   return float(frame_iou(off,u['W'],u['H'],u['ratio'],u['gt']).mean())
  a=run([list(p) if p is not None else None for p in u['pts']]);b=run(pts)
  grp=('rv' if u['ds']=='rv' else 'live')+':'+str(u['vid'])
  rows.append({'ds':u['ds'],'comp':int(u['comp']),'vid':str(u['vid']),'src':grp,'mother':a,'selector':b,'delta':b-a})
 return rows


def boot(d,seed=20260929):
 d=np.asarray(d,float)
 if not len(d):return None
 bs=np.random.default_rng(seed).choice(d,(5000,len(d))).mean(1)
 return {'n':len(d),'mean':float(d.mean()),'ci95':[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))],'better':int((d>1e-4).sum()),'worse':int((d<-1e-4).sum()),'worst':float(d.min())}


def v5_style_aggregates(rows):
 # exactly the V5 flatten: per (ds,comp) dict insertion order == unit order
 per={}
 for r in rows:per.setdefault((r['ds'],r['comp']),{}).setdefault(r['vid'],[]).append((r['mother'],r['selector']))
 vids={k:{v:float(np.mean(np.array(x),0)[1]-np.mean(np.array(x),0)[0]) for v,x in d.items()} for k,d in per.items()}
 out={'all':boot([v for k in vids for v in vids[k].values()]),
  'axis0':boot([v for k in vids for v in vids[k].values() if k[1]==0]),
  'axis1':boot([v for k in vids for v in vids[k].values() if k[1]==1])}
 return out,vids


def source_group_aggregates(rows):
 per={}
 for r in rows:per.setdefault(r['src'],[]).append(r['delta'])
 src={g:float(np.mean(v)) for g,v in per.items()}
 def axis(comp):
  pp={}
  for r in rows:
   if r['comp']==comp:pp.setdefault(r['src'],[]).append(r['delta'])
  return boot([float(np.mean(v)) for g,v in sorted(pp.items())])
 return {'all':boot(list(src.values())),'axis0':axis(0),'axis1':axis(1),'by_group':dict(sorted(src.items()))}


def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--tables',type=Path,default=Path('/data/aic/experiments/V5_H3_VISUAL/tables'))
 ap.add_argument('--visual',type=Path,default=Path('/data/aic/experiments/V5_H3_VISUAL/features'))
 ap.add_argument('--heads',type=Path,default=Path('/data/aic/experiments/V5_H3_VISUAL/train/h3_heads.pt'))
 ap.add_argument('--v5metrics',type=Path,default=Path('reports/h3/metrics.json'))
 ap.add_argument('--output',type=Path,default=Path('reports/v6_source_group_metrics.csv'))
 ap.add_argument('--summary',type=Path,default=Path('reports/v6_source_group_summary.json'))
 ap.add_argument('--cuda',default='cuda:2');a=ap.parse_args()
 import torch
 dev=a.cuda if torch.cuda.is_available() else 'cpu'
 v5=json.loads(a.v5metrics.read_text())
 H_=torch.load(a.heads,map_location=dev)
 heads={}
 for k in ('G','V','VQ'):
  net,proj=make_head(k,dev);net.load_state_dict(H_[k]['net']);net.eval()
  if H_[k]['proj'] is not None:proj.load_state_dict(H_[k]['proj']);proj.eval()
  heads[k]=(net,proj,H_[k]['mu'].to(dev),H_[k]['sd'].to(dev))
 D=load_all(a.tables,a.visual,['rv_dev','rv_confirm2','live_dev','live_val'])
 csv=['split,head,ds,comp,vid,src_group,mother_iou,selector_iou,delta']
 summary={'self_check_vs_v5':{},'unit_level_v5_style':{},'source_group':{},'gate1_full_prereg':{},'gate2_source_level':{}}
 for name in ('rv_dev','rv_confirm2','live_dev','live_val'):
  units=D[name];summary['unit_level_v5_style'][name]={};summary['source_group'][name]={}
  for k in ('G','V','VQ'):
   net,proj,mu,sd=heads[k]
   s=[score_unit(net,proj,k,u,dev,mu,sd) for u in units]
   rows=per_unit_pipeline(units,s)
   for r in rows:csv.append(f"{name},{k},{r['ds']},{r['comp']},{r['vid']},{r['src']},{r['mother']:.6f},{r['selector']:.6f},{r['delta']:.6f}")
   v5agg,_=v5_style_aggregates(rows)
   saved=v5['pipeline'][name][k]
   ok={}
   for key in ('all','axis0','axis1'):
    new,old=v5agg[key],saved.get(key)
    if new is None or old is None:ok[key]=(new is None and old is None);continue
    ok[key]=abs(new['mean']-old['mean'])<1e-9 and abs(new['ci95'][0]-old['ci95'][0])<1e-9 and abs(new['ci95'][1]-old['ci95'][1])<1e-9
   summary['self_check_vs_v5'][f'{name}.{k}']=ok
   summary['unit_level_v5_style'][name][k]={'all':v5agg['all'],'axis1':v5agg['axis1']}
   sg=source_group_aggregates(rows)
   summary['source_group'][name][k]=sg
   print(json.dumps({'split':name,'head':k,'selfcheck':ok,'v5_all':v5agg['all'],'src_all':sg['all']}),flush=True)
 # gate1 full prereg: dataset-equal dev AND rv_dev alone, both best(V,VQ)-G>0
 def kfu(name,k,grp):return v5['keyframe'][name][k][grp]['sel_minus_mother']
 deveq={k:(kfu('rv_dev',k,'rv:all')+kfu('live_dev',k,'live:all'))/2 for k in ('G','V','VQ')}
 rvonly={k:kfu('rv_dev',k,'rv:all') for k in ('G','V','VQ')}
 cond_eq=max(deveq['V'],deveq['VQ'])-deveq['G']
 cond_rv=max(rvonly['V'],rvonly['VQ'])-rvonly['G']
 summary['gate1_full_prereg']={'dev_equal':deveq,'rv_dev_only':rvonly,'cond_dev_equal':cond_eq,'cond_rv_dev_alone':cond_rv,
  'pass':bool(cond_eq>0 and cond_rv>0),
  'note':'cond values are keyframe sel_minus_mother differences (visual head minus G) on each dev set'}
 sg_dev=summary['source_group']['rv_dev'];sg_conf=summary['source_group']['rv_confirm2']
 gate2={}
 for k in ('V','VQ'):
  d,c=sg_dev[k],sg_conf[k]
  gate2[k]={'rv_dev':{'mean':d['all']['mean'],'ci_low':d['all']['ci95'][0],'axis1_mean':(d['axis1']['mean'] if d['axis1'] else None)},
   'confirm':{'mean':c['all']['mean'],'ci_low':c['all']['ci95'][0],'axis1_mean':(c['axis1']['mean'] if c['axis1'] else None)}}
  g=gate2[k]['rv_dev'];gg=gate2[k]['confirm']
  gate2[k]['pass_rv_dev']=bool(g['mean']>0 and g['ci_low']>-0.002 and (g['axis1_mean'] is None or g['axis1_mean']>=-0.002))
  gate2[k]['pass_confirm']=bool(gg['mean']>0 and gg['ci_low']>-0.002 and (gg['axis1_mean'] is None or gg['axis1_mean']>=-0.002))
 summary['gate2_source_level']=gate2
 a.output.parent.mkdir(parents=True,exist_ok=True);a.summary.parent.mkdir(parents=True,exist_ok=True)
 Path(a.output).write_text('\n'.join(csv)+'\n')
 Path(a.summary).write_text(json.dumps(summary,indent=1,default=float)+'\n')
 print(json.dumps({'gate1':summary['gate1_full_prereg'],'gate2':gate2}),flush=True)
 print('done')
if __name__=='__main__':main()
