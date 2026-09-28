"""P1 all-select diagnosis + P3 minimal temporal ablation on native QVH train/dev.

Uses only cached VideoMAE-L features (train/dev) and V0 best.pt; never touches
official test content.  Crop is held fixed, so with exact frame matching the
AIC F reduces to a temporal set F1 over the full timeline:
  F = 2|P∩G| / (|P|+|G|)   on 2 s clips of each dev video.
GT proxies (both reported, neither is AIC GT):
  G_hi  = clips with mean saliency/4 >= .75 (unannotated clips are negatives)
  G_rel = annotated (query-relevant) clips
Rows: B0 all-select, B1 V0 head @ .35, B1q V0 head top-q, E1 head retrained with
unannotated clips as 0 targets (full timeline), E2 E1 + top-q per video.
Thresholds / q are chosen on TRAIN only; dev-oracle values are reported as
upper bounds, never used for selection.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from aic.models import TemporalUNet


def f_set(pred,gt):
 den=pred.sum()+gt.sum();return float(2*(pred&gt).sum()/den) if den else 1.


def clip_scores(score,times,n):
 # V0 eval convention: clip i centred at 2i+1 s, linear interpolation on anchor times
 return np.interp(np.arange(n)*2.+1,times,score)


def load(records,cache):
 items={'train':[],'dev':[]}
 for r in records:
  with np.load(cache/(r['vid']+'.npz')) as z:
   items[r['split']].append({'vid':r['vid'],'x':z['features'].astype(np.float32),'t':z['timestamps'].astype(np.float64),
    'labels':np.asarray(r['labels'],np.float64),'mask':np.asarray(r['mask'],bool)})
 return items


def infer(head,items,device):
 out=[]
 with torch.inference_mode():
  for it in items:
   z=head(torch.from_numpy(it['x']).to(device)[None])[0].float().cpu().numpy()
   out.append((z,1/(1+np.exp(-z))))
 return out


def fit_full(items,dim,device,seed,epochs=20):
 torch.manual_seed(seed);rng=np.random.default_rng(seed)
 head=TemporalUNet(dim).to(device);opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001);log=[]
 for ep in range(epochs):
  head.train();losses=[]
  for i in rng.permutation(len(items)):
   it=items[i];bins=np.minimum((it['t']/2).astype(int),len(it['labels'])-1)
   # E1: every clip is supervised; unannotated clips carry target 0 (not relevant)
   y=torch.tensor(it['labels'][bins],device=device,dtype=torch.float32)
   z=head(torch.from_numpy(it['x']).to(device)[None])[0];loss=torch.nn.functional.binary_cross_entropy_with_logits(z,y)
   opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step();losses.append(loss.item())
  log.append(float(np.mean(losses)))
 head.eval();return head,log


def rule_select(s,rule,val):
 if rule=='thr':return s>=val
 k=max(1,int(round(val*len(s))));idx=np.argsort(-s,kind='stable')[:k];p=np.zeros(len(s),bool);p[idx]=True;return p


def evaluate(scores,items,rule,val,gt_key):
 return [f_set(rule_select(clip_scores(s,it['t'],len(it['labels'])),rule,val),gt(it,gt_key)) for (_,s),it in zip(scores,items)]


def gt(it,key):
 return (it['labels']>=.75)&it['mask'] if key=='hi' else it['mask'].copy()


GRID={'thr':np.round(np.arange(.05,.96,.05),2),'topq':np.round(np.arange(.1,1.01,.05),2)}


def choose(scores,items,rule,gt_key):
 vals=[(float(np.mean(evaluate(scores,items,rule,v,gt_key))),float(v)) for v in GRID[rule]]
 return max(vals)[1]


def boot(a,b,seed=0,n=5000):
 d=np.asarray(a)-np.asarray(b);rng=np.random.default_rng(seed);m=[d[rng.integers(0,len(d),len(d))].mean() for _ in range(n)]
 return {'mean':float(d.mean()),'ci95':[float(np.percentile(m,2.5)),float(np.percentile(m,97.5))],'positive_videos':int((d>0).sum()),'videos':len(d)}


def main():
 p=argparse.ArgumentParser();p.add_argument('--records',required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--v0',required=True)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=20260927);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False);device='cuda';start=time.time()
 records=[json.loads(l) for l in open(a.records)];items=load(records,a.cache)
 # ---- P1: label / mask statistics
 stats={}
 for sp,its in items.items():
  lab=np.concatenate([i['labels'] for i in its]);m=np.concatenate([i['mask'] for i in its])
  stats[sp]={'videos':len(its),'clips':int(len(lab)),'mask_coverage':float(m.mean()),'label_mean_in_mask':float(lab[m].mean()),
   'label_ge_075_in_mask':float((lab[m]>=.75).mean()),'label_quantiles_in_mask':np.percentile(lab[m],[10,25,50,75,90]).round(4).tolist(),
   'label_max_outside_mask':float(lab[~m].max()) if (~m).any() else None,'hi_rate_full_timeline':float(((lab>=.75)&m).mean())}
 # time mapping: anchor times vs clip bins
 t0=items['dev'][0]['t'];tm={'anchor_times_first':t0[:6].round(3).tolist(),'anchor_step_s':float(np.median(np.diff(t0))),
  'train_bin_rule':'(t/2).astype(int)','eval_rule':'interp at 2i+1 s','max_offset_s':1.0}
 # ---- V0 head
 ck=torch.load(a.v0,map_location=device,weights_only=False);v0=TemporalUNet(ck['input_dim']).to(device);v0.load_state_dict(ck['model']);v0.eval()
 sv={sp:infer(v0,its,device) for sp,its in items.items()}
 dev_sig=np.concatenate([s for _,s in sv['dev']]);dev_logit=np.concatenate([z for z,_ in sv['dev']])
 per_min=[float(s.min()) for _,s in sv['dev']]
 v0diag={'epoch':ck['epoch'],'dev_sigmoid_quantiles':np.percentile(dev_sig,[0,1,10,50,90,100]).round(4).tolist(),'dev_logit_quantiles':np.percentile(dev_logit,[0,1,10,50,90,100]).round(4).tolist(),
  'dev_frac_ge_035':float((dev_sig>=.35).mean()),'dev_videos_all_selected':int(sum(m>=.35 for m in per_min)),'dev_per_video_min_sigmoid_min':float(min(per_min)),
  'dev_within_video_std_mean':float(np.mean([s.std() for _,s in sv['dev']])),
  'dev_sigmoid_in_mask_mean':float(np.mean(np.concatenate([clip_scores(s,it['t'],len(it['labels']))[it['mask']] for (_,s),it in zip(sv['dev'],items['dev'])]))),
  'dev_sigmoid_outside_mask_mean':float(np.mean(np.concatenate([clip_scores(s,it['t'],len(it['labels']))[~it['mask']] for (_,s),it in zip(sv['dev'],items['dev'])])))}
 # ---- E1 retrain (full-timeline supervision)
 e1,e1log=fit_full(items['train'],ck['input_dim'],device,a.seed);se={sp:infer(e1,its,device) for sp,its in items.items()}
 const=lambda its,v:[(np.zeros(len(i['t'])),np.full(len(i['t']),v)) for i in its]
 results=[];per={}
 for g in ('hi','rel'):
  rows=[('B0_ALL_SELECT','all select (every clip)',const(items['dev'],1.),'thr',0.,None),
        ('B1_V0_THR035','V0 head, fixed .35',sv['dev'],'thr',.35,None),
        ('B1_V0_TRAINTHR','V0 head, threshold chosen on train',sv['dev'],'thr',choose(sv['train'],items['train'],'thr',g),None),
        ('B1Q_V0_TOPQ','V0 head, per-video top-q chosen on train',sv['dev'],'topq',choose(sv['train'],items['train'],'topq',g),None),
        ('E1_FULLNEG_THR','E1 head (unannotated=0), threshold chosen on train',se['dev'],'thr',choose(se['train'],items['train'],'thr',g),None),
        ('E2_FULLNEG_TOPQ','E1 head, per-video top-q chosen on train',se['dev'],'topq',choose(se['train'],items['train'],'topq',g),None),
        ('ORACLE_V0_THR','upper bound: V0 dev-oracle threshold (not selectable)',sv['dev'],'thr',choose(sv['dev'],items['dev'],'thr',g),'oracle'),
        ('ORACLE_E1_THR','upper bound: E1 dev-oracle threshold (not selectable)',se['dev'],'thr',choose(se['dev'],items['dev'],'thr',g),'oracle')]
  base=None
  for rid,desc,sc,rule,val,flag in rows:
   f=evaluate(sc,items['dev'],rule,val,g);sel=[float(rule_select(clip_scores(s,it['t'],len(it['labels'])),rule,val).mean()) for (_,s),it in zip(sc,items['dev'])]
   if rid=='B0_ALL_SELECT':base=f
   results.append({'run_id':rid,'gt_proxy':g,'description':desc,'rule':rule,'value':val,'dev_videos':len(f),'dev_f_mean':float(np.mean(f)),
    'selected_ratio_mean':float(np.mean(sel)),'vs_B0':None if rid=='B0_ALL_SELECT' else boot(f,base),'selectable':flag is None});per[f'{rid}@{g}']=f
 out={'protocol':'QVH native dev temporal set-F1 proxy, crop fixed, 2 s clips','test_content_used':False,'label_stats':stats,'time_mapping':tm,'v0':v0diag,
  'e1_train_loss':e1log,'results':results,'per_video':per,'dev_vids':[i['vid'] for i in items['dev']],'elapsed_seconds':time.time()-start,'official_f_video':None}
 (a.output/'metrics.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
 torch.save({'model':e1.state_dict(),'input_dim':ck['input_dim']},a.output/'e1_head.pt')
 for r in results:print(json.dumps({k:r[k] for k in ('run_id','gt_proxy','value','dev_f_mean','selected_ratio_mean')}|({'ci':r['vs_B0']['ci95']} if r['vs_B0'] else {})),flush=True)
if __name__=='__main__':main()
