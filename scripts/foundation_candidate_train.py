"""Native QVH frozen representation control: train/dev only, masked targets."""
import argparse,json,random,time,os,platform
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from aic.foundation import load_encoder,encode_video
from aic.models import TemporalUNet,sha256_file


def ndcg(pred,target):
 from scipy.stats import rankdata
 # Expected DCG over tied predictions, using mean gain in each tie block.
 order=np.argsort(-pred,kind='stable'); gain=np.asarray(target)[order];p=np.asarray(pred)[order]
 discount=1/np.log2(np.arange(len(p))+2);dcg=0.;start=0
 while start<len(p):
  end=start+1
  while end<len(p) and p[end]==p[start]:end+=1
  dcg+=gain[start:end].mean()*discount[start:end].sum();start=end
 ideal=np.sort(target)[::-1]@discount
 return float(dcg/ideal) if ideal>0 else None


def evaluate(head, items, device):
 rows=[]
 with torch.inference_mode():
  for rec,feat,times in items:
   x=torch.from_numpy(feat).float().to(device)[None];score=head(x)[0].sigmoid().cpu().numpy()
   for q in rec['queries']:
    ids=np.array(q['relevant_clip_ids']);y=np.mean(q['saliency_scores'],axis=1)/4.
    p=np.interp(ids*2.+1,times,score);truth=y>=.75;pred=p>=.35
    tp=np.sum(pred&truth);den=pred.sum()+truth.sum()
    rho=float(spearmanr(p,y).statistic) if np.std(p)>0 and np.std(y)>0 else None
    rows.append({'video_id':rec['vid'],'qid':q['qid'],'f1':float(2*tp/den) if den else 1.,'spearman':rho,'ndcg':ndcg(p,y),'selected_ratio':float(pred.mean())})
 # one vote per video, average queries first
 per=[]
 for vid in sorted({r['video_id'] for r in rows}):
  group=[r for r in rows if r['video_id']==vid]
  per.append({'video_id':vid,**{key:float(np.mean([r[key] for r in group if r[key] is not None])) if any(r[key] is not None for r in group) else None for key in ['f1','spearman','ndcg','selected_ratio']}})
 return {'per_video':per,**{key:float(np.mean([r[key] for r in per if r[key] is not None])) for key in ['f1','spearman','ndcg','selected_ratio']}}


def main():
 p=argparse.ArgumentParser();p.add_argument('--kind',choices=['deit','videomae','videomae_large','internvideo'],required=True);p.add_argument('--records',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--part',type=int,default=0);p.add_argument('--parts',type=int,default=1);p.add_argument('--extract-only',action='store_true');p.add_argument('--fit-only',action='store_true');p.add_argument('--seed',type=int,default=20260926);a=p.parse_args()
 torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
 a.output.mkdir(parents=True,exist_ok=True);cache=a.output/'features';cache.mkdir(exist_ok=True)
 records=list(map(json.loads,Path(a.records).read_text().splitlines()))
 if {r['split'] for r in records}!={'train','dev'}:raise ValueError('train/dev only')
 device='cuda';start=time.time()
 config={**vars(a),'output':str(a.output),'threshold':.35,'target_binary_threshold':.75,'sample_fps':2,'anchor_stride_samples':4,'clip_frames':{'deit':1,'videomae':16,'videomae_large':16,'internvideo':8}[a.kind],'target':'mean human saliency /4 among supplied query ratings; unannotated clips masked, not negatives','official_f_video':None,'selection':'dev video-macro Spearman; tie dev NDCG','epochs':20,'lr':.001,'weight_decay':.0001,'records_sha256':sha256_file(a.records),'script_sha256':sha256_file(__file__),'adapter_sha256':sha256_file('aic/foundation.py'),'environment':{'python':platform.python_version(),'torch':torch.__version__},'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES')}
 (a.output/f'config_part{a.part}.json').write_text(json.dumps(config,indent=2)+'\n')
 if not a.fit_only:
  model,_=load_encoder(a.kind,'/data/aic/pretrained',device)
  for i,r in enumerate(records):
   if i%a.parts!=a.part:continue
   dest=cache/(r['vid']+'.npz')
   if dest.exists():continue
   feat,t,idx=encode_video(model,a.kind,r['video_path'],device,batch=2 if a.kind=='internvideo' else 4)
   np.savez_compressed(dest,features=feat,timestamps=t,frame_indices=idx)
   print(json.dumps({'stage':'extract','kind':a.kind,'index':i,'total':len(records),'timesteps':len(t),'seconds':time.time()-start}),flush=True)
  (a.output/f'extraction_part{a.part}.json').write_text(json.dumps({'seconds':time.time()-start,'peak_vram':torch.cuda.max_memory_allocated(),'parameters':sum(p.numel() for p in model.parameters())})+'\n')
  del model;torch.cuda.empty_cache()
 if a.extract_only:return
 items={s:[] for s in ['train','dev']}
 for r in records:
  with np.load(cache/(r['vid']+'.npz')) as z:items[r['split']].append((r,z['features'],z['timestamps']))
 head=TemporalUNet(items['train'][0][1].shape[1]).to(device);opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001)
 history=[];best=None;trainstart=time.time();rng=np.random.default_rng(a.seed)
 for epoch in range(20):
  head.train();losses=[]
  for i in rng.permutation(len(items['train'])):
   r,x,t=items['train'][i];bins=np.minimum((t/2).astype(int),len(r['labels'])-1)
   y=torch.tensor(np.asarray(r['labels'])[bins],device=device,dtype=torch.float32);mask=torch.tensor(np.asarray(r['mask'])[bins],device=device)
   if not mask.any():continue
   z=head(torch.tensor(x,device=device)[None])[0];loss=torch.nn.functional.binary_cross_entropy_with_logits(z[mask],y[mask])
   if not torch.isfinite(loss):raise FloatingPointError('loss')
   opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step();losses.append(loss.item())
  head.eval();met=evaluate(head,items['dev'],device);entry={'epoch':epoch,'loss':float(np.mean(losses)),**met};history.append(entry)
  state={'model':head.state_dict(),'input_dim':head.input_dim,'epoch':epoch,'config':config}
  torch.save(state,a.output/'last.pt')
  if best is None or (met['spearman'],met['ndcg'])>(best['spearman'],best['ndcg']):best=entry;torch.save(state,a.output/'best.pt')
  print(json.dumps({k:v for k,v in entry.items() if k!='per_video'}),flush=True)
 result={'run_id':'NATIVE_CANDIDATE_'+a.kind+'_V1','best':best,'history':history,'training_seconds':time.time()-trainstart,'head_bytes':(a.output/'best.pt').stat().st_size,'config':config,'status':'dev_evaluated_not_submission'}
 (a.output/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
