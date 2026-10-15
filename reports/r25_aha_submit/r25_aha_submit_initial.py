"""AHA BF16 streaming scoring and a validated temporal-only competition package.

No training or label access. Native relevance + informativeness score, 1fps,
causal context across the whole video, fixed 85% frame budget, B3 spatial.
"""
import os,sys,json,time,hashlib,argparse,subprocess,math,zipfile
from pathlib import Path
os.environ.update(RANK='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled')
ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--model',required=True); ap.add_argument('--index',required=True); ap.add_argument('--spatial',required=True); ap.add_argument('--keep',type=float,default=.85); ap.add_argument('--chunk',type=int,default=8); args=ap.parse_args()
root=Path(args.root); root.mkdir(parents=True,exist_ok=True); scoresdir=root/'scores'; scoresdir.mkdir(exist_ok=True)
sys.path[:0]=['/root/autodl-tmp/aha_repo','/root/autodl-tmp/aha_repo/LLaVA_NeXT',str(Path(__file__).parent)]
import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from transformers import AutoTokenizer
from models.live_llava.video_head_live_llava_qwen import VideoHeadLiveLlavaQwenForCausalLM as Model,VideoHeadLiveLlavaQwenConfig as Config
from independent_submission_check import check

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,data):
 tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)); tmp.replace(p)

original_init=Model.__init__
def init_without_alias(self,config):
 original_init(self,config); del self.vision_encoder
Model.__init__=init_without_alias
start=time.time()
model,loading=Model.from_pretrained(args.model,config=Config.from_pretrained(args.model),torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map='cuda',output_loading_info=True)
loading={k:sorted(v) if isinstance(v,set) else v for k,v in loading.items()}
assert not loading['missing_keys'] and not loading['unexpected_keys'] and not loading['mismatched_keys'],loading
object.__setattr__(model,'vision_encoder',model.get_vision_tower())
model.eval(); model.currently_training=False
model.requires_grad_(False)
tok=AutoTokenizer.from_pretrained(args.model)
params=sum(p.numel() for p in model.parameters())
spatial_params=448718848+1511425+53104
assert params+spatial_params<=9_000_000_000, (params,spatial_params)
weight_files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(Path(args.model).glob('*.safetensors'))]
head_identity={}
with safe_open('/root/autodl-tmp/pretrained/aha_weights/adapter_model.safetensors',framework='pt') as f:
 for name in ['informative_head','relevance_head','uncertainty_head']:
  actual=getattr(model,name).weight.detach().cpu(); expected=f.get_tensor('base_model.model.'+name+'.weight').to(actual.dtype)
  diff=float((actual-expected).abs().max()); assert diff==0,(name,diff); head_identity[name]=diff
system='A multimodal AI assistant is helping users with some activities. Below is their conversation, interleaved with the list of video frames received by the assistant.'
query='Identify the most informative and important highlights of this video.'
prefix=tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':query}],add_stream_prompt=True,return_tensors='pt').cuda()
processor=model.get_vision_tower().image_processor
@torch.inference_mode()
def heads(h):
 info=model.informative_head(h).float().softmax(-1)[...,1]
 rel=model.relevance_head(h).float().sigmoid().squeeze(-1)
 unc=model.uncertainty_head(h).float().squeeze(-1)
 return torch.stack([info,rel,unc],-1)
@torch.inference_mode()
def score_video(images):
 past=None; out=[]; native_diff=None
 for begin in range(0,len(images),args.chunk):
  pixel=processor.preprocess(images[begin:begin+args.chunk],return_tensors='pt')['pixel_values'].cuda().to(torch.bfloat16)
  visual=model.visual_embed(pixel); n=len(pixel); nt=len(visual)//n
  assert nt==49 and bool(torch.isfinite(visual).all())
  inp=visual.view(1,-1,model.config.hidden_size)
  prompt_n=0
  if begin==0:
   text=model.get_input_embeddings()(prefix); prompt_n=text.shape[1]
   inp=torch.cat([text,inp],1)
  # Decode the native causal transformer; omit the unused 152064-vocabulary LM
  # projection, which does not feed any scoring head or recurrent state.
  result=model.model(inputs_embeds=inp,past_key_values=past,use_cache=True,return_dict=True)
  hh=result.last_hidden_state; idx=prompt_n+torch.arange(1,n+1,device='cuda')*nt-1
  values=heads(hh[0,idx]); assert bool(torch.isfinite(values).all())
  if begin==0:
   # Native forward cross-check on the first frame using the same prefix/context.
   one=inp[:,:prompt_n+nt]
   native=model(inputs_embeds=one,use_cache=False,return_dict=True)
   nv=torch.stack([native.informative_logits[0,-1].float().softmax(-1)[1],native.relevance_logits[0,-1,0],native.uncertainty[0,-1,0]])
   native_diff=float((nv-values[0]).abs().max()); assert native_diff<.05,native_diff
  past=result.past_key_values
  out.extend(values.cpu().tolist())
 return out,native_diff

def decode(meta,indices):
 expr='+'.join('eq(n\\,%d)'%i for i in indices)
 vf=f'select={expr},scale=384:384:force_original_aspect_ratio=decrease,pad=384:384:(ow-iw)/2:(oh-ih)/2'
 raw=subprocess.check_output(['ffmpeg','-v','error','-threads','2','-noautorotate','-i',meta['video_path'],'-vf',vf,'-vsync','0','-f','rawvideo','-pix_fmt','rgb24','pipe:1'])
 n=len(raw)//(384*384*3); assert len(raw)==len(indices)*384*384*3,(meta['video_id'],n,len(indices))
 return [Image.fromarray(x) for x in np.frombuffer(raw,np.uint8).reshape(n,384,384,3)]

metas=[json.loads(l) for l in Path(args.index).read_text().splitlines()]
assert len(metas)==426 and len({m['video_id'] for m in metas})==426
run=dict(experiment_id='R25_AHA_BF16_K085',model=args.model,parameter_count=params,spatial_parameter_count=spatial_params,total_parameters=params+spatial_params,parameter_limit=9_000_000_000,rule_basis='user confirmed official 9B parameter limit 2026-10-15',weight_files=weight_files,head_identity=head_identity,loading_info=loading,precision='bf16',attention='sdpa',sample_fps=1,chunk=args.chunk,system_prompt=system,query=query,score='softmax(informative)[1] + sigmoid(relevance)',keep=args.keep,index_sha256=sha(args.index),spatial_sha256=sha(args.spatial),source_code_sha256=sha(__file__),no_training=True)
write(root/'run_config.json',run)
print('MODEL_READY',params,'total_with_spatial',params+spatial_params,'load_s',time.time()-start,flush=True)
for i,meta in enumerate(metas):
 vid=meta['video_id']; target=scoresdir/(vid+'.json')
 if target.exists():
  old=json.loads(target.read_text()); assert old['run_code_sha256']==run['source_code_sha256']; continue
 tick=time.time(); total=meta['frame_count']; fps=meta['fps']
 indices=sorted(set(np.clip(np.rint(np.arange(max(1,math.ceil(meta['duration'])))*fps).astype(int),0,total-1).tolist()))
 images=decode(meta,indices)
 values,native_diff=score_video(images)
 record=dict(video_id=vid,frame_indices=indices,values=values,scores=[v[0]+v[1] for v in values],native_forward_max_abs_difference=native_diff,seconds=time.time()-tick,run_code_sha256=run['source_code_sha256'])
 write(target,record)
 progress=dict(status='RUNNING',completed=i+1,total=len(metas),wall_s=time.time()-start,latest_video_id=vid,latest_seconds=record['seconds'],peak_vram_bytes=torch.cuda.max_memory_allocated())
 write(root/'progress.json',progress)
 print('VIDEO',i+1,vid,'samples',len(indices),'seconds',round(record['seconds'],3),'wall',round(time.time()-start,2),flush=True)

spatial={r['video_id']:r for r in map(json.loads,Path(args.spatial).read_text().splitlines())}
assert set(spatial)=={m['video_id'] for m in metas}
# model_size_mb is a legacy platform field; write total parameters / 1e6 under
# the user-confirmed 9B/500M/100M parameter interpretation. Preserve file bytes
# separately in the manifest, never label this value as measured file size.
size_field=run['total_parameters']/1e6
rows=[]; pervideo=[]; bbox_checked=0
for meta in metas:
 vid=meta['video_id']; n=meta['frame_count']; d=json.loads((scoresdir/(vid+'.json')).read_text())
 x=np.asarray(d['frame_indices']); sc=np.asarray(d['scores']); allframes=np.arange(n)
 # Nearest sampled-frame ownership, ties to the earlier sample. Keep complete
 # temporal cells until the fixed frame budget is reached (same historical rule).
 owners=np.abs(allframes[:,None]-x[None,:]).argmin(1)
 ranked=np.argsort(-sc,kind='stable'); keep=np.zeros(n,dtype=bool); budget=max(1,round(args.keep*n))
 for j in ranked:
  keep[owners==j]=True
  if int(keep.sum())>=budget:break
 source=spatial[vid]; byframe={p['frame']:p['bboxes'] for p in source['predictions']}
 assert set(byframe)==set(range(n)),vid
 selected=[{'frame':f,'bboxes':byframe[f]} for f in range(n) if keep[f]]; bbox_checked+=len(selected)
 rows.append(dict(video_id=vid,targetRatioWH=meta['targetRatioWH'],model_size_mb=size_field,predictions=selected))
 pervideo.append(dict(video_id=vid,frame_count=n,kept=len(selected),keep_fraction=len(selected)/n,sample_count=len(x),score_std=float(sc.std()),native_forward_max_abs_difference=d['native_forward_max_abs_difference']))
package=root/'R25_AHA_BF16_K085'; package.mkdir(exist_ok=True); pred=package/'predictions.jsonl'
pred.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows))
validation=check(args.index,pred,require_size=True)
zpath=package/'R25_AHA_BF16_K085.zip'
with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:z.write(pred,'predictions.jsonl')
with zipfile.ZipFile(zpath) as z:
 assert z.namelist()==['predictions.jsonl']; assert hashlib.sha256(z.read('predictions.jsonl')).hexdigest()==sha(pred)
manifest=dict(submission_id='R25_AHA_BF16_K085',status='PACKAGED_NOT_UPLOADED',uploaded=False,official_platform_score=None,run=run,validation=validation,video_count=len(rows),prediction_count=bbox_checked,bbox_changes_on_common_frames=0,per_video=pervideo,model_size_mb_semantics='total parameters in millions per user-confirmed official parameter interpretation; NOT weight file MB',actual_aha_weight_bytes=sum(f['bytes'] for f in weight_files),spatial_component_bytes_reported=903762857,spatial_components_source='/data/aic/semifinal_20261001/submissions/LFM_V8_B3_SEMIFINAL/manifest.json',predictions_sha256=sha(pred),zip_sha256=sha(zpath),zip_bytes=zpath.stat().st_size,wall_s=time.time()-start,peak_vram_bytes=torch.cuda.max_memory_allocated())
write(package/'manifest.json',manifest); (package/'R25_AHA_BF16_K085.zip.sha256').write_text(manifest['zip_sha256']+'  '+zpath.name+'\n')
write(root/'progress.json',dict(status='PACKAGED',completed=426,package=str(package),zip_sha256=manifest['zip_sha256'],wall_s=time.time()-start))
print('PACKAGED',json.dumps(validation),manifest['zip_sha256'],flush=True)
