"""AHA BF16 streaming scoring and a validated temporal-only competition package.

No training or label access. Native relevance + informativeness score, 1fps,
causal context across the whole video, fixed 85% frame budget, B3 spatial.
"""
import os,sys,json,time,hashlib,argparse,subprocess,math,zipfile
from pathlib import Path
os.environ.update(RANK='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled')
ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--model',required=True); ap.add_argument('--index',required=True); ap.add_argument('--spatial',required=True); ap.add_argument('--keep',type=float,default=.85); ap.add_argument('--chunk',type=int,default=8); ap.add_argument('--resume-compatible-code-sha'); ap.add_argument('--config',required=True); args=ap.parse_args()
root=Path(args.root); root.mkdir(parents=True,exist_ok=True); scoresdir=root/'scores_native_shape'; scoresdir.mkdir(exist_ok=True)
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
  values=heads(hh)[0,idx]; assert bool(torch.isfinite(values).all())
  if begin==0:
   # Identical full-token head matrix shapes and transformer/cache settings.
   # BF16 GEMM rounding depends on shape; selecting hidden states before the
   # head is mathematically equivalent but not numerically identical.
   native=model(inputs_embeds=inp,use_cache=True,return_dict=True)
   native_values=torch.stack([native.informative_logits[0,idx].float().softmax(-1)[:,1],native.relevance_logits[0,idx,0],native.uncertainty[0,idx,0]],-1)
   native_diff=float((native_values-values).abs().max())
   assert native_diff<1e-5,native_diff
   del native,native_values
  past=result.past_key_values
  out.extend(values.cpu().tolist())
 return out,native_diff

def decode(meta,indices):
 expr='+'.join('eq(n\\,%d)'%i for i in indices)
 vf=f'select={expr},scale=384:384:force_original_aspect_ratio=decrease,pad=384:384:(ow-iw)/2:(oh-ih)/2'
 raw=subprocess.check_output(['ffmpeg','-v','error','-threads','2','-noautorotate','-i',meta['video_path'],'-vf',vf,'-vsync','0','-f','rawvideo','-pix_fmt','rgb24','pipe:1'])
 n=len(raw)//(384*384*3); assert len(raw)==len(indices)*384*384*3,(meta['video_id'],n,len(indices))
 return [Image.fromarray(x) for x in np.frombuffer(raw,np.uint8).reshape(n,384,384,3)]


c=json.loads(Path(args.config).read_text()); data=Path(c['data_root'])
report=json.loads((data/'report.json').read_text()); splits=report['qc_summary']['pass_ids_by_split']
axes={r['video_id']:r for r in report['axis']}; bindings=[]
write(root/'public_run.json',dict(config=c,source_code_sha256=sha(__file__),parameter_count=params,system=system,query=query,loading=loading,head_identity=head_identity,weight_files=weight_files))
for split in ['train','dev']:
 for vid in splits[split]:
  a=axes[vid]; media=data/a['media_relpath']; axis=data/a['axis_array_relpath']
  assert sha(media)==a['media_sha256'] and sha(axis)==a['axis_array_sha256']
  z=np.load(axis); pts=z['timestamps']; assert z['raw_summary_membership'].shape==(10,len(pts))
  targets=np.arange(0,float(pts[-1])+1e-9,1.)
  ix=np.unique(np.abs(pts[:,None]-targets).argmin(0))
  bindings.append(dict(video_id=vid,split=split,media_sha256=a['media_sha256'],axis_sha256=a['axis_array_sha256'],sample_frame_ids=ix.tolist(),actual_pts=pts[ix].tolist()))
  target=scoresdir/(vid+'.json')
  if target.exists():
   assert json.loads(target.read_text())['run_code_sha256']==sha(__file__); continue
  tick=time.time(); images=decode(dict(video_id=vid,video_path=str(media)),ix.tolist())
  if split=='train' and vid==splits['train'][0]:
   probe=images[:16]; chunk_before=args.chunk
   batched,_=score_video(probe); args.chunk=1; sequential,_=score_video(probe); args.chunk=chunk_before
   static,_=score_video([probe[0]]*len(probe)); reverse,_=score_video(probe[::-1])
   write(root/'train_protocol_probe.json',dict(video_id=vid,sample_frame_ids=ix[:16].tolist(),chunk8=batched,chunk1=sequential,max_abs_head_difference=float(np.max(np.abs(np.array(batched)-np.array(sequential)))),static_first_frame_repeated=static,reversed_frames=reverse,note='TRAIN-only 16-frame numerical and causal position probes; chunk BF16 shape changes may cause rounding'))
  values,diff=score_video(images)
  write(target,dict(video_id=vid,split=split,frame_indices=ix.tolist(),actual_pts=pts[ix].tolist(),values=values,scores=[v[0]+v[1] for v in values],native_forward_max_abs_difference=diff,seconds=time.time()-tick,run_code_sha256=sha(__file__)))
  write(root/'progress.json',dict(status='RUNNING',completed=len(list(scoresdir.glob('*.json'))),total=40,latest_video_id=vid,wall_s=time.time()-start,peak_vram_bytes=torch.cuda.max_memory_allocated()))
  print('PUBLIC',split,vid,len(ix),round(time.time()-tick,2),flush=True)
write(root/'bindings.json',bindings)
write(root/'progress.json',dict(status='SCORED',completed=40,total=40,wall_s=time.time()-start,peak_vram_bytes=torch.cuda.max_memory_allocated()))
