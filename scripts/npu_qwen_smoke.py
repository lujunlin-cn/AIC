"""910A (Ascend) smoke test for the Qwen3-VL-32B point teacher.

Loads the local weights with transformers + torch_npu (fp16, layers split over
the visible NPUs with device_map=auto), asks the frozen N0/DENSE point prompt
on already-extracted official keyframes, and compares each reply with the
V100 vLLM reply stored in point_d2/points/<vid>.json.  Greedy decoding on both
sides; numeric differences between kernels can still change a few tokens, so
the report gives exact-match rate and point distance, not a pass/fail claim.

  --videos      official video ids (default 0 1 2)
  --max-keys    keyframes per video (default 6)
"""
import argparse,json,sys,time
from pathlib import Path
import torch,torch_npu  # noqa: F401  (registers the npu device)
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.qwen_subject_point import SYSTEM,USER,parse,prompt_sha


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',default='/data/aic/pretrained/qwen3_vl_32b_instruct')
 ap.add_argument('--frames-dir',type=Path,default=Path('/data/aic/experiments/QWEN32B_OFFICIAL_V2/keyframes_d2'))
 ap.add_argument('--ref-dir',type=Path,default=Path('/data/aic/experiments/QWEN32B_OFFICIAL_V2/point_d2/points'))
 ap.add_argument('--videos',nargs='*',default=['0','1','2']);ap.add_argument('--max-keys',type=int,default=6)
 ap.add_argument('--batch',type=int,default=1);ap.add_argument('--output',type=Path,required=True)
 # sdpa on 910A fails in the vision tower ("can not cast format when output is input", ERR01007)
 ap.add_argument('--attn',default='eager',choices=['eager','sdpa']);a=ap.parse_args()
 from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
 t0=time.time()
 proc=AutoProcessor.from_pretrained(a.model)
 model=Qwen3VLForConditionalGeneration.from_pretrained(a.model,dtype=torch.float16,device_map='auto',low_cpu_mem_usage=True,attn_implementation=a.attn).eval()
 load_s=time.time()-t0;print(json.dumps({'load_s':round(load_s,1),'attn':a.attn,'devices':sorted({str(p.device) for p in model.parameters()})}),flush=True)
 assert prompt_sha('point')=='918f0730f5407c7186a3208a9f76662c3121c8ea25475655d8ce3f5dc8acb53d','point prompt changed'
 rows=[];gen_s=0.
 for vid in a.videos:
  ref=json.loads((a.ref_dir/f'{vid}.json').read_text());W,H=ref['W'],ref['H']
  for k,r in ref['ratios'].items():
   rw,rh=r['ratio'];orient='portrait, narrower than the frame' if rw/rh<W/H else 'landscape, wider than the frame'
   text=USER.format(w=W,h=H,rw=rw,rh=rh,orient=orient)
   for j,key in list(enumerate(ref['keyframes']))[:a.max_keys]:
    img=Image.open(a.frames_dir/vid/f'{key}.png').convert('RGB')
    msgs=[{'role':'system','content':[{'type':'text','text':SYSTEM}]},{'role':'user','content':[{'type':'image','image':img},{'type':'text','text':text}]}]
    inp=proc.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to(model.device)
    t=time.time()
    with torch.no_grad():out=model.generate(**inp,max_new_tokens=64,do_sample=False)
    torch.npu.synchronize();dt=time.time()-t;gen_s+=dt
    reply=proc.batch_decode(out[:,inp['input_ids'].shape[1]:],skip_special_tokens=True)[0]
    p,s=parse(reply);q,_=parse(r['raw'][j])
    d=None if p is None or q is None else ((p['x']-q['x'])**2+(p['y']-q['y'])**2)**.5
    rows.append({'vid':vid,'ratio':k,'key':key,'npu':reply,'v100':r['raw'][j],'status':s,'exact':reply.strip()==r['raw'][j].strip(),'dist':d,'s':round(dt,2),'in_tokens':int(inp['input_ids'].shape[1])})
    print(json.dumps(rows[-1],ensure_ascii=False),flush=True)
 ds=[x['dist'] for x in rows if x['dist'] is not None]
 summ={'n':len(rows),'parse_ok':sum(x['status']=='ok' for x in rows),'exact':sum(x['exact'] for x in rows),
  'mean_dist':None if not ds else sum(ds)/len(ds),'max_dist':None if not ds else max(ds),
  'attn':a.attn,'load_s':round(load_s,1),'gen_s_per_query':round(gen_s/max(1,len(rows)),2),'npu_mem_gb':[round(torch.npu.max_memory_allocated(i)/2**30,1) for i in range(torch.npu.device_count())]}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'summary':summ,'rows':rows},ensure_ascii=False,indent=1)+'\n')
 print('SUMMARY',json.dumps(summ),flush=True)
if __name__=='__main__':main()
