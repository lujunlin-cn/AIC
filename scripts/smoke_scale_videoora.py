"""Real-weight CPU smoke, separate from formal training and GPU budget.

This never claims that structural/meta probes are a real-model smoke pass.
A public, hash-bound train sample with an actual P1 center is required.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--sample',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threads',type=int,default=2)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import torch
    from aic.scale_execution import save_checkpoint, resume_checkpoint
    from aic.scale_objectives import dpo_loss
    from aic.scale_preferences import fixed_center_sft_target
    from aic.scale_geometry import checked_scale_action
    from aic.scale_trainer import set_trainable_parameter_allowlist
    from aic.scale_videoora import create_backend
    c=json.loads(a.config.read_text());r=json.loads(a.sample.read_text())
    if r.get('split')!='train' or r.get('position_source')!='P1_frozen':
        raise ValueError('real smoke requires a public training sample with actual P1 center')
    torch.set_num_threads(a.threads);torch.manual_seed(c['training']['seed'])
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    started=time.monotonic();backend=create_backend(c,'cpu')
    bins=c['data']['scale_bins']['values']
    label=fixed_center_sft_target(r,bins,0.)
    if label['scale_id'] is None:raise ValueError('position_shift_invalid')
    before={n:v.detach().clone() for n,v in backend.model.named_parameters() if v.requires_grad}
    optimizer=torch.optim.AdamW([v for v in backend.model.parameters() if v.requires_grad],lr=1e-5)
    backend.train_mode();loss=backend.completion_loss([r],[label['scale_id']]);loss.backward()
    gradients={n:float(v.grad.norm()) for n,v in backend.model.named_parameters() if v.requires_grad and v.grad is not None}
    if not torch.isfinite(loss) or not any(v>0 for v in gradients.values()):raise ValueError('real SFT backward failed')
    # No optimizer step: this is a differentiation/checkpoint integrity test.
    digest=save_checkpoint(a.output/'checkpoint',backend,optimizer,0,'cpu_real_smoke',None)
    with torch.no_grad():
        for n,v in backend.model.named_parameters():
            if n in before:v.add_(.01)
    resume_checkpoint(a.output/'checkpoint',backend,optimizer,'cpu_real_smoke')
    assert all(torch.equal(dict(backend.model.named_parameters())[n],v) for n,v in before.items())
    optimizer.zero_grad(set_to_none=True);backend.model.eval();backend.reference_frozen=True
    backend.model.requires_grad_(False)
    chosen=label['scale_id'];rejected=next(i for i,s in enumerate(bins) if i!=chosen and checked_scale_action(r['image_width'],r['image_height'],r['target_ratio'],r['center_xy'],s)['valid'])
    with torch.no_grad():
        rc,_=backend.completion_logps([r],[chosen]);rr,_=backend.completion_logps([r],[rejected])
    backend.reference_frozen=False;set_trainable_parameter_allowlist(backend.model,c['training']['trainable_parameter_allowlist'])
    pc,_=backend.completion_logps([r],[chosen]);pr,_=backend.completion_logps([r],[rejected])
    dpo=dpo_loss(pc,pr,rc,rr,.1);dpo.backward()
    assert torch.isfinite(dpo) and any(v.grad is not None and v.grad.abs().sum()>0 for v in backend.model.parameters() if v.requires_grad)
    assert backend._parent_digest()==backend.parent_state_sha256
    report={'status':'passed','real_weights':True,'device':'cpu','cuda_initialized':torch.cuda.is_initialized(),
            'optimizer_steps':0,'sft_loss':float(loss.detach()),'dpo_loss':float(dpo.detach()),
            'sft_gradients':gradients,'checkpoint_sha256':digest,'checkpoint_restore_exact':True,
            'parent_immutable':True,'geometry':label,'elapsed_seconds':time.monotonic()-started}
    (a.output/'real_smoke.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
