"""Real-weight differentiation smoke; no optimizer steps or formal training.

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


def run_smoke(backend,c,r,output):
    output.mkdir(parents=True,exist_ok=False)
    import torch
    from aic.scale_execution import save_checkpoint, resume_checkpoint
    from aic.scale_objectives import dpo_loss
    from aic.scale_preferences import fixed_center_sft_target
    from aic.scale_geometry import checked_scale_action
    from aic.scale_trainer import set_trainable_parameter_allowlist
    if r.get('split')!='train' or r.get('position_source')!='P1_frozen':
        raise ValueError('real smoke requires a public training sample with actual P1 center')
    torch.manual_seed(c['training']['seed']);started=time.monotonic()
    bins=c['data']['scale_bins']['values']
    backend._activate(scale=True,ratio=r['target_ratio'])
    generated=backend.predict([r])[0]
    generated_geometry=checked_scale_action(r['image_width'],r['image_height'],r['target_ratio'],r['center_xy'],bins[generated])
    if not generated_geometry['valid']:raise ValueError('generated action shifts frozen P1 center')
    label=fixed_center_sft_target(r,bins,0.)
    if label['scale_id'] is None:raise ValueError('position_shift_invalid')
    before={n:v.detach().clone() for n,v in backend.model.named_parameters() if v.requires_grad}
    optimizer=torch.optim.AdamW([v for v in backend.model.parameters() if v.requires_grad],lr=1e-5)
    backend.train_mode();loss=backend.completion_loss([r],[label['scale_id']]);loss.backward()
    gradients={n:float(v.grad.norm()) for n,v in backend.model.named_parameters() if v.requires_grad and v.grad is not None}
    if not torch.isfinite(loss) or not any(v>0 for v in gradients.values()):raise ValueError('real SFT backward failed')
    # No optimizer step: this is a differentiation/checkpoint integrity test.
    digest=save_checkpoint(output/'checkpoint',backend,optimizer,0,'real_smoke',None)
    with torch.no_grad():
        for n,v in backend.model.named_parameters():
            if n in before:v.add_(.01)
    resume_checkpoint(output/'checkpoint',backend,optimizer,'real_smoke')
    assert all(torch.equal(dict(backend.model.named_parameters())[n],v) for n,v in before.items())
    optimizer.zero_grad(set_to_none=True);backend.model.eval();backend.reference_frozen=True
    backend.model.requires_grad_(False)
    chosen=label['scale_id'];rejected=next(i for i,s in enumerate(bins) if i!=chosen and checked_scale_action(r['image_width'],r['image_height'],r['target_ratio'],r['center_xy'],s)['valid'])
    with torch.no_grad():
        rc,_=backend.completion_logps([r],[chosen]);rr,_=backend.completion_logps([r],[rejected])
        rc2,_=backend.completion_logps([r],[chosen]);rr2,_=backend.completion_logps([r],[rejected])
    reference_deterministic=torch.equal(rc,rc2) and torch.equal(rr,rr2)
    if not reference_deterministic:raise ValueError('reference dropout/numerical reproducibility failed')
    backend.reference_frozen=False;set_trainable_parameter_allowlist(backend.model,c['training']['trainable_parameter_allowlist'])
    pc,_=backend.completion_logps([r],[chosen]);pr,_=backend.completion_logps([r],[rejected])
    if not torch.equal(pc,rc) or not torch.equal(pr,rr):raise ValueError('initial policy/reference differ')
    dpo=dpo_loss(pc,pr,rc,rr,.1)
    swapped=dpo_loss(pr,pc,rr,rc,.1)
    # Check the objective's swap symmetry at float32 completion-logp inputs.
    # Repeating checkpointed BF16 parameter backprop is not an exact-arithmetic
    # sign test: reductions can accumulate in a different order. Actual model
    # parameter gradients are checked separately by the backward below.
    grad=torch.autograd.grad(dpo,(pc,pr),retain_graph=True)
    inverse_grad=torch.autograd.grad(swapped,(pc,pr),retain_graph=True)
    swap_gradient_error=max(float((one+two).abs().max()) for one,two in zip(grad,inverse_grad))
    if swap_gradient_error>1e-5:raise ValueError('chosen/rejected gradient sign check failed')
    dpo.backward()
    assert torch.isfinite(dpo) and any(v.grad is not None and v.grad.abs().sum()>0 for v in backend.model.parameters() if v.requires_grad)
    assert backend._parent_digest()==backend.parent_state_sha256
    report={'status':'passed','real_weights':True,'device':backend.device,'cuda_initialized':torch.cuda.is_initialized(),
            'optimizer_steps':0,'sft_loss':float(loss.detach()),'dpo_loss':float(dpo.detach()),
            'generated_scale_id':generated,'target_ratio':r['target_ratio'],
            'generated_deployment_geometry':generated_geometry,
            'sft_gradients':gradients,'checkpoint_sha256':digest,'checkpoint_restore_exact':True,
            'reference_deterministic_eval':reference_deterministic,
            'initial_policy_equals_reference':True,'swapped_loss':float(swapped.detach()),
            'swap_gradient_max_sign_error':swap_gradient_error,
            'swap_gradient_check_space':'float32 completion log probabilities',
            'reference_and_policy_same_image_prompt_geometry':True,
            'preprocessing_contract':backend.preprocessing_contract,
            'parent_immutable':True,'geometry':label,'elapsed_seconds':time.monotonic()-started}
    (output/'real_smoke.json').write_text(json.dumps(report,indent=2)+'\n')
    optimizer.zero_grad(set_to_none=True)
    return report


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--sample',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--device',default='cpu')
    a=p.parse_args()
    import torch
    from aic.scale_videoora import create_backend
    torch.set_num_threads(a.threads)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    c=json.loads(a.config.read_text());r=json.loads(a.sample.read_text())
    backend=create_backend(c,a.device)
    run_smoke(backend,c,r,a.output)
if __name__=='__main__':main()
