from __future__ import annotations
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import t as student_t

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(20261004)


def expected_f_dp(p: torch.Tensor, d: list[int], w: torch.Tensor, g: int) -> torch.Tensor:
    prob = p.new_ones(1)
    mass = p.new_zeros(1)
    for i, cost in enumerate(d):
        prev_p, prev_m = prob, mass
        prob = (1-p[i])*torch.nn.functional.pad(prev_p,(0,cost)) + p[i]*torch.nn.functional.pad(prev_p,(cost,0))
        mass = (1-p[i])*torch.nn.functional.pad(prev_m,(0,cost)) + p[i]*torch.nn.functional.pad(prev_m + w[i]*prev_p,(cost,0))
    k = torch.arange(len(prob), dtype=p.dtype)
    if g == 0:
        return prob[0]
    return (2*mass/(k+g)).sum()


def exhaustive(p: torch.Tensor, d: list[int], w: torch.Tensor, g: int) -> torch.Tensor:
    val = p.new_zeros(())
    for a_tuple in itertools.product([0,1], repeat=len(d)):
        a = p.new_tensor(a_tuple)
        prob = torch.where(a.bool(),p,1-p).prod()
        k = sum(x*y for x,y in zip(a_tuple,d))
        reward = 1.0 if k+g == 0 else 2*float((a*w).sum())/(k+g)
        val = val+prob*reward
    return val

max_val=max_grad=max_marg=0.0
cases=40
for case in range(cases):
    n=int(rng.integers(2,9))
    d=rng.integers(1,5,size=n).tolist()
    h=np.array([rng.integers(0,x+1) for x in d])
    if not h.sum(): h[0]=1
    g=int(h.sum())
    w=torch.tensor(h*rng.uniform(.2,1,size=n))
    z=torch.tensor(rng.normal(size=n),requires_grad=True)
    p=z.sigmoid()
    a=expected_f_dp(p,d,w,g)
    b=exhaustive(p,d,w,g)
    ga=torch.autograd.grad(a,z,retain_graph=True)[0]
    gb=torch.autograd.grad(b,z,retain_graph=True)[0]
    marg=[]
    for i in range(n):
        pon=p.detach().clone(); poff=pon.clone()
        pon[i]=1.; poff[i]=0.
        marg.append(float(exhaustive(pon,d,w,g)-exhaustive(poff,d,w,g)))
    gm=p.detach()*(1-p.detach())*torch.tensor(marg)
    max_val=max(max_val,abs(float(a-b)))
    max_grad=max(max_grad,float((ga-gb).abs().max()))
    max_marg=max(max_marg,float((ga-gm).abs().max()))
    assert -1e-12 <= float(a) <= 1+1e-12
    assert torch.allclose(a,b,atol=1e-12,rtol=1e-12)
    assert torch.allclose(ga,gb,atol=1e-12,rtol=1e-12)
    assert torch.allclose(ga,gm,atol=1e-12,rtol=1e-12)

mse=np.array([.6273,.6276,.6514]); dp=np.array([.6625,.6626,.6625]); delta=dp-mse
half=student_t.ppf(.975,2)*delta.std(ddof=1)/np.sqrt(3)
# Reproduce the unit-mismatch path reviewed in v11_native_occ_labels.py.
p=torch.tensor([1.,0.,0.,0.]); d=[100]*4; w=torch.tensor([100.,0.,0.,0.])
invalid_reward=float(expected_f_dp(p,d,w,8))
correct_reward=float(expected_f_dp(p,d,w,100))
assert invalid_reward>1 and correct_reward==1
out={
 'scope':'Independent synthetic arithmetic only; no production training or platform inference run',
 'seed_means':{'mse':float(mse.mean()),'exactdp':float(dp.mean()),'delta':float(delta.mean()),'per_seed_delta':delta.tolist()},
 'illustrative_seed_t_interval':{'ci95':[(float(delta.mean()-half)),float(delta.mean()+half)],'caveat':'3 independent paired seeds and approximate normality assumed; not a valid final hierarchical CI'},
 'numeric_verification':{'n_cases':cases,'max_abs_value_error':max_val,'max_abs_gradient_error':max_grad,'max_abs_conditional_gradient_error':max_marg,'passed':True},
 'unit_mismatch_counterexample':{'cost_per_block':100,'positive_blocks':1,'wrong_gt_count':8,'correct_gt_count':100,'wrong_reward':invalid_reward,'correct_reward':correct_reward},
 'a0_recovery_fraction':{str(s):(s-33.85)/(34.73-33.85) for s in (34.7,34.3,33.9)},
 'iid_cluster_se_factor_equal_cluster_size':(1954/984)**.5,
 'fixed_keep_temporal_ceiling_reference_p04722_k080':2*.4722/(.8+.4722),
 'decode_onepass_cpu_hours_reference':{str(fps):234159/fps/3600 for fps in [100,300]},
 'native426_window_count':{'four_windows_per_video':426*4,'eight_windows_per_video':426*8},
 'no_slide_fraction':159/426,
 'no_slide_only_delta05_temporal':.5/100*426/159,
}
Path('/mnt/data/aic_round4_work/numeric_checks.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
