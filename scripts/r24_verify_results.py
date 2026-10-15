"""Independent original-frame metric reconstruction and saved-head replay."""
import json, hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
root=Path('/root/autodl-tmp/experiments/R24_EXEC_20261015')
data=Path('/root/autodl-tmp/experiments/R22_INSTRUMENT_20261015')
d=json.loads((root/'metrics.json').read_text()); bindings=json.loads((root/'bindings.json').read_text())
assert set(d['train_ids']).isdisjoint(d['dev_ids'])
checks=[]; torch.set_num_threads(2)
for name,res in d['results'].items():
    backbone=name.split('_')[0]
    ck=torch.load(root/(name+'.pt'),map_location='cpu',weights_only=False)
    net=nn.Sequential(nn.Linear(768,512),nn.GELU(),nn.Linear(512,1)); net.load_state_dict(ck['head']); net.eval()
    for keep in ['0.85','0.6','0.25']:
        per=[]; largest=0.
        for v in res[keep]['videos']:
            vid=v['video_id']; z=np.load(data/'axis_arrays'/(vid+'.npz')); pts=z['timestamps']; mask=z['raw_summary_membership'].astype(bool)
            binding=next(b for b in bindings if b['video_id']==vid)
            block=np.abs(pts[:,None]-np.array(binding['actual_pts'])).argmin(1)
            k=round(float(keep)*len(pts)); order=sorted(range(len(pts)),key=lambda i:(-v['scores'][block[i]],i))
            tp=mask[:,order[:k]].sum(1); f=float(np.mean(2*tp/(k+mask.sum(1))))
            assert abs(f-v['f'])<1e-12 and k==v['kept']
            per.append(f)
            features=np.load(root/(vid+'.npz'))[backbone]
            with torch.inference_mode():
                x=torch.tensor((features-ck['mu'])/ck['sd']); p=net(x).squeeze(-1).sigmoid().numpy()
            largest=max(largest,float(np.max(np.abs(p-v['scores']))))
        assert abs(np.mean(per)-res[keep]['f'])<1e-12
        assert largest<1e-5
        checks.append(dict(arm=name,keep=keep,metric_reconstruction='EXACT',saved_head_cpu_max_abs_difference=largest))
report=dict(status='PASS',checks=checks,train_dev_source_intersection=0,protected_opened=False)
(root/'verification.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
