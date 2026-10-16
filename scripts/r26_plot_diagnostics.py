"""Plot aggregate diagnostics only; never publish test per-video scores."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=json.loads(Path('reports/r26/diagnostics.json').read_text()); x=[(z['bin']+.5)/10 for z in r['macro_head_time_deciles']]
fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for key in ['info','rel']:
    axs[0].plot(x,[z[key] for z in r['macro_head_time_deciles']],marker='o',label=key)
axs[0].set(xlabel='Normalized sampled frame position',ylabel='Mean probability (video macro)',title='R25: opposing head time trends'); axs[0].legend(); axs[0].grid(alpha=.2)
for label,name in [('R25 original','R25_original'),('AHA exact 85%','R26_AHA_EXACT085_B3'),('Center exact 85%','R26_POS_CENTER_EXACT085_B3')]:
    axs[1].plot([1,2,3],r['mask_distributions'][name]['macro_keep_by_third'],marker='o',label=label)
axs[1].set(xticks=[1,2,3],xticklabels=['First third','Middle third','Last third'],ylabel='Video macro keep fraction',ylim=(.5,1.02),title='Matched frame budgets'); axs[1].legend(); axs[1].grid(alpha=.2)
fig.savefig('reports/r26/distribution.png',dpi=160)
