#!/usr/bin/env python3
"""Plot paired held-out predictions on one deterministic outer repeat/seed."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--analysis',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();d=json.loads(Path(a.analysis).read_text());a0=d['per_video']['A0'];bs=d['per_video']['DeiT_S']
    order=sorted(a0,key=lambda v:bs[v]['summary_f1']-a0[v]['summary_f1']);chosen=order[:3]+order[-3:]
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);manifest=[]
    for vid in chosen:
        fig,axes=plt.subplots(2,1,figsize=(11,5.5),sharex=True,gridspec_kw={'height_ratios':[3,1]})
        records=[]
        for model,color in [('A0','#377eb8'),('DeiT_S','#e6550d'),('ViT_B','#4daf4a')]:
            files=list(Path(a.root).glob(f'RG_NCV_{model}_r0_*_s20260925/*_outer/{vid}.npz'))
            if len(files)!=1:raise ValueError((model,vid,files))
            with np.load(files[0]) as z:
                t=z['timestamps'];p=z['probabilities'];y=z['labels']
            metric=json.loads((files[0].parent/'metrics.json').read_text());threshold=metric['job']['threshold']
            axes[0].plot(t,p,label=model,color=color,lw=1.1)
            axes[1].plot(t,(p>=threshold).astype(float)+{'A0':0,'DeiT_S':1.4,'ViT_B':2.8}[model],color=color,lw=.8)
            records.append({'model':model,'prediction_path':str(files[0]),'threshold':threshold})
        axes[0].plot(t,y,label='human importance',color='black',alpha=.7,lw=1.2)
        axes[0].set_ylim(0,1);axes[0].set_ylabel('score');axes[0].legend(ncol=4,loc='upper right')
        axes[0].set_title(f"{vid} | {a0[vid]['category']} | summary delta DeiT-A0={bs[vid]['summary_f1']-a0[vid]['summary_f1']:+.3f}\n{a0[vid]['title'][:95]}",fontsize=10)
        axes[1].set_ylabel('selection');axes[1].set_xlabel('seconds');axes[1].set_yticks([.5,1.9,3.3],['A0','DeiT-S','ViT-B'])
        fig.tight_layout();fig.savefig(out/(vid+'_timeline.png'),dpi=130);plt.close(fig)
        manifest.append({'video_id':vid,'category':a0[vid]['category'],'title':a0[vid]['title'],'outer_repeat':0,'seed':20260925,'records':records})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(chosen))
if __name__=='__main__':main()
