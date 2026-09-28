"""P3 check: alignment offset of legacy vs contract targets, and label-state inventory.

No training: measures (1) mean |anchor time - centre of the clip its target
comes from| under both conventions, (2) target disagreement, (3) per-source
label states on native QVH train/dev (unlabelled stays IGNORE).
"""
import argparse,json
from pathlib import Path
import numpy as np
from aic.temporal_contract import anchor_targets,legacy_targets,label_states,clip_centres,POS,NEG,IGNORE


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--records',required=True);ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 recs=[json.loads(l) for l in open(a.records)];off_leg=[];off_new=[];dis=[];sup_leg=[];sup_new=[];states={}
 for r in recs:
  with np.load(a.cache/(r['vid']+'.npz')) as z:t=z['timestamps'].astype(float)
  lab=np.asarray(r['labels'],float);m=np.asarray(r['mask'],bool);c=clip_centres(len(lab))
  yl,ml=legacy_targets(t,lab,m);yn,mn=anchor_targets(t,lab,m)
  b=np.minimum((t/2).astype(int),len(lab)-1);off_leg+=list(c[b]-t)
  near=np.clip(np.rint((t-1)/2).astype(int),0,len(lab)-1);off_new+=list(c[near]-t)
  both=ml&mn;dis+=list(np.abs(yl-yn)[both]);sup_leg.append(ml.mean());sup_new.append(mn.mean())
  s=label_states(lab,m);d=states.setdefault(r['split'],{'POS':0,'NEG':0,'IGNORE':0,'clips':0})
  d['POS']+=int((s==POS).sum());d['NEG']+=int((s==NEG).sum());d['IGNORE']+=int((s==IGNORE).sum());d['clips']+=len(s)
 out={'legacy_offset_s':{'mean':float(np.mean(off_leg)),'abs_mean':float(np.mean(np.abs(off_leg)))},'contract_nearest_centre_offset_s':{'mean':float(np.mean(off_new)),'abs_mean':float(np.mean(np.abs(off_new)))},
  'target_abs_diff_where_both_supervised':{'mean':float(np.mean(dis)),'p90':float(np.percentile(dis,90))},'supervised_anchor_rate':{'legacy':float(np.mean(sup_leg)),'contract':float(np.mean(sup_new))},
  'label_states_qvh_native':states,'note':'QVH native has no reliable generic NEG: unrated clips are IGNORE; rated low clips are only query-conditioned','test_content_used':False}
 a.output.mkdir(parents=True,exist_ok=True);(a.output/'metrics.json').write_text(json.dumps(out,indent=1)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
