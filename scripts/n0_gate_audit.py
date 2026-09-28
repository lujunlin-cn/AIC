"""T0 audit of the frozen N0/P0 packages on the official set (no GT used).

1. Rebuild N0 / P0 from OBS cache + frozen Qwen points; must equal the scored
   predictions.jsonl byte-for-byte in bbox values (max abs 0).
2. Gate order: N0 = EMA( where(face_valid, B0 raw, Qwen point) ), face_valid =
   YuNet chosen >= 0.  Count face frames whose N0 box still differs from B0
   (EMA state inherited from a Qwen frame in the same shot).
3. Keyframes: decoded-frame indices in range, every shot start present, gap
   statistics, resized-image aspect error (coordinate inverse transform).
4. Where N0 and P0 differ: face frames only (by construction) - verified.
"""
import argparse,json
from pathlib import Path
import numpy as np
from aic.max_window_path import geometry,to_crops,ema_offsets,qwen_centres,shots


def load(p):
 return {r['video_id']:np.array([x['bboxes'] for x in r['predictions']]) for r in map(json.loads,open(p))}


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--index',required=True);ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--points',type=Path,required=True)
 ap.add_argument('--keyframes',type=Path,required=True);ap.add_argument('--n0',required=True);ap.add_argument('--p0',required=True);ap.add_argument('--b0',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 N0,P0,B0=load(a.n0),load(a.p0),load(a.b0);rows=[];rep={'N0':0.,'P0':0.}
 for r in map(json.loads,open(a.index)):
  vid=r['video_id'];z=np.load(a.cache/f'{vid}_t.npz');W,H=int(z['W']),int(z['H']);ratio=r['targetRatioWH'];w,h,axis=geometry(W,H,ratio);reset=z['reset'].astype(bool);face=z['chosen']>=0;n=len(reset)
  q=json.loads((a.points/f'{vid}.json').read_text());mt=json.loads((a.keyframes/vid/'meta.json').read_text());keys=q['keyframes']
  if keys!=mt['keyframes']:raise ValueError('keyframe list mismatch '+vid)
  sh=shots(reset);starts=[s for s,_ in sh];missing=[s for s in starts if s not in set(keys)]
  from PIL import Image
  im=Image.open(a.keyframes/vid/f'{keys[0]}.png');aspect_err=abs(im.size[0]/im.size[1]-W/H)/(W/H)
  gaps=np.diff(keys+[n]);row={'video_id':vid,'frames':n,'fps':q['fps'],'axis':None if axis is None else 'xy'[axis],'keyframes':len(keys),'shots':len(sh),'shot_starts_missing':len(missing),
   'max_gap_s':float(gaps.max()/q['fps']),'aspect_err':float(aspect_err),'in_range':bool(min(keys)>=0 and max(keys)<n),'face_rate':float(face.mean())}
  if axis is not None:
   comp=axis;raw=z['raw'][:,comp];c,src=qwen_centres(q['ratios']['t']['points'],keys,reset,raw,comp)
   p0=np.asarray(to_crops(W,H,ratio,ema_offsets(c,reset,W,H,ratio)));n0=np.asarray(to_crops(W,H,ratio,ema_offsets(np.where(face,raw,c),reset,W,H,ratio)))
   rep['P0']=max(rep['P0'],float(np.abs(p0-P0[vid]).max()));rep['N0']=max(rep['N0'],float(np.abs(n0-N0[vid]).max()))
   d_nb=np.abs(N0[vid]-B0[vid]).max(1)>1e-9;d_np=np.abs(N0[vid]-P0[vid]).max(1)>1e-9
   row.update({'qwen_frames':float((src=='qwen').mean()),'n0_qwen_driven':float(((src=='qwen')&~face).mean()),'n0_diff_b0':float(d_nb.mean()),'n0_diff_b0_on_face':float(d_nb[face].mean()) if face.any() else None,
    'n0_diff_p0':float(d_np.mean()),'n0_diff_p0_on_noface':float(d_np[~face].mean()) if (~face).any() else None,
    'switches_per_shot':float(np.mean([np.abs(np.diff(face[s:e].astype(int))).sum() for s,e in sh]))})
  rows.append(row)
 f=lambda k:[x[k] for x in rows if x.get(k) is not None]
 out={'reproduction_max_abs':rep,'videos':len(rows),'keyframes_total':int(sum(f('keyframes'))),'shot_starts_missing_total':int(sum(f('shot_starts_missing'))),'all_in_range':all(f('in_range')),
  'aspect_err_max':float(max(f('aspect_err'))),'max_gap_s_p50_p90_max':np.percentile(f('max_gap_s'),[50,90,100]).round(3).tolist(),
  'n0_qwen_driven_frame_rate_mean':float(np.mean(f('n0_qwen_driven'))),'n0_diff_b0_on_face_mean':float(np.mean(f('n0_diff_b0_on_face'))),
  'n0_diff_p0_on_noface_mean':float(np.mean(f('n0_diff_p0_on_noface'))),'face_noface_switches_per_shot_mean':float(np.mean(f('switches_per_shot'))),
  'axis_counts':{k:sum(1 for x in rows if x['axis']==k) for k in ('x','y',None)},'per_video':rows}
 a.output.mkdir(parents=True,exist_ok=True);(a.output/'audit.json').write_text(json.dumps(out,indent=1)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='per_video'}))
if __name__=='__main__':main()
