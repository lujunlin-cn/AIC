"""Reproducible CPU readiness audit using local datasets and the verified remote snapshot.

No official media/labels are opened; train-only labels supply all calibration analysis.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile
from PIL import Image

from aic.scale_audit import gaicd_retention, repair_sacd_splits, sha256, summary
from aic.scale_geometry import box_iou_xywh, checked_scale_action, fit_scale_bins_from_rows, maximum_window


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def audit(root: Path, output: Path, snapshot: Path):
    bins=json.loads((output/'config/scale_bins.json').read_text())['values']
    split=repair_sacd_splits(root/'datasets/SACD')
    write(output/'config/sacd_image_split.json',split)
    schema=Counter(); explicit=0
    for group in split['groups']:
        if group['split']!='train':continue
        for subject in group['subjects']:
            d=json.loads((root/'datasets/SACD'/subject['annotation']).read_text())
            schema.update(d.keys())
            explicit+=sum(len(d.get(k,[])) for k in ('ranked_pairs','preference_pairs','pairwise_preferences'))
    sacd={'new_split_source_counts':split['source_counts'],'new_split_subject_counts':split['subject_counts'],
          'manifest_sha256':sha256(output/'config/sacd_image_split.json'),'training_annotation_schema_counts':dict(schema),
          'explicit_ranked_pairs':explicit,'best_box_not_converted_to_pairs':True,
          'dpo_included':False,'source_disjoint':True}
    write(output/'audit/sacd_split_audit.json',sacd)
    gaicd=root/'datasets/GAICD';media={}
    for line in (gaicd/'processed/media.jsonl').read_text().splitlines():
        d=json.loads(line)
        if 'journal' in d.get('release','') and d.get('official_split')=='train':media[d['item_id']]=d
    rows=[]
    for line in (gaicd/'processed/annotations.jsonl').read_text().splitlines():
        d=json.loads(line)
        if d.get('variant')!='journal':continue
        p=Path(d['annotation_path']);ann=json.loads(p.read_text()) if d.get('media_uid','').split(':')[-1] in media else None
        # ID is taken from the manifest join, never from a positional array index.
        if ann is None:continue
        key=str(ann.get('image'))
        if key not in media:continue
        m=media[key]
        rows.append({'dataset_source':'GAICD','source_id':m.get('group_id') or m['media_uid'],
                     'image_id':key,'split':'train','release':'journal','image_width':ann['W'],
                     'image_height':ann['H'],'target_ratio':[9,16], 'center_xy':None,
                     'candidates':[{'crop_xywh':b,'mos':mos,'valid':valid}
                                   for b,mos,valid in zip(ann['boxes_xywh'],ann['mos'],ann['valid'])]})
    retention=gaicd_retention(rows,bins,1.)
    retention['quality_threshold_status']='not_calibrated; no preferences generated'
    retention_b = gaicd_retention([dict(row, target_ratio=[16,9]) for row in rows], bins, 1.)
    retention['by_target_ratio'] = {
        '9:16': {k:v for k,v in retention.items() if k not in {'effective_candidates_by_image','preference_pairs_by_source'}},
        '16:9': {k:v for k,v in retention_b.items() if k not in {'effective_candidates_by_image','preference_pairs_by_source'}}}
    write(output/'audit/gaicd_candidate_retention.json',retention)
    r=snapshot/'repro/liveytvc_full_retrain_20261023/data';m=json.loads((r/'build_manifest.json').read_text())
    tr=set(m['train_videos_list']);val=set(m['dev_videos']); hashes=defaultdict(list)
    for rel,h in m['images_sha256'].items():hashes[h].append(rel)
    cross=[v for v in hashes.values() if len({Path(p).stem.split('_')[0] in tr for p in v})>1]
    quarantine=sorted({Path(p).stem.split('_')[0] for group in cross for p in group if Path(p).stem.split('_')[0] in tr})
    # Quarantine complete train sources, retaining the historical dev set intact.
    live_split={'train':sorted(tr-set(quarantine)),'val':sorted(val),
                'quarantined_train_sources':quarantine,'policy':'exclude_train_sources_with_any_exact_dev_image_duplicate',
                'original_manifest_sha256':sha256(r/'build_manifest.json')}
    write(output/'config/live_source_split.json',live_split)
    write(output/'audit/live_duplicate_quarantine.json',{'cross_split_identical_image_groups':len(cross),
          'groups':cross,'quarantined_train_sources':quarantine,'historical_train_sources':len(tr),
          'historical_val_sources':len(val),'clean_train_sources':len(live_split['train']),
          'clean_val_sources':len(val),'cross_split_after_quarantine':0,
          'note':'A byte-identical group may be static/blank content; quarantine avoids treating it as independent evidence.'})
    live_train=json.loads((r/'train_a.json').read_text())
    live_train=[row for row in live_train if row['group'] not in quarantine]
    # Human-scale analysis uses original pixels. GT-center oracles below are named
    # upper bounds, not P1 predictions and are never emitted as training labels.
    scalars={'LIVE_YT_VC':[],'GNMC':[]}; geometry={'LIVE_YT_VC':[],'GNMC':[]}
    by_source=defaultdict(list)
    for row in live_train:
        w,h=row['original_size'];g=row['human_raw_crop_original_pixels'];box=[g['left'],g['top'],g['right']-g['left'],g['bottom']-g['top']]
        by_source[row['group']].append(box[2]/maximum_window(w,h,(9,16))[0])
        geometry['LIVE_YT_VC'].append((w,h,(9,16),box))
    scalars['LIVE_YT_VC']=[sum(v)/len(v) for v in by_source.values()]
    with ZipFile(root/'.cache_videoora_data/GNMC/GNMC.zip') as z:
        annotations=json.loads(z.read('GNMC/json/train.json'))
        for d in annotations:
            with Image.open(io.BytesIO(z.read('GNMC/train/'+Path(d['filename']).name))) as im:w,h=im.size
            x1,y1,x2,y2=d['crop_bboxes']['16:9'];box=[x1*w,y1*h,(x2-x1)*w,(y2-y1)*h]
            scalars['GNMC'].append(box[2]/maximum_window(w,h,(16,9))[0]);geometry['GNMC'].append((w,h,(16,9),box))
    analysis={'current_bins':bins,'current_adjacent_gaps':[b-a for a,b in zip(bins,bins[1:])],
              'validation_or_test_used_for_fitting':False,'distribution_unit':'LIVE video-mean scale; GNMC image scale',
              'distributions':{k:summary(v) for k,v in scalars.items()},'gt_center_upper_bound_by_dataset':{}}
    for dataset,values in scalars.items():
        proposed=fit_scale_bins_from_rows([{'split':'train','scale_ratio':v} for v in values],8)
        entries={}
        for label,grid in [('current8',bins),('dataset_train_quantile8',proposed),
                           ('dataset_train_quantile16',fit_scale_bins_from_rows([{'split':'train','scale_ratio':v} for v in values],16))]:
            scores=[];counts=Counter()
            for w,h,ratio,box in geometry[dataset]:
                center=[box[0]+box[2]/2,box[1]+box[3]/2]
                options=[(box_iou_xywh(a['crop_xywh'],box),i) for i,scale in enumerate(grid)
                         if (a:=checked_scale_action(w,h,ratio,center,scale))['valid']]
                if options:
                    score,i=max(options);scores.append(score);counts[i]+=1
            entries[label]={'bins':grid,'oracle_iou_at_GT_center_NOT_P1':summary(scores),
                            'gt_nearest_deployed_iou_bin_counts':dict(counts)}
        analysis['gt_center_upper_bound_by_dataset'][dataset]=entries
    analysis['decision']='current grid remains provisional; lock only after train P1 coverage. No validation-driven refit.'
    write(output/'audit/scale_bin_analysis.json',analysis)
    # Do not fill unavailable P1-conditioned counts with GT-center statistics.
    coverage={'status':'blocked_missing_training_P1_centers','valid_scale_sft_labels':None,
              'p1_fixed_center_scale_optimizable_samples':None,'position_mismatch_rate':None,
              'threshold':None,'threshold_status':'requires_train_only_P1_quality_distribution',
              'original_samples':{'LIVE_YT_VC_A':len(live_train),'LIVE_YT_VC_B_transposed':len(live_train),
                                  'GNMC_native_B':7000,'GNMC_transposed_A':7000},
              'centers_from_GT_or_image_center_substituted':False,
              'position_center_cache_required':True,'validation_used_to_set_train_labels':False}
    write(output/'audit/position_conditioned_scale_coverage.json',coverage)
    (output/'audit/position_conditioned_scale_coverage.md').write_text(
        '# Position-conditioned Scale coverage\n\nStatus: blocked.\n\n'
        f"LIVE clean train: {len(live_train)} frames / {len(live_split['train'])} sources; GNMC: 7000 train sources.\n\n"
        'Training P1 centers are absent. Effective labels, position mismatch, bin coverage and Scale improvement space are unmeasured. '
        'GT-centered bounds in scale_bin_analysis.json are separate diagnostics and are not P1 evidence. '
        'After generating hash-bound P1 centers, run position_conditioned_coverage() on train only; calibrate and lock thresholds before validation.\n')
    return {'SACD':sacd,'GAICD_retention':{k:v for k,v in retention.items() if k not in ['effective_candidates_by_image','preference_pairs_by_source']},
            'LIVE_quarantine':quarantine,'coverage':coverage}


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('artifacts/scale_sft_dpo'))
    p.add_argument('--snapshot',type=Path,default=Path('artifacts/scale_sft_dpo/remote_snapshot'))
    a=p.parse_args();print(json.dumps(audit(a.root,a.output,a.snapshot),indent=2,ensure_ascii=False))
if __name__=='__main__':main()
