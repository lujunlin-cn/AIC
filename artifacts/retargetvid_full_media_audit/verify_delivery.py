"""Offline consistency check of the delivered evidence; no media redecode."""
import hashlib
import json
from pathlib import Path

P=Path(__file__).resolve().parent
ROOT=P.parents[1]
def get(name): return json.loads((P/(name+'.json')).read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def rows(path): return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]

m=get('media_inventory'); a=get('frame_annotation_alignment'); c=get('coordinate_audit')
s=get('split_leakage_audit'); r=get('adapter_read_test'); d=get('decision')
expected={f'{i:03d}' for i in [*range(1,101),*range(601,701)]}
assert {x['video_id'] for x in m['source_videos']}==expected
assert m['decoded_verified']==200 and m['unique_video_sha256']==200 and not m['errors']
assert a['actual_annotation_files']==a['aligned_annotation_files']==2400
assert sum(x['frame_count'] for x in m['source_videos'])==a['decoded_source_frames']==122684
assert a['annotated_box_rows']==c['raw_boxes_checked']==1472208
assert not a['failures']
assert c['roundtrip_ltrb_maxabs']==0 and c['raw_annotations_unchanged']
assert not any(c[k] for k in ['outside_image_raw_boxes','nonpositive_windows','ratio_mismatch_1px','not_max_window_1px'])
assert s['exposed_eval_sources']==200 and s['fresh_validation_sources']==s['training_sources_created']==0
assert s['protected_original_sha256_before']==s['protected_original_sha256_after']
for key,path in {'ledger':'configs/holdout_exposure_v1.json','release':'reports/releases/spatial_crop_v1.RELEASE.json',
                 'quality':'reports/releases/spatial_crop_v1.quality.json'}.items():
    assert sha(ROOT/'ext_data'/path)==s['protected_original_sha256_after'][key]
rv=rows(P/'registered_manifests/RetargetVid/media.jsonl')
dhf=rows(P/'registered_manifests/DHF1K/media.jsonl')
assert len(rv)==len(dhf)==200
for x,y in zip(rv,dhf):
    assert x['group_id']==y['group_id'] and x['media_path']==y['media_path'] and x['sha256']==y['sha256']
    assert x['release_split']==y['release_split']=='exposed_eval'
    assert not x['fresh_validation_eligible'] and not x['training_eligible']
    assert x['project_split'] in ['holdout_dev_exposed','holdout_confirm_exposed']
ann=rows(P/'registered_manifests/RetargetVid/annotations.jsonl')
assert len(ann)==2400
for x in m['source_videos']:
    record=json.loads((P/'decoded_sources'/(x['video_id']+'.json')).read_text())
    assert sha(P/'timelines'/Path(x['timeline_path']).name)==x['timeline_sha256']==record['timeline_sha256']
    assert record['source_sha256']==x['sha256'] and record['actual_DenseCropAdapter_RGB_read']
assert r['passed'] and r['raw_annotation_files_read']==2400 and r['annotation_units_read']==400
assert r['Train_units']==r['fresh_validation_units']==0
assert d['all_200_media_available'] and d['all_2400_annotations_aligned'] and d['safe_for_exposed_spatial_diagnostics']
assert not any(d[k] for k in ['fresh_validation_eligible','training_started','GPU_used','P1_P2_weights_changed'])
assert get('runs/initial/media_inventory')['full_decode_passes_this_run']==200
assert m['full_decode_passes_this_run']==0 and m['verified_decode_cache_reused']==200
assert get('remote_cpu_validation')['passed'] and get('post_audit_identity_check')['passed']
code=get('remote_source_sha256')
assert all(sha(ROOT/name)==value for name,value in code.items())
print(json.dumps(dict(passed=True,real_media_sources=200,aligned_annotation_files=2400,
      shared_source_groups=200,actual_Adapter_ratio_units=400,fresh_validation_sources=0,
      decoded_source_frames=122684,raw_box_rows=1472208,remote_source_hashes_match_local=len(code))))
