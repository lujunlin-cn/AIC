"""CPU contracts for the real P1 preflight; no fake model success claims."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from aic.scale_parent import OriginalP1Projector, atomic_json, file_sha
from scripts.generate_p1_scale_positions import choose_rows
from scripts.check_scale_gate import dataset_use_review_ready
from scripts.audit_scale_live_media import audit as audit_live_media
from scripts.isolate_scale_public_labels import isolate


def test_source_limits_keep_every_view_and_frame_together():
    rows = [dict(dataset_source=d,source_id=s,split='train',frame_id=f)
            for d in ['LIVE_YT_VC','GNMC'] for s in ['one','two'] for f in range(3)]
    videos, stills = choose_rows(rows, {'LIVE_YT_VC','GNMC'},1,1)
    assert len(videos)==len(stills)==1
    assert len(videos[0][1])==len(stills[0][1])==3
    with pytest.raises(ValueError,match='Test'):
        choose_rows([dict(rows[0],split='test')], {'LIVE_YT_VC'})


def test_singleton_mode_calls_exact_original_functions():
    import numpy as np
    from contextlib import nullcontext
    calls=[]
    p=OriginalP1Projector.__new__(OriginalP1Projector)
    p.batch_size=1
    p.backend=SimpleNamespace(model=SimpleNamespace(eval=lambda:None),processor=object(),
                              _activate=lambda **kw:calls.append(kw))
    p.subject_function=lambda m,processor,img: {'raw':'original subject'}
    p.candidate_function=lambda m,processor,img,W,H,original,ratio: {'composition_raw':f'{W}x{H}'}
    image=np.zeros((32,64,3),dtype=np.uint8)
    assert p.generate([image],'unused')==['original subject']
    assert p.generate([image],'unused',[16,9])==['64x32']
    assert calls==[dict(scale=False,ratio=[9,16]),dict(scale=False,ratio=[16,9])]


def test_atomic_receipts_and_file_hash(tmp_path):
    path=tmp_path/'nested'/'receipt.json'
    atomic_json(path,{'n':3})
    assert len(file_sha(path))==64
    assert not path.with_suffix('.tmp').exists()


def test_user_review_does_not_create_permission_or_override_prohibition():
    datasets={'GNMC':{'competition_use_authorized':None}}
    config={'data':{'authorization_review':{'owner':'user',
            'instruction':'user_handles_dataset_authorization',
            'agent_scope':'technical_readiness_only','datasets':['GNMC']}}}
    assert not dataset_use_review_ready('GNMC',datasets,{})
    assert dataset_use_review_ready('GNMC',datasets,config)
    assert datasets['GNMC']['competition_use_authorized'] is None
    datasets['GNMC']['competition_use_authorized']=False
    assert not dataset_use_review_ready('GNMC',datasets,config)


def test_live_media_inventory_requires_original_hashes_and_all_sources(tmp_path):
    import hashlib,json
    root=tmp_path/'videos';root.mkdir();(root/'one.mp4').write_bytes(b'original')
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'video_sha256':{'one':hashlib.sha256(b'original').hexdigest(),
                                                    'two':hashlib.sha256(b'missing').hexdigest()}}))
    result=audit_live_media(manifest,root)
    assert result['verified_sources']==1 and result['missing_sources']==['two']
    (root/'two.mp4').write_bytes(b'wrong bytes')
    result=audit_live_media(manifest,root)
    assert result['hash_mismatch_sources']==['two']
    assert result['status']=='incomplete'


def test_identical_val_media_quarantines_all_train_views_and_groups_train_aliases():
    rows=[dict(dataset_source='GNMC',source_id=s,image_sha256=h,split=split,target_ratio=ratio)
          for s,h,split in [('alias','same','train'),('heldout','same','val'),('other','ok','train')]
          for ratio in ([9,16],[16,9])]
    result,audit=isolate(rows)
    assert audit['removed_views']==2
    assert all(r['source_id']!='alias' for r in result)
    assert all(r['independent_source_id'].startswith('image-sha256:') for r in result)
    assert len(rows)==6
