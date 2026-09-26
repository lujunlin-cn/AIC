from scripts.audit_qvh_source_split import original_source, audit


def test_keep_youtube_underscores_and_strip_only_clip_times():
    assert original_source('_dL1_68nwtw_30.0_180.0') == '_dL1_68nwtw'
    assert original_source('_dL1_68nwtw') == '_dL1_68nwtw'


def test_different_clips_same_source_are_purged_from_train():
    rows = [{'video_id': 'a', 'source_id': 'youtube_0.0_150.0', 'split': 'train'},
            {'video_id': 'b', 'source_id': 'youtube_150.0_300.0', 'split': 'val'},
            {'video_id': 'c', 'source_id': 'independent_0_150', 'split': 'train'}]
    result = audit(rows)
    assert result['shared_sources'] == ['youtube']
    assert result['removed_train_video_ids'] == ['a']
    assert result['purged_train_count'] == 1
