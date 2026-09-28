import json

import pytest

from aic.qwen_teacher import (Anchor, TeacherSegment, build_anchors,
                              parse_segments, segments_to_frames)
from aic.video import FrameStamp


def _timeline(n=10, step=0.5):
    return [FrameStamp(i, i, i * step) for i in range(n)]


def _anchors(n=5):
    return [Anchor(i, i, i * 1.0) for i in range(n)]  # 1 fps over 5 s


def test_parse_segments_ok():
    text = 'prefix {"segments":[{"start_id":1,"end_id":3,"importance":0.9,"reason":"goal"}],"summary":"x"} tail'
    segs, status = parse_segments(text, 5)
    assert status == "ok" and len(segs) == 1
    assert segs[0].start_id == 1 and segs[0].end_id == 3 and segs[0].importance == 0.9


def test_parse_segments_repaired_and_invalid():
    # out-of-range ids dropped -> repaired
    segs, status = parse_segments(
        '{"segments":[{"start_id":0,"end_id":2},{"start_id":9,"end_id":12}]}', 5)
    assert status == "repaired" and [ (s.start_id,s.end_id) for s in segs] == [(0,2)]
    # no json -> failed
    segs, status = parse_segments("no json here", 5)
    assert status == "failed" and segs == []
    # empty list -> empty
    segs, status = parse_segments('{"segments":[]}', 5)
    assert status == "empty" and segs == []


def test_segments_to_frames_coverage():
    timeline = _timeline(10, 0.5)   # times 0.0 .. 4.5
    anchors = _anchors(5)           # times 0,1,2,3,4
    segs = [TeacherSegment(1, 2, 0.8)]
    frames = segments_to_frames(segs, anchors, timeline)
    # covers anchor times [1.0, 3.0]: frames at t=1.0..3.0 -> idx 2,3,4,5,6
    assert frames == [2, 3, 4, 5, 6]


def test_segments_to_frames_last_segment_to_end():
    timeline = _timeline(10, 0.5)
    anchors = _anchors(5)
    segs = [TeacherSegment(4, 4, 0.9)]   # last anchor -> to video end
    frames = segments_to_frames(segs, anchors, timeline)
    assert frames == [8, 9]


def test_segments_to_frames_empty():
    assert segments_to_frames([], _anchors(5), _timeline()) == []


def test_merge_block_replies_keeps_segments_across_blocks():
    from aic.qwen_teacher import _merge_block_replies
    replies = ['{"segments":[{"start_id":1,"end_id":2}]}', "garbage",
               '{"segments":[{"start_id":0,"end_id":0}]}']
    segs, status, raw = _merge_block_replies(replies, 5)
    assert [(s.start_id, s.end_id) for s in segs] == [(0, 0), (1, 2)]
    assert status == "failed" and raw.count("\n") == 2
    segs, status, _ = _merge_block_replies(['{"segments":[]}'], 5)
    assert segs == [] and status == "empty"
