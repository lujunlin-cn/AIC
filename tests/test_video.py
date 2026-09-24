import subprocess

import pytest

from aic.video import expand_scores, frame_timeline, iter_sampled_frames, probe_video


def make_video(path, vf=None):
    command = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
               "testsrc2=size=96x64:rate=10:duration=2"]
    if vf:
        command += ["-vf", vf, "-vsync", "vfr"]
    subprocess.run(command + ["-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", str(path)], check=True)


def test_cfr_original_indices_and_dense_expansion(tmp_path):
    path = tmp_path / "cfr.mp4"
    make_video(path)
    info = probe_video(path)
    timeline = frame_timeline(path)
    sampled = list(iter_sampled_frames(path, 2))
    assert info.frame_count == len(timeline) == 20
    assert [s[0] for s in sampled] == [0, 5, 10, 15]
    assert sampled[0][2].shape == (224, 224, 3)
    assert expand_scores(timeline, [0, 1.5], [1, 1], .5) == list(range(20))
    assert expand_scores(timeline, [0, 1.5], [0, 0], .5) == []


def test_vfr_uses_actual_pts(tmp_path):
    path = tmp_path / "vfr.mp4"
    make_video(path, "select='if(lt(n,10),1,not(mod(n,3)))'")
    info = probe_video(path)
    timeline = frame_timeline(path)
    assert len(timeline) == info.frame_count == 13
    assert timeline[10].time_seconds == pytest.approx(1.2)
    assert timeline[-1].index == 12
    assert timeline[-1].time_seconds == pytest.approx(1.8)
    assert expand_scores(timeline, [0, 1.8], [1, 1], .5) == list(range(13))


def test_rotation_is_recorded_not_silently_applied(monkeypatch, tmp_path):
    # Some MP4 muxers discard the legacy rotate tag. Exercise the parser with
    # the ffprobe shape used by containers that retain it.
    import json
    import aic.video as video

    class Result:
        stdout = json.dumps({"streams": [{"codec_type": "video", "width": 96,
            "height": 64, "nb_read_frames": "20", "avg_frame_rate": "10/1",
            "time_base": "1/10", "duration": "2", "start_time": "0",
            "tags": {"rotate": "90"}}]})

    monkeypatch.setattr(video.subprocess, "run", lambda *args, **kwargs: Result())
    info = probe_video(tmp_path / "tagged.mp4")
    assert (info.width, info.height) == (96, 64)
    assert abs(info.rotation) == 90
    assert info.coordinate_convention == "coded_pixels_no_autorotate"


def test_bad_video_fails_explicitly(tmp_path):
    path = tmp_path / "bad.mp4"
    path.write_bytes(b"not a video")
    with pytest.raises(subprocess.CalledProcessError):
        probe_video(path)
