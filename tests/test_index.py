import json
import subprocess

from aic.index import enrich_index, load_compact_index


def test_compact_index_is_enriched_from_decoded_video(tmp_path):
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    video = video_dir / "0.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=size=64x32:rate=4:duration=1", "-c:v", "libx264",
                    "-threads", "1", "-pix_fmt", "yuv420p", str(video)], check=True)
    compact = tmp_path / "test_index.json"
    compact.write_text(json.dumps([{"video_id": "0", "targetRatioWH": [16, 9]}]))
    output = tmp_path / "index.jsonl"
    result = enrich_index(compact, video_dir, output)
    assert result["videos"] == 1
    row = json.loads(output.read_text().splitlines()[0])
    assert row["frame_count"] == 4
    assert row["width"] == 64 and row["height"] == 32
    assert load_compact_index(compact)[0]["video_id"] == "0"
