import json
import subprocess

from aic.inference import run_inference
from aic.contract import load_jsonl


def _video(path):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=size=80x40:rate=5:duration=1", "-c:v", "libx264",
                    "-threads", "1", "-pix_fmt", "yuv420p", str(path)], check=True)


def test_dummy_raw_video_to_valid_jsonl(tmp_path):
    video = tmp_path / "one.mp4"
    _video(video)
    index = tmp_path / "index.jsonl"
    index.write_text(json.dumps({"video_id": "one", "video_path": str(video),
                                 "width": 80, "height": 40, "frame_count": 5,
                                 "targetRatioWH": [9, 16]}) + "\n")
    output = tmp_path / "submission.jsonl"
    result = run_inference(index, output, dummy=True, stage="preliminary")
    assert result["validation"]["valid"]
    row = load_jsonl(output)[0]
    assert len(row["predictions"]) == 5
    assert row["predictions"][0]["frame"] == 0
    assert row["targetRatioWH"] == [9, 16]
