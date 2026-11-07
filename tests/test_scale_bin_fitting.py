import json
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

from scripts.fit_scale_bins import fit_gnmc_train_bins


def _jpeg_bytes(size):
    from io import BytesIO

    buffer = BytesIO()
    Image.new("RGB", size, color=(120, 80, 20)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_gnmc_bins_use_train_native_labels_only_and_transpose_geometry(tmp_path: Path):
    archive_path = tmp_path / "gnmc.zip"
    train = [
        {"filename": "train-1.jpeg", "crop_bboxes": {"16:9": [0.05, 0.1, 0.85, 0.9]}},
        {"filename": "train-2.jpeg", "crop_bboxes": {"16:9": [0.0, 0.1, 1.0, 0.9]}},
        {"filename": "train-3.jpeg", "crop_bboxes": {"16:9": [0.2, 0.1, 0.8, 0.9]}},
    ]
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("GNMC/json/train.json", json.dumps(train))
        archive.writestr("GNMC/json/validation.json", "invalid validation payload")
        archive.writestr("GNMC/json/test.json", "invalid test payload")
        for row in train:
            archive.writestr(f"GNMC/train/{row['filename']}", _jpeg_bytes((640, 360)))

    result = fit_gnmc_train_bins(archive_path, count=4)

    assert result["fit_split"] == "train"
    assert result["fit_sources"]["source_count"] == 3
    assert result["fit_sources"]["native_geometry"] == "16:9"
    assert result["fit_sources"]["transpose_geometry"] == "9:16"
    assert result["validation_or_test_labels_read"] is False
    assert result["p1_centers_used"] is False
    assert len(result["values"]) <= 4
    assert result["values"] == sorted(set(result["values"]))
    assert result["values"][-1] == 1.0
