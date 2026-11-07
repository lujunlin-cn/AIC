import pytest

from aic.scale_data import attach_frozen_positions, gnmc_training_row, live_youtube_crop_rows


def test_gnmc_native_and_transposed_geometry_are_consistent():
    row = {"filename": "image-1.jpeg", "crop_bboxes": {"16:9": [0.1, 0.1, 0.9, 0.9]}}
    native = gnmc_training_row(row, split="train", image_width=640, image_height=360,
                               target_ratio=(16, 9))
    assert native["gt_crop_xywh"] == pytest.approx([64, 36, 512, 288])
    transposed = gnmc_training_row(row, split="train", image_width=640, image_height=360,
                                   target_ratio=(9, 16), transpose=True)
    assert transposed["image_width"] == 360
    assert transposed["image_height"] == 640
    assert transposed["gt_crop_xywh"] == pytest.approx([36, 64, 288, 512])
    assert transposed["transposed_geometry_augmentation"]
    with pytest.raises(ValueError, match="test annotations"):
        gnmc_training_row(row, split="test", image_width=640, image_height=360,
                          target_ratio=(16, 9))


def test_live_requires_audited_source_split_and_parses_sparse_human_boxes(tmp_path):
    csv_path = tmp_path / "live.csv"
    csv_path.write_text(
        "video,num_scenes,frame0,frame6\n"
        'video1,1,"{\'left\': 10, \'right\': 30, \'top\': 5, \'bottom\': 45}",\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source-level"):
        list(live_youtube_crop_rows(csv_path, {}))
    rows = list(live_youtube_crop_rows(csv_path, {"video1": "train"}))
    assert len(rows) == 1
    assert rows[0]["source_id"] == "video1"
    assert rows[0]["frame_id"] == 0
    assert rows[0]["gt_crop_xywh"] == [10, 5, 20, 40]
    with pytest.raises(ValueError, match="absent from source split"):
        list(live_youtube_crop_rows(csv_path, {"other": "train"}))
    dev_rows = list(live_youtube_crop_rows(csv_path, {"video1": "dev"}))
    assert dev_rows[0]["split"] == "val"


def test_frozen_position_join_requires_exact_split_and_source_frame_identity():
    label = {"source_id": "video-1", "image_id": "video-1:0", "dataset_source": "LIVE_YT_VC",
             "split": "train", "target_ratio": [9, 16], "gt_crop_xywh": [1, 2, 3, 4]}
    center = {"source_id": "video-1", "image_id": "video-1:0", "dataset_source": "LIVE_YT_VC",
              "split": "train", "target_ratio": [9, 16], "center_xy": [20, 30],
              "image_width": 640, "image_height": 360}
    joined = attach_frozen_positions([label], [center])
    assert joined[0]["center_xy"] == [20, 30]
    assert joined[0]["position_source"] == "P1_frozen"
    with pytest.raises(ValueError, match="split mismatch"):
        attach_frozen_positions([label], [dict(center, split="val")])
