"""RetargetVid raw-byte, display-frame, coordinate and exposure regressions."""
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aicext import retargetvid_audit as audit
from aicext.adapters.spatial import DenseCropAdapter, ltrb_to_xywh
from aicext.common import read_jsonl, write_jsonl
from aicext.media import load_timeline, read_frames, save_timeline
from aicext.retargetvid_archive import RETARGET_IDS, digest, ensure_selected_media, existing_avi


def test_halfopen_roundtrip_raw_unmodified_and_no_clipping():
    raw = np.tile(np.array([[-2, 0, 118, 360], [20, 0, 140, 360]], np.float32), (6, 1, 1))
    before = raw.copy()
    boxes, result = audit.coordinate_one(raw, 640, 360, "1-3")
    assert np.array_equal(raw, before)
    assert boxes[0, 0].tolist() == [-2, 0, 120, 360]
    assert result["outside_image_exact"] == 6
    assert result["raw_negative_left_top"] == 6
    assert result["roundtrip_ltrb_maxabs"] == 0
    assert result["default_gt_boundary"] == "none_raw_unclipped"
    assert ltrb_to_xywh(np.array([[0, 0, 1, 1]], np.float32)).tolist() == [[0, 0, 1, 1]]


@pytest.mark.parametrize("offsets,expected", [([0]*6, 0), ([0, 0, 0, 400, 400, 400], 1), ([0]*5+[400], 0)])
def test_multicluster_proxy_and_singleton_outlier(offsets, expected):
    boxes = np.array([[[x, 0, 120, 360]]*3 for x in offsets], dtype=float)
    result = audit.agreement_one(boxes, 640, 360, "1-3")
    assert result["multimodal_proxy_fraction"] == expected
    if len(set(offsets)) == 1:
        assert result["pairwise_IoU"]["mean"] == 1
        assert result["pairwise_center_L2_pixels"]["mean"] == 0
        assert result["scale_across_raters_std"]["mean"] == 0


def test_source_macro_does_not_weight_by_frame_or_pair_count():
    units = []
    for ratio in audit.RATIOS:
        for n, frames, offsets in [(1, 1, [0]*6), (2, 100, [0, 0, 0, 400, 400, 400])]:
            boxes = np.array([[[x, 0, 120, 360]]*frames for x in offsets], dtype=float)
            units.append(dict(video_id=str(n), ratio=ratio, **audit.agreement_one(boxes, 640, 360, ratio)))
    result = audit.aggregate_agreement(units)
    for ratio in audit.RATIOS:
        rows = [r for r in units if r["ratio"] == ratio]
        q = result["by_ratio"][ratio]["source_macro_pairwise_IoU"]
        assert q["sources"] == 2
        assert q["mean"] == pytest.approx(np.mean([r["pairwise_IoU"]["mean"] for r in rows]))
        assert q["mean"] != pytest.approx(np.average([r["pairwise_IoU"]["mean"] for r in rows], weights=[1,100]))
        assert q["bootstrap_ci95"] is not None


def test_exposure_applies_to_both_datasets_all_ratios_and_raters():
    m = dict(path="original/001.AVI", file_size=100, sha256="a"*64, width=640, height=360,
             fps=30, frame_count=3, rotation=0, codec="mpeg4", has_audio=False)
    groups = {"dhf1k:001": [{"role":"selection", "experiment":"existing", "split_name":"exposed"}]}
    dhf, rv = audit.media_rows({1:m}, groups)
    for row in dhf + rv:
        assert row["group_id"] == "dhf1k:001"
        assert row["media_path"] == m["path"]
        assert row["project_split"] == "holdout_dev_exposed"
        assert row["release_split"] == "exposed_eval"
        assert row["training_eligible"] is False
        assert row["fresh_validation_eligible"] is False
    with pytest.raises(ValueError, match="fresh/train"):
        audit.media_rows({1:m}, {"dhf1k:001":[]})


def test_original_release_and_ledger_identity_guard(tmp_path):
    root = Path(__file__).resolve().parents[1]
    paths, before, groups = audit.protected_sources(root)
    assert len(groups) == 200
    for key,path in paths.items():
        target = tmp_path/path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    ledger = tmp_path/"configs/holdout_exposure_v1.json"
    ledger.write_text(ledger.read_text()+" ")
    with pytest.raises(ValueError, match="immutable"):
        audit.protected_sources(tmp_path)
    assert {k:digest(p) for k,p in paths.items()} == before


def make_archives(root):
    (root/"annotations").mkdir()
    payload = b"0,0,120,360\r\n1,0,121,360\r\n"
    for ann in audit.RATERS:
        with ZipFile(root/"annotations"/f"annotator_{ann}.zip", "w") as z:
            for n in RETARGET_IDS:
                for ratio in audit.RATIOS:
                    z.writestr(f"annotator_{ann}/{n:03d}_{ratio}.txt", payload)
    return payload


def test_annotations_keep_original_bytes_all_2400_and_refuse_overwrite(tmp_path):
    payload = make_archives(tmp_path)
    files, hashes = audit.prepare_annotations(tmp_path)
    assert len(files) == 2400 and len(hashes) == 6
    assert all(p.read_bytes() == payload for p in files.values())
    rows, original_hashes = audit.read_annotations(files)
    assert len(rows) == len(original_hashes) == 2400
    assert all(len(a) == 2 for a in rows.values())
    chosen = files[1,"1-3",1]
    chosen.write_bytes(payload+b"0,0,1,1\n")
    changed_hash = digest(chosen)
    with pytest.raises(ValueError, match="never overwrite"):
        audit.prepare_annotations(tmp_path)
    assert digest(chosen) == changed_hash


def test_missing_annotation_file_fails_closed(tmp_path):
    make_archives(tmp_path)
    with ZipFile(tmp_path/"annotations/annotator_1.zip", "w") as z:
        z.writestr("annotator_1/001_1-3.txt", "0,0,120,360\n")
    with pytest.raises(ValueError, match="incomplete/duplicate"):
        audit.prepare_annotations(tmp_path)


def test_media_reused_without_archive_library_or_extra_copies(tmp_path):
    (tmp_path/"video").mkdir()
    for n in RETARGET_IDS:
        (tmp_path/"video"/f"{n:03d}.AVI").write_bytes(str(n).encode())
    before = sorted(str(p) for p in tmp_path.rglob("*.AVI"))
    paths = ensure_selected_media(tmp_path, tmp_path/"receipt.json")
    assert len(paths) == 200
    assert sorted(str(p) for p in tmp_path.rglob("*.AVI")) == before
    assert json.loads((tmp_path/"receipt.json").read_text())["action"] == "reuse_existing_AVI_no_archive_unpack"


def test_conflicting_video_duplicates_are_not_silently_chosen(tmp_path):
    (tmp_path/"video").mkdir()
    (tmp_path/"raw").mkdir()
    (tmp_path/"video/001.AVI").write_bytes(b"original")
    (tmp_path/"raw/001.avi").write_bytes(b"different")
    with pytest.raises(ValueError, match="conflicting duplicate"):
        existing_avi(tmp_path)


def test_working_registry_merge_preserves_unrelated_records_and_backup(tmp_path):
    p = tmp_path/"registry/DHF1K/processed/media.jsonl"
    write_jsonl(p,[{"uid":"other:1","notes":"keep"}, {"uid":"dhf1k:001","notes":"old"}])
    before = audit.snapshot_working_registry(tmp_path/"registry", tmp_path/"audit")
    audit.merge_records(p,[{"uid":"dhf1k:001","media_path":"reuse.AVI"}])
    records = {r["uid"]:r for r in read_jsonl(p)}
    assert records["other:1"]["notes"] == "keep"
    assert records["dhf1k:001"]["notes"] == "old"
    backup = next((tmp_path/"audit/registry_before/DHF1K").iterdir())
    assert digest(backup) == before[str(p)]


def make_video(path, count=5):
    import av
    with av.open(str(path), "w") as c:
        stream = c.add_stream("ffv1", rate=25)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i in range(count):
            rgb = np.full((48,64,3), i*30, np.uint8)
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1,25)
            for packet in stream.encode(frame): c.mux(packet)
        for packet in stream.encode(): c.mux(packet)


def test_real_adapter_rgb_and_eof_observer_share_single_decode(tmp_path, monkeypatch):
    import av
    path = tmp_path/"001.AVI"
    make_video(path)
    m = dict(uid="retargetvid:001", group_id="dhf1k:001", media_kind="video_file", media_path=str(path),
             width=64,height=48,fps=25,frame_count=5,project_split="holdout_dev_exposed",download_status="verified")
    anns = []
    for ratio in audit.RATIOS:
        for ann in audit.RATERS:
            p = tmp_path/f"{ratio}_a{ann}.txt"
            np.savetxt(p,np.tile([0,0,16,48],(5,1)),delimiter=",",fmt="%d")
            anns.append(dict(uid=p.stem,media_uid=m["uid"],annotation_path=str(p),annotation_type="crop_box_dense",variant=ratio,annotator=f"annotator_{ann}"))
    write_jsonl(tmp_path/"RetargetVid/processed/media.jsonl",[m])
    write_jsonl(tmp_path/"RetargetVid/processed/annotations.jsonl",anns)
    original_open = av.open
    calls = []
    def observed_open(*args, **kwargs):
        calls.append(str(args[0]))
        return original_open(*args, **kwargs)
    monkeypatch.setattr(av,"open",observed_open)
    trace = audit.DecodeTrace(5)
    ds = DenseCropAdapter(root=tmp_path,ratios=("1-3",),fps=None,max_frames=1,decode=True,size=None,
                          decode_observer=trace,decode_through_eof=True)
    item = ds[0]
    result = trace.finish()
    assert len(calls) == 1
    assert item["frame_idx"].tolist() == [0]
    assert item["frames"].shape == (1,48,64,3)
    assert result["frame_count"] == 5
    assert set(result["frame_probes"]) == {"0","2","4"}
    assert result["non_increasing"] == 0 and result["first_time_zero"]
    assert result["time_s"].tolist() == pytest.approx([0,.04,.08,.12,.16])
    save_timeline(result,tmp_path/"clock.npz")
    assert np.array_equal(load_timeline(tmp_path/"clock.npz")["pts"],result["pts"])
    assert len(DenseCropAdapter(root=tmp_path,split="train")) == 0
    assert len(DenseCropAdapter(root=tmp_path,split=["dev","confirmation","holdout_confirm_reserved"])) == 0
    assert len(DenseCropAdapter(root=tmp_path,split="holdout_dev_exposed")) == 2


def test_legacy_early_stop_and_full_eof_keep_selected_pixels_identical(tmp_path):
    path = tmp_path/"v.AVI"
    make_video(path)
    short = []
    all_frames = []
    a = read_frames(path,[0],observer=lambda i,f,t:short.append(i))
    b = read_frames(path,[0],observer=lambda i,f,t:all_frames.append(i),through_eof=True)
    assert short == [0] and all_frames == [0,1,2,3,4]
    assert np.array_equal(a[0],b[0])
    assert np.array_equal(read_frames(path,[0])[0],a[0])


def test_decode_trace_rejects_missing_or_reordered_pts():
    class F:
        pts = None
        width,height = 64,48
        def to_ndarray(self,format): return np.zeros((48,64,3),np.uint8)
    trace = audit.DecodeTrace(2)
    trace(0,F(),Fraction(1,25))
    with pytest.raises(ValueError,match="missing real PTS"):
        trace.finish()
    with pytest.raises(ValueError,match="reordered/skipped"):
        trace(2,F(),Fraction(1,25))


def test_annotation_only_audit_reports_mismatch_without_truncation(tmp_path):
    for ann, count in [(1,2),(2,3)]:
        p=tmp_path/"annotations_all"/f"annotator_{ann}"/"001_1-3.txt"
        p.parent.mkdir(parents=True)
        p.write_text("0,0,120,360\n"*count)
    output=tmp_path/"audit.json"
    script=Path(__file__).resolve().parents[2]/"scripts/audit_retargetvid.py"
    subprocess.run([sys.executable,str(script),"--root",str(tmp_path),"--output",str(output)],check=True,capture_output=True)
    row=json.loads(output.read_text())["files"][0]
    assert row["frame_counts"] == [2,3]
    assert row["consistent_frame_count"] is False


def test_full_audit_preserves_each_file_alignment_and_excludes_failed_unit():
    raw={(1,"1-3",a):np.zeros((3 if a!=2 else 2,4)) for a in audit.RATERS}
    rows=audit.alignment_rows(raw,1,"1-3",dict(frame_count=3,decode_verified=True))
    assert [r['aligned'] for r in rows] == [True,False,True,True,True,True]
    assert [r['annotation_rows'] for r in rows] == [3,2,3,3,3,3]
    assert not all(r['aligned'] for r in rows)
    rows=audit.alignment_rows(raw,1,"1-3",dict(frame_count=3,decode_verified=False))
    assert not any(r['aligned'] for r in rows)
    assert all(r['decoded_frames'] is None for r in rows)
