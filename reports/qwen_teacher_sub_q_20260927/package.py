"""Merge original Qwen 174 run with vLLM rerun of 99/104, validate, zip."""
import glob, hashlib, json, os, tempfile, zipfile
from pathlib import Path
from aic.contract import load_index, load_jsonl, write_submission
from scripts.independent_submission_check import check
INDEX = "/data/aic/official_test_20260926/intake/index.enriched.jsonl"
BASE = "/data/aic/experiments/qwen_official_174/predictions.jsonl"
RERUN = "/data/aic/experiments/qwen_official_174_rerun_vllm/shards"
OUT = Path(os.environ["O"]); SIZE_MB = 350.5
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
index = load_index(INDEX)
rows = {r["video_id"]: r for r in load_jsonl(BASE)}
replaced = {}
for vid in ("99", "104"):
    new = load_jsonl(f"{RERUN}/{vid}.json")[0]
    replaced[vid] = {"before": len(rows[vid]["predictions"]), "after": len(new["predictions"])}
    rows[vid] = new
assert set(rows) == set(index)
ordered = [rows[v] for v in index]
pred = OUT / "predictions.jsonl"
validation = write_submission(pred, ordered, index, stage="final",
                              actual_model_size_mb=SIZE_MB).to_dict()
assert validation["valid"], validation
independent = check(INDEX, pred)
zp = OUT / "upload.zip"
with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_DEFLATED) as z:
    z.write(pred, "predictions.jsonl")
with tempfile.TemporaryDirectory() as t:
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ["predictions.jsonl"]; z.extractall(t)
    unzip = check(INDEX, Path(t) / "predictions.jsonl")
    assert sha(Path(t) / "predictions.jsonl") == sha(pred)
qwen = sorted(glob.glob("/data/aic/pretrained/qwen3_vl_32b_instruct/*.safetensors"))
yunet = "/data/aic/pretrained/yunet/face_detection_yunet_2023mar.onnx"
inventory = [{"path": p, "bytes": os.path.getsize(p)} for p in qwen + [yunet]]
true_bytes = sum(w["bytes"] for w in inventory)
manifest = {
  "submission_id": "SUB_Q_QWEN3VL32B_TEACHER_V1",
  "status": "READY_TO_UPLOAD_TEACHER_EVAL_ONLY", "uploaded": False,
  "purpose": "teacher score probe; not a rule-compliant final submission",
  "model_size_mb_declared": SIZE_MB, "model_size_mb_is_placeholder": True,
  "true_weight_bytes": true_bytes, "true_model_size_mb": true_bytes / 1e6,
  "rule_limit_mb": 9216, "exceeds_rule_limit": true_bytes / 1e6 > 9216,
  "sources": {"base": BASE, "base_sha256": sha(BASE), "rerun_dir": RERUN,
              "rerun_backend": "vllm 0.13.0 TP=4 fp16 greedy, GPUs 4-7",
              "replaced_videos": replaced},
  "index_sha256": sha(INDEX), "prompt_version": "qwen_temporal_v1",
  "video_count": len(ordered),
  "prediction_count": sum(len(r["predictions"]) for r in ordered),
  "empty_videos": [r["video_id"] for r in ordered if not r["predictions"]],
  "validator": validation, "independent": independent, "unzip_independent": unzip,
  "predictions_sha256": sha(pred), "zip_sha256": sha(zp),
  "zip_bytes": zp.stat().st_size, "official_score": None}
(OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
(OUT / "weight_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
(OUT / "upload.zip.sha256").write_text(manifest["zip_sha256"] + "  upload.zip\n")
print(json.dumps({k: manifest[k] for k in ("status","video_count","prediction_count","empty_videos","true_model_size_mb","zip_sha256","zip_bytes")}, ensure_ascii=False, indent=1))
print(validation, independent, unzip, replaced)
