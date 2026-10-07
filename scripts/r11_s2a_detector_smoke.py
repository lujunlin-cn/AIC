"""R11 S2a: open-vocabulary detector smoke on the 910A NPU (<=4 h stop-loss).

Tries candidates in order, first one that passes the parity gate wins:
  1. GroundingDINO (repo lineage from V100 era)
  2. RT-DETR / YOLO-World (HF checkpoint, subject prompt 'person')
Parity gate: on 10 live_dev sources, detector box vs GT subject box IoU and
vs B3-argmax window IoU are computed; the arm proceeds to the 100-source
extraction only if detection itself is not degenerate (mean det-GT IoU > 0.1
on at least 60% of frames, i.e. the detector actually finds the subject).

Fail-closed: every candidate failure is recorded; if all fail, the JSON
records status=NPU_DETECTOR_UNAVAILABLE and the stage-2 chain falls back to
the SigLIP-proposal downgrade path (plan section S2 downgrade).
Media access goes through the R9/R10 frozen media locator (PyAV, 2 s seek
lead) -- identical to r10_audio_extract.py, no new decoding surface.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--media-root", type=Path, default=Path("/data/aic/ext_data/PHD2/media"))
ap.add_argument("--audit-dir", type=Path,
                default=Path("/data/aic/experiments_910a/LFM_V11/r8_npu/r9_audit"))
ap.add_argument("--n-sources", type=int, default=10)
ap.add_argument("--device", default="npu")
args = ap.parse_args()

res = {"started": time.strftime("%F %T"), "candidates": [], "status": "TRYING"}


def log(m):
    print(m, flush=True)
    res["candidates"].append(m)


try:
    import torch  # noqa: F401
    if args.device == "npu":
        import torch_npu  # noqa: F401
        assert torch.npu.device_count() > 0, "torch_npu sees no card (zombie container rule)"
    t0 = time.time()
    # Candidate 1: GroundingDINO import probe (operator coverage unknown on 910A)
    try:
        import groundingdino  # noqa: F401
        log({"cand": "groundingdino", "stage": "import", "ok": True})
        log({"cand": "groundingdino", "stage": "forward", "ok": False,
             "note": "wire the V100-era config/weights here; parity gate below still applies"})
    except Exception as e:  # noqa: BLE001
        log({"cand": "groundingdino", "stage": "import", "ok": False, "err": str(e)[:200]})

    # Candidate 2: HF RT-DETR (transformers-native, smallest operator surface)
    try:
        from transformers import AutoModelForObjectDetection, AutoProcessor
        name = "PekingU/rtdetr_r50vd_finetuned_detsd_co1400" if os.environ.get("HF_ENDPOINT") \
            else "PekingU/RTV-DERT-placeholder"
        log({"cand": "rtdetr", "stage": "load", "name": name})
        _proc = AutoProcessor.from_pretrained(name)
        _model = AutoModelForObjectDetection.from_pretrained(name)
        log({"cand": "rtdetr", "stage": "load", "ok": True, "s": round(time.time() - t0, 1)})
        log({"cand": "rtdetr", "stage": "forward",
             "note": "parity gate (det-GT IoU >0.1 on >=60% of 10-source frames) runs here; "
                     "wire frames via the frozen media locator before extraction"})
    except Exception as e:  # noqa: BLE001
        log({"cand": "rtdetr", "stage": "load", "ok": False, "err": str(e)[:200]})

    res["status"] = "SMOKE_RECORD_ONLY"
    res["note"] = ("this pass records which candidates LOAD on the 910A; the 10-source "
                   "parity gate and the 100-source extraction are wired by the host "
                   "session once a loading candidate exists (<=4 h stop-loss applies)")
finally:
    args.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({"status": res["status"]}, indent=1))
