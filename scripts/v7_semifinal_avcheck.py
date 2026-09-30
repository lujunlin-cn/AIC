"""Probe decoder compatibility of the lfm_venv PyAV against every official video.

Semifinal readiness check: the keyframe/extraction step of the student chain must
decode the intake videos on the 910A (V100 is unavailable).  Probes first frame,
a middle frame and the last frame of each video.
"""
import json, sys, time
from pathlib import Path

import av

index = [json.loads(l) for l in open("/data/aic/official_test_20260926/intake/index.enriched.jsonl")]
ok = bad = 0
fails = []
t0 = time.time()
for r in index:
    vid = r["video_id"]
    p = Path(r.get("video_path") or f"/data/aic/official_test_20260926/intake/videos/{vid}.mp4")
    if not p.exists():
        alt = sorted(Path("/data/aic/official_test_20260926/intake").rglob(f"{vid}.*"))
        p = alt[0] if alt else p
    try:
        c = av.open(str(p))
        st = c.streams.video[0]
        n = r["frame_count"]
        for tgt in (0, n // 2, n - 1):
            c.seek(int(tgt / float(st.average_rate)))
            got = False
            for fr in c.decode(video=0):
                fr.to_ndarray(format="rgb24")
                got = True
                break
            if not got:
                raise RuntimeError(f"no frame at {tgt}")
        c.close()
        ok += 1
    except Exception as e:
        bad += 1
        fails.append({"vid": vid, "err": str(e)[:160]})
        print("FAIL", vid, str(e)[:120], flush=True)
print("SUMMARY", json.dumps({"ok": ok, "bad": bad, "n": len(index), "wall_s": round(time.time() - t0, 1),
                            "fails": fails[:20]}, ensure_ascii=False), flush=True)
