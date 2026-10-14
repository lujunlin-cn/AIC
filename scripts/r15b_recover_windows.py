#!/usr/bin/env python3
"""从已落盘的「高光窗居中」段反解出每个视频的高光窗绝对时间。

克隆机上没有 PHD2 的 annotations/selections 与 .info.json，但
sft_pool_phd2_2k.jsonl 记录了居中段的 (seg_t0, seg_t1, windows_rel)，
足以还原窗的绝对位置：
    w_abs = seg_t0 + rel * (seg_t1 - seg_t0)

顺带输出「居中偏差」诊断量 seg_center_off，验证 93.5% 的居中假说。
用法: r15b_recover_windows.py <pool.jsonl> <out.jsonl>
"""
import json
import sys
from collections import defaultdict

pool, out = sys.argv[1], sys.argv[2]
per_vid = defaultdict(list)
diag = []
with open(pool) as f:
    for line in f:
        r = json.loads(line)
        s0, s1 = float(r["seg_t0"]), float(r["seg_t1"])
        L = max(1e-9, s1 - s0)
        wins = [(s0 + a * L, s0 + b * L) for a, b in r.get("windows_rel", [])
                if b > a]
        if not wins:
            continue
        vid = r["video_id"]
        per_vid[vid].extend(wins)
        c = (s0 + s1) / 2.0
        off = min(abs(c - (a + b) / 2.0) for a, b in wins) / L
        diag.append(off)

n_seg = len(diag)
frac = sum(1 for d in diag if d < 0.05) / max(1, n_seg)
print(f"segments={n_seg}  videos={len(per_vid)}")
print(f"seg_center_off < 0.05: {frac:.4f}   (diagnosis agent measured 0.935)")

with open(out, "w") as f:
    for vid, wins in sorted(per_vid.items()):
        wins.sort()
        merged = []
        for a, b in wins:
            if merged and a <= merged[-1][1] + 2.0:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        f.write(json.dumps({"video_id": vid,
                            "windows_abs": [[round(a, 3), round(b, 3)]
                                            for a, b in merged]}) + "\n")
print(f"WROTE {out}")
