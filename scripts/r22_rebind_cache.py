#!/usr/bin/env python3
"""R22 阶段1 第 3 项：缓存重绑清单。

## 为什么需要这个脚本

R21 审计判定本地读数失效的直接原因之一是**缓存合同破裂**：标签按行号索引到
另一个池的特征缓存上。本轮在 `vpb_artifacts/feat` 上把该破裂**完整复原**：

  feat/shard_{0,1,2}of4.npz  02:14 写，各 627 行        (= rows[i::4]，i=0,1,2)
  feat/shard_3of4.npz        13:35 写，819 行 = 627 + 192 追加行

`627 × 4 = 2508` = `sft_data_v2.jsonl` 行数 ⇒ **训练时缓存是干净的 2508 行**。
训练脚本 `train_vp_head.py:56` 有 `assert len(vids) == len(rows)`，被这条断言保护。
但评估脚本 `e04_compare.py:118` / `resid_e03.py:85` 只有
`concat → argsort(order)`，**既无长度断言也无截断**，于是 2700 行缓存被静默
喂给 `hold_idx`（索引域 <2508），行号错配且不报错。
—— 断言只存在于训练路径，评估路径被丢掉了。这就是那类翻车的**具体机制**。

本脚本按 `(sft_row, shard, row_in_shard)` 把每一行绑到它的来源上，
禁止跨池复用行号；并把训练时的干净视图**冻结**为可复算的配方。

## 硬约束

- 只读。不训练、不拟合、不出预测。
- `PER=627` 是**冻结前的实测产物**，不是可调超参：它由 `len(rows)/4` 与
  `rows[i::4]` 逐位相等共同钉死，两处都已在清单里留证。

用法：
  python r22_rebind_cache.py --sft <sft_data_v2.jsonl> --feat-dir <feat> \
      --out <dir>
"""
import argparse
import glob
import hashlib
import json
import os
from pathlib import Path

import numpy as np

PER = 627  # = 2508/4；由 len(rows)/4 与 rows[i::4] 逐位相等共同钉死


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(a):
    rows = [json.loads(l) for l in open(a.sft)]
    sft_vids = [r["vid"] for r in rows]
    N = len(rows)

    shard_files = sorted(glob.glob(os.path.join(a.feat_dir, "shard_*of*.npz")))
    report = {
        "scope": "read-only cache rebinding; no training, no fitting, no prediction",
        "sft": str(a.sft), "sft_sha256": sha256(a.sft), "sft_rows": N,
        "feat_dir": str(a.feat_dir),
        "training_time_rows_per_shard": PER,
        "shards": [], "duplicates": [], "bindings": [],
    }

    loaded, vids_all, kept_temporal = [], [], []
    for i, f in enumerate(shard_files):
        z = np.load(f)
        v = z["vid"].tolist()
        n = len(v)
        expected = sft_vids[i::len(shard_files)]
        rec = {
            "shard": os.path.basename(f), "shard_index": i,
            "shard_count": len(shard_files),
            "sha256": sha256(f), "rows_on_disk": n,
            "rows_expected_for_full_sft": len(expected),
            "prefix_matches_rows_i_step_n": bool(v[:len(expected)] == expected),
            "appended_rows": max(0, n - len(expected)),
            "shard_style": f"rows[{i}::len(shards)]  (strided, not contiguous block)",
        }
        if n > len(expected):
            extra = v[len(expected):]
            first = {x: k for k, x in enumerate(v[:len(expected)])}
            same_vals = int(sum(
                1 for k, x in enumerate(extra)
                if x in first and np.array_equal(
                    z["feat_temporal"][len(expected) + k],
                    z["feat_temporal"][first[x]])))
            rec.update(
                appended_sample=extra[:5],
                appended_vid_set_equals_first_block=bool(set(extra) == set(v[:len(expected)])),
                appended_rows_identical_to_first_occurrence=same_vals,
            )
            report["duplicates"].append({
                "shard": os.path.basename(f), "appended_rows": len(extra),
                "distinct_vids": len(set(extra)),
                "identical_to_first_occurrence": same_vals,
                "note": "同 vid 两份特征逐值不同 ⇒ 抽取本身不可复现（见 EXPERIMENTS R22c）",
            })
        report["shards"].append(rec)
        loaded.append(z)
        vids_all.extend(v[:PER])
        kept_temporal.append(z["feat_temporal"][:PER])

    # --- 冻结的训练时视图 ---
    assert len(vids_all) == N, f"冻结视图 {len(vids_all)} != SFT {N}"
    assert len(set(vids_all)) == N, "冻结视图存在重复 vid"
    assert set(vids_all) == set(sft_vids), "冻结视图 vid 集合 != SFT"
    row_of = {v: k for k, v in enumerate(sft_vids)}
    order = np.argsort([row_of[v] for v in vids_all])
    ft = np.concatenate(kept_temporal)[order]
    reordered = [vids_all[i] for i in order]  # list 不能被 ndarray 索引
    aligned = reordered == sft_vids
    report["frozen_training_view"] = {
        "rows": int(ft.shape[0]), "feat_shape": list(ft.shape),
        "recipe": "每个 shard 取前 PER 行 → concat → 按 sft_row 重排",
        "row_aligned_to_sft": bool(aligned),
        "feat_temporal_f64_sha256": hashlib.sha256(
            np.ascontiguousarray(ft.astype(np.float64)).tobytes()).hexdigest(),
    }
    assert aligned, "冻结视图重排后未与 SFT 逐行对齐"

    # --- 逐行绑定：禁止跨池复用行号 ---
    rows_on_disk_cursor = {}
    for i, z in enumerate(loaded):
        rows_on_disk_cursor[os.path.basename(shard_files[i])] = z["vid"].tolist()
    for sft_row, vid in enumerate(sft_vids):
        i = sft_row % len(shard_files)
        k = sft_row // len(shard_files)
        name = os.path.basename(shard_files[i])
        assert rows_on_disk_cursor[name][k] == vid, f"绑定不一致 row={sft_row}"
        report["bindings"].append({
            "sft_row": sft_row, "vid": vid,
            "frames_dir": rows[sft_row].get("frames_dir"),
            "source_shard": name, "row_in_shard": k,
            "feature_revision": f"{name}:{sha256(shard_files[i])[:16]}",
            "must_not_index_by_row_across_pools": True,
        })

    # --- 评估侧缺陷取证 ---
    report["evaluation_path_defect"] = {
        "training_path_guard": "train_vp_head.py:56  assert len(vids) == len(rows)",
        "evaluation_paths_without_guard": [
            "e04_compare.py:118-124  concat → argsort(order) → 无断言无截断",
            "resid_e03.py:85-93      同上，且用 ScoreHead()=ReLU 且未施加 ckpt 的 norm",
        ],
        "consequence": "2700 行缓存喂给索引域 <2508 的 hold_idx ⇒ 静默行号错配",
        "affected_readouts": ["E0.3 VPB head", "E0.4 VideoPrism-B 列"],
        "status": "两者均已在 R21 判废；本节补充的是**具体机制**而非新判决",
    }

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "r22_cache_rebind.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    with open(out / "r22_row_bindings.jsonl", "w") as f:
        for b in report["bindings"]:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    print("REBIND_DONE", json.dumps({
        "shards": len(report["shards"]),
        "frozen_rows": report["frozen_training_view"]["rows"],
        "aligned": aligned,
        "shards_with_appended_rows": len(report["duplicates"]),
    }))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sft", type=Path, required=True)
    p.add_argument("--feat-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args())
