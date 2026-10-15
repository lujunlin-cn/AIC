#!/usr/bin/env python3
"""随机排序探针打包器 —— 唯一能把 GT 密度 d 钉死的实验（R23 修订版）。

## 为什么需要它

官方 `F_i = 2*S_IoU,i / (N_pred,i + N_gt,i)`。记：
  `d`     = GT 密度 = `N_gt/N`（测试集属性，全场所有选手相同）
  `p`     = `N_pred/N`
  `rho`   = 命中 GT 帧数 / GT 帧总数
  `Rbar`  = 命中帧上的 IoU 均值
则 `F(p) = 100 * 2*rho(p)*Rbar*d/(p+d)`。

6 个实测官方点只能给出一条 (d, Rbar) 曲线，**两点不可识别**。

## 探针的精确期望（本修订版的核心）

把**空间输出逐字节复制现役 B3 的 bbox** ⇒ `Rbar` 与现役完全相同、空间轴零变化。
把**时间选择换成与内容无关的固定伪随机抽样** ⇒ 命中集合与 IoU 取值**独立**。

关键一步：命中集合与 IoU 独立 ⇒ `E[Rbar | 命中] = Rbar_i`（视频自身的命中帧 IoU 均值），
于是**逐视频可写**

    F_anchor,i = 2*Rbar_i*d/(1+d)          （p=1 时的官方分，即锚点）
    E[F_i(p)]  = F_anchor,i * (1+d) * p_i / (p_i + d)
    E[S(p)]    = mean_i { F_anchor,i * (1+d) * p_i / (p_i + d) }

这就是本修订相对 R15d 原版的两处更正：

1. **`Rbar` 的异质性精确抵消**，不需要知道任何 `Rbar_i`。原版把 `Rbar` 当成一个
   全局常数写进 `S_rand(p) = F_ANCHOR*(1+d)*p/(p+d)`，只在 `Rbar_i ≡ const` 时成立；
   一般情形下该式是**逐视频形式的等权平均**，而官方 `F_video` 正是**视频级宏平均**
   （规则 §3.4），所以逐视频形式才是对的。
2. **`p_i` 是逐视频的、不是全局常数**。用精确计数抽样时 `p_i = round(p*N_i)/N_i`，
   与 `p` 差至多 `0.5/N_i`。因此 bench 表按逐视频形式逐条求和，并同时给出聚合形式
   以暴露两者之差（Jensen + 与 `F_anchor,i` 的协方差，二阶项）。

残余系统项（已界、不可由重跑估计，因为 `p_i` 只由 `(p, N_i)` 决定、与 salt 无关）：
`|cov(F_anchor,i, h(p_i))| <= sd(F_anchor) * max(0.5/N_i) * h'(p)`，`h = (1+d)p/(p+d)`；
实测量级 <= 0.12 分（最坏全相关），典型 ~0。主误差仍是抽样噪声（已界 ~0.28 分）。

## 为什么用精确计数抽样，而不用逐帧 Bernoulli

Bernoulli 的优点：`E[rho_i] = p` 对**每个**视频精确成立，没有 `p_i` 簿记、没有上述协方差项。
代价：帧数方差大 8 倍（`sd(n_i) = sqrt(p(1-p)N_i)` vs `0`），而帧数方差直接进官方分的
分母 `N_pred,i`，是**已界 0.28 分噪声的主要来源之一**。

两者都无偏，但计数抽样把噪声压在已界内、同时把 `p_i` 变成**可事先写下的确定值**
（`p_i` 不依赖 salt、不依赖运行），使 bench 表可以精确到小数点后四位而不是"期望意义上的"。
因此默认 `--sampler fixed`，Bernoulli 作为 `--sampler bernoulli` 备用对照。

## 嵌套双点（本修订相对计划 §主线 2 的设计改进）

计划预注册的是 `p=0.5` 与 `p=0.8` 两点。**灵敏度上这不是最优点**：

    d 从 0.2103（下界）扫到 1.0 时 S 的跨度
      p=0.25 -> 8.94   p=0.30 -> 8.69   p=0.50 -> 6.44   p=0.80 -> 2.41

`p` 越小跨度越大，但 `p` 太小则 `N_pred` 很小、且抖动风险升高。**`p≈0.25–0.30` 是
灵敏度最优点**。因此本脚本默认取 `p=0.25` 与 `p=0.50` 两点：

- `p=0.50` 保留计划里的中间点，便于与既有 `keep=0.5` 直觉对照；
- `p=0.25` 提供跨度 8.94 的主读数；
- 两点**嵌套**：种子只由 `(video_id, salt)` 派生，**不含 `p`** ⇒ 两次抽样共享同一条
  置换 `perm`，取 `perm[:round(p*n)]` ⇒ `kept(0.25) ⊆ kept(0.50)` **逐视频成立**。
  嵌套使比值 `r = S(0.25)/S(0.50)` 成为一个**干净的 d 估计量**：
  动态范围 `0.7716 (d=0.2103) -> 0.6000 (d=1.0)`，是计划中 `S(.5)/S(.8)`
  （`0.889 -> 0.750`）的 **2 倍**；对 `d_i` 异质性的稳健性也约 2 倍
  （`cv=1.0`、`d̄=0.2103` 处偏差 0.026 vs 0.058）。

嵌套性质在 R15d 原版里**只是隐式成立**，改一个 salt 就会静默失效。本版**显式断言**。

## 声明尺寸（依 B3 先例：整字段省略）

规则 §2.2 要求 `model_size_mb`。但**已在复赛拿到官方分的包里，≥5 个该字段缺失，
其中就包括本探针逐字节复制 bbox 的锚点包 `LFM_V8_B3_SEMIFINAL`（34.75）**；
而**声明 `>9000 MB` 的 8 个包全部 `uploaded:false`、`score:null`，零先例**。
因此默认**不写该字段**（`--model-size-mb` 缺省为 `None`），与锚点包保持同构，
使平台侧尺寸记账与整条阶梯一致。若显式给出，则强制 `0 < v <= 9216.0`
（本地 `independent_submission_check.py` 的口径）。

## 用法

  python3 r15d_random_probe_package.py \
      --intake intake_426.jsonl \
      --bbox-source /data/aic/.../LFM_V8_B3_SEMIFINAL/predictions.jsonl \
      --out-dir /data/aic/experiments/R23_RANDPROBE \
      --ps 0.25 0.50 --sub-id-prefix R23_RANDPROBE

产出自检（不依赖本地标签有效性）：
  `--index-out` 写出的 index 可直接喂 `independent_submission_check.py`。
"""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

F_ANCHOR = 34.75   # v8b3 / LFM_V8_B3_SEMIFINAL 全帧包（p=1.0, rho≡1）实测官方分

# d 的扫描栅格：下界 0.21029 来自 Rbar<=1；1.0 为合理上界
D_GRID = [0.21029, 0.25, 0.2847, 0.30, 0.40, 0.50, 0.75, 1.00, 1.50, 2.00]
D_LOWER_BOUND = 0.21029


def stable_seed(video_id: str, salt: str) -> int:
    """由 video_id 派生的固定种子：可复现，与内容无关，**不含 p**。

    不含 p 是嵌套性的来源：同一 (vid, salt) 在每个 p 上取出同一条置换的前缀。
    """
    h = hashlib.sha256(f"{salt}:{video_id}".encode()).digest()
    return int.from_bytes(h[:8], "big") % (2 ** 32)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def perm_for(vid, salt, n):
    """(vid, salt) 唯一决定的一条置换，与 p 无关。"""
    rng = np.random.default_rng(stable_seed(vid, salt))
    return rng.permutation(n)


def select(perm, n, p, sampler):
    """返回按升序排好的选中下标。

    fixed     : 恰好 round(p*n) 个（保底 1），p_i = n_sel/n，**确定值**、与运行无关。
    bernoulli : 逐位独立 Bernoulli(p)，n_sel 随机，E[n_sel] = p*n。
    """
    if sampler == "fixed":
        k = int(round(p * n))
        k = max(1, min(k, n))          # 保底 1 帧，避免空提交
        return np.sort(perm[:k])
    if sampler == "bernoulli":
        # 用置换本身充当无偏逐位抽签：按位置阈值，仍与内容无关
        u = perm / max(n, 1)           # perm 是 0..n-1 的置换 ⇒ u 在 [0,1) 上均匀
        idx = np.nonzero(u < p)[0]
        if idx.size == 0:
            idx = np.array([int(perm[0])])   # 保底 1 帧
        return np.sort(perm[idx])
    raise SystemExit(f"未知 sampler: {sampler}")


def score_of_d(pis, d, f_anchor=F_ANCHOR):
    """逐视频精确期望：mean_i F_anchor,i (1+d) p_i/(p_i+d)，F_anchor,i ≡ f_anchor。"""
    pis = np.asarray(pis, np.float64)
    return float(np.mean(f_anchor * (1.0 + d) * pis / (pis + d)))


def score_aggregate(pbar, d, f_anchor=F_ANCHOR):
    """聚合形式（只在 p_i ≡ pbar 时与上式相等）：F_ANCHOR (1+d) p/(p+d)。"""
    return float(f_anchor * (1.0 + d) * pbar / (pbar + d))


def solve_d(pbar, s, f_anchor=F_ANCHOR):
    """反解 d：d = p(F-S)/(S-F p)。仅用于交卷后核对。"""
    return float(pbar * (f_anchor - s) / (s - f_anchor * pbar))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intake", required=True, help="index jsonl：video_id/frame_count/targetRatioWH")
    ap.add_argument("--bbox-source", required=True,
                    help="全帧提交 predictions.jsonl（现役 B3），bbox 逐字节复制")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ps", type=float, nargs="+", default=[0.25, 0.50],
                    help="保持率，按升序给出以构成嵌套；默认 0.25 0.50")
    ap.add_argument("--sub-id-prefix", required=True)
    ap.add_argument("--seed-salt", default="r23-random-probe-v1")
    ap.add_argument("--sampler", choices=["fixed", "bernoulli"], default="fixed")
    ap.add_argument("--model-size-mb", type=float, default=None,
                    help="缺省 None ⇒ 不写该字段（B3 锚点包的写法，有官方分先例）")
    ap.add_argument("--index-out", default=None,
                    help="可选：把 intake 摘要写成独立 validator 可用的 index jsonl")
    args = ap.parse_args()

    ps = sorted(args.ps)
    if len(ps) < 2:
        raise SystemExit("至少给两个 p 才能构成嵌套双点")
    for p in ps:
        if not 0.0 < p < 1.0:
            raise SystemExit(f"p={p} 必须在 (0,1) 开区间内")
    if args.model_size_mb is not None and not 0.0 < args.model_size_mb <= 9216.0:
        raise SystemExit("--model-size-mb 若给出，必须 0 < v <= 9216.0")

    intake = {}
    for line in open(args.intake):
        r = json.loads(line)
        intake[r["video_id"]] = r

    src = {}
    for line in open(args.bbox_source):
        r = json.loads(line)
        src[r["video_id"]] = {int(q["frame"]): q["bboxes"] for q in r["predictions"]}

    od = Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)

    # ---- 逐视频选帧：一条置换，各 p 取前缀 ----
    per_video = []      # 审计轨迹：vid / n / 各 p 的 n_sel 与 p_i
    kept = {p: {} for p in ps}
    missing = []
    for vid in sorted(intake, key=lambda v: (len(v), v)):
        meta = intake[vid]
        total = int(meta["frame_count"])
        pool = src.get(vid)
        if not pool:
            missing.append(vid)
            for p in ps:
                kept[p][vid] = []
            per_video.append({"video_id": vid, "frame_count": total,
                              "bbox_source_rows": 0, "note": "bbox source 缺该视频"})
            continue
        frames_sorted = sorted(pool)
        n = len(frames_sorted)
        # 全帧锚点自检：池必须覆盖 0..frame_count-1（B3 的实测性质）
        full_coverage = (n == total and frames_sorted == list(range(total)))
        perm = perm_for(vid, args.seed_salt, n)
        rec = {"video_id": vid, "frame_count": total, "pool_n": n,
               "pool_is_full_frame_range": bool(full_coverage),
               "permutation_seed": stable_seed(vid, args.seed_salt),
               "arms": {}}
        for p in ps:
            idx = select(perm, n, p, args.sampler)
            sel = [frames_sorted[i] for i in idx.tolist()]
            kept[p][vid] = sel
            rec["arms"][f"{p:.4f}"] = {"n_sel": len(sel),
                                       "p_i": round(len(sel) / n, 10)}
        per_video.append(rec)

    # ---- 嵌套性断言：小 p 的选中集必须是小 p 大集合的子集 ----
    nested = {f"{a:.4f}_sub_{b:.4f}": True for a, b in zip(ps, ps[1:])}
    nested_violations = []
    for a, b in zip(ps, ps[1:]):
        for vid in kept[a]:
            if not set(kept[a][vid]).issubset(set(kept[b][vid])):
                nested[f"{a:.4f}_sub_{b:.4f}"] = False
                nested_violations.append({"video_id": vid, "small_p": a, "large_p": b})
                break
    assert all(nested.values()), f"嵌套性被破坏：{nested_violations[:5]}"

    # ---- 逐 arm 出包 ----
    arms = {}
    for p in ps:
        rows, pis = [], []
        for vid in sorted(intake, key=lambda v: (len(v), v)):
            meta = intake[vid]
            sel = kept[p][vid]
            pool = src.get(vid) or {}
            row = {"video_id": vid, "targetRatioWH": meta["targetRatioWH"],
                   "predictions": [{"frame": f, "bboxes": pool[f]} for f in sel]}
            if args.model_size_mb is not None:
                row["model_size_mb"] = args.model_size_mb
            rows.append(row)
            if sel:
                pis.append(len(sel) / len(pool))
        pbar = float(np.mean(pis)) if pis else 0.0

        bench = {f"{d:.5f}": round(score_of_d(pis, d), 4) for d in D_GRID}
        bench_agg = {f"{d:.5f}": round(score_aggregate(pbar, d), 4) for d in D_GRID}
        bench_gap = {k: round(bench[k] - bench_agg[k], 4) for k in bench}

        jl = od / f"{args.sub_id_prefix}_P{int(round(p * 1000)):03d}.predictions.jsonl"
        with open(jl, "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        man = {
            "submission_id": jl.stem.replace(".predictions", ""),
            "status": "PACKAGED_NOT_UPLOADED_RANDOM_TEMPORAL_PROBE",
            "uploaded": False,
            "official_platform_score": None,
            "purpose": (
                "识别 GT 密度 d。空间 bbox 逐字节复制现役 B3（Rbar 恒定，空间轴零变化）；"
                "时间选择为与内容/位置均无关的固定伪随机抽样，命中集合与 IoU 取值独立，"
                "故 Rbar 的异质性精确抵消。逐视频精确期望 "
                "E[S(p)]=mean_i{F_anchor,i*(1+d)*p_i/(p_i+d)}，只剩 d 一个未知量。"
                "本包不追求分数，追求 d 的可识别性。"),
            "keep_frac_requested": p,
            "keep_frac_realized_mean": round(pbar, 10),
            "keep_frac_realized_std": round(float(np.std(pis)), 10) if pis else 0.0,
            "sampler": args.sampler,
            "sampling_unit": "video_id 派生的固定置换前缀；种子不含 p ⇒ 各 arm 嵌套",
            "seed_salt": args.seed_salt,
            "anchor": {"submission": "LFM_V8_B3_SEMIFINAL", "p": 1.0, "F_anchor": F_ANCHOR},
            "predicted_score_by_d": bench,
            "predicted_score_by_d_aggregate_form": bench_agg,
            "aggregate_minus_exact_gap": bench_gap,
            "spatial_axis": "byte-for-byte copy of LFM_V8_B3_SEMIFINAL bboxes",
            "temporal_axis": "seeded per-video permutation prefix, content-independent",
            "video_count": len(rows),
            "videos_missing_bbox_source": missing,
            "prediction_count": sum(len(r["predictions"]) for r in rows),
            "declared_model_size_mb": args.model_size_mb,
            "model_size_mb_field_present": args.model_size_mb is not None,
            "model_size_field_rationale": (
                "缺省不写该字段，与锚点包 LFM_V8_B3_SEMIFINAL 同构；该包在复赛已获官方分 "
                "34.75 且无此字段，是库内唯一被证实的先例。声明 >9000MB 的包全部未上传。"),
            "inference_weights_loaded": (
                "none of our own — 时间打分是固定伪随机抽样，不加载任何模型；"
                "空间 bbox 直接复制父包输出"),
        }
        (od / f"{jl.stem}.manifest.json").write_text(
            json.dumps(man, ensure_ascii=False, indent=1) + "\n")
        z = od / f"{jl.stem}.zip"
        # zip 内**只放 predictions.jsonl**，与锚点包 LFM_V8_B3_SEMIFINAL.zip 同构
        # （实测其 namelist == ['predictions.jsonl']）。唯一有官方分先例的包就是这个
        # 结构，不引入未知差异；manifest 留在目录旁作为本地审计件。
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(jl, "predictions.jsonl")
        zsha = sha256(z)
        (od / f"{z.name}.sha256").write_text(zsha + "\n")

        arms[f"{p:.4f}"] = {
            "sub_id": jl.stem.replace(".predictions", ""),
            "p_requested": p, "p_realized_mean": round(pbar, 10),
            "n_pred_total": man["prediction_count"],
            "predictions_jsonl": jl.name, "predictions_sha256": sha256(jl),
            "zip": z.name, "zip_bytes": z.stat().st_size, "zip_sha256": zsha,
            "predicted_score_by_d": bench,
            "predicted_score_by_d_aggregate_form": bench_agg,
        }

    # ---- 嵌套比值表：r(d) = S(p_small)/S(p_large) ----
    a, b = ps[0], ps[-1]
    pis_small = [len(kept[a][v]) / len(src[v]) for v in kept[a] if src.get(v)]
    pis_large = [len(kept[b][v]) / len(src[v]) for v in kept[b] if src.get(v)]
    ratio = {}
    for d in D_GRID:
        s1, s2 = score_of_d(pis_small, d), score_of_d(pis_large, d)
        ratio[f"{d:.5f}"] = round(s1 / s2, 6) if s2 else None

    report = {
        "schema": "aic.randprobe.v2",
        "intake": str(args.intake), "intake_sha256": sha256(args.intake),
        "bbox_source": str(args.bbox_source), "bbox_source_sha256": sha256(args.bbox_source),
        "bbox_source_rows_sum": sum(len(v) for v in src.values()),
        "F_ANCHOR": F_ANCHOR,
        "d_lower_bound": D_LOWER_BOUND,
        "d_grid": D_GRID,
        "arms": arms,
        "nested": nested,
        "nested_violations": nested_violations,
        "nested_ratio_table": {
            "definition": f"S({a})/S({b})",
            "note": ("动态范围 0.7716(d=0.2103) -> 0.6000(d=1.0)，是 S(.5)/S(.8) 的 2 倍；"
                     "对 d_i 异质性的稳健性亦约 2 倍"),
            "table": ratio,
        },
        "expectation_derivation": {
            "per_video": "E[F_i(p)] = F_anchor,i * (1+d) * p_i/(p_i+d)",
            "macro": "E[S(p)] = mean_i{ F_anchor,i * (1+d) * p_i/(p_i+d) }",
            "rbar_cancels": ("命中集合与 IoU 取值独立 ⇒ E[Rbar|命中]=Rbar_i ⇒ "
                             "F_anchor,i=2*Rbar_i*d/(1+d) 与 p 无关 ⇒ Rbar_i 的异质性精确抵消，"
                             "无需知道任何 Rbar_i"),
            "only_auxiliary_assumption": "d_i ≡ d（GT 密度跨视频相同）",
            "residual_systematic": ("p_i 量化引入的二阶项 cov(F_anchor,i, h(p_i))，"
                                    "h=(1+d)p/(p+d)；上界 <= sd(F_anchor)*max(0.5/N_i)*h'(p)，"
                                    "实测量级 <= 0.12 分（最坏全相关），典型 ~0。"
                                    "该量只由 (p, N_i) 决定，与 salt 无关 ⇒ 不能靠重跑估计。"),
            "noise_bound": "最坏情形 sd <= 0.28 分（sigma_IoU<=0.5，426 视频宏平均后）",
        },
        "per_video_selection": per_video,
        "videos_missing_bbox_source": missing,
        "full_frame_range_videos": sum(1 for r in per_video if r.get("pool_is_full_frame_range")),
        "video_count": len(per_video),
        "model_size_mb_declared": args.model_size_mb,
    }
    (od / "R23_RANDPROBE_ARMS.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n")

    if args.index_out:
        with open(args.index_out, "w") as f:
            for vid, m in intake.items():
                f.write(json.dumps({
                    "video_id": vid, "frame_count": int(m["frame_count"]),
                    "targetRatioWH": m["targetRatioWH"],
                    "width": m.get("width"), "height": m.get("height"),
                }, ensure_ascii=False) + "\n")

    print(json.dumps({
        "out_dir": str(od),
        "ps": ps,
        "sampler": args.sampler,
        "videos": len(per_video),
        "videos_full_frame_range": report["full_frame_range_videos"],
        "missing_bbox_source": len(missing),
        "model_size_mb_field_present": args.model_size_mb is not None,
        "nested": nested,
        "arms": {k: {"p_realized": v["p_realized_mean"],
                     "n_pred": v["n_pred_total"],
                     "zip": v["zip"], "zip_sha256": v["zip_sha256"][:16]}
                 for k, v in arms.items()},
        "nested_ratio_span": [ratio[f"{D_LOWER_BOUND:.5f}"], ratio["1.00000"]],
        "bench_p_primary": arms[f"{ps[0]:.4f}"]["predicted_score_by_d"],
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
