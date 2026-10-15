#!/usr/bin/env python3
"""R22 阶段1 仪器重建 —— 有界下载 64 公共源 + PTS 轴审计 + 10 人原始秒标注栅格化。

## 为什么需要这个脚本

R21 审计判定：项目过去用以 steering 的本地数字全部失效——泄漏、行号错配、指标混比。
根因之一是**缓存合同破裂**：标签按行号索引到另一个池的特征缓存上，
hold125 有 124/125 源错配。

本脚本按 R20 已冻结的合同重建评估基线，每个源绑定
`(source_id, media_hash, actual_pts, preprocess)`，禁止跨池复用行号。

## 与既有脚本的关系

- 媒体恢复逻辑复现 `R18_DATA_20261015/code/r18_recover_videoxum_pilot.py`
  （huggingface.co 在本机不可达 → 换 hf-mirror.com；逐文件大小校验）。
- 轴审计复现 `R18_DATA_20261015/code/r18_audit_media_axis.py`
  （完整解码、PTS 单调性、`ID_DURATION_RAW_SECONDS_CONSISTENT` 判据）。
- 标注来自 `VideoXum_R16/annotations/train_videoxum.json`：
  `vsum` 是 [段][10 rater][lo,hi]，`vsum_onehot` 是 10×N 布尔栅格。
  **rater 恒为 10**（8000/8000 视频均如此）；变化的是"段"的个数。

## 硬约束

- 总量 <=2GB，wall <=2400s，2 worker（R20 预注册预算）。
- 只下载公共 TRAIN 媒体；不碰任何赛事测试数据。
- 失败保留，**不自动替换源**（R20_RULES）。
- 不训练、不拟合、不出预测。

用法：
  /root/miniconda3/bin/python r22_build_instrument.py \
      --candidates <candidate_sources.jsonl> \
      --annotations <train_videoxum.json> \
      --output /root/autodl-tmp/experiments/R22_INSTRUMENT_20261015
"""
import argparse
import hashlib
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

MIRROR = "hf-mirror.com"
ORIGIN = "huggingface.co"
MAX_TOTAL_BYTES = 2_000_000_000
MAX_WALL_S = 2400
MAX_FILE_BYTES = 128_000_000


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def probe(url, timeout=30, attempts=3):
    """确认可用 transport host，返回 (host, 全部尝试记录)。

    镜像偶发超时（实测同一 URL 手动请求 3s 成功、脚本内首次失败），
    故每个 host 重试 attempts 次并保留全部记录，不静默丢弃失败原因。
    """
    assert urllib.parse.urlparse(url).hostname == ORIGIN
    recs = []
    for host in (MIRROR, ORIGIN):
        u = url.replace(f"https://{ORIGIN}/", f"https://{host}/", 1)
        for k in range(attempts):
            rec = {"host": host, "attempt": k + 1}
            try:
                req = urllib.request.Request(
                    u, headers={"User-Agent": "AIC-R22-instrument/1.0",
                                "Range": "bytes=0-65535"})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    b = r.read(65536)
                    rec.update(status=r.status, bytes=len(b),
                               mp4_ftyp=b"ftyp" in b[:64])
                    if not rec["mp4_ftyp"]:
                        raise ValueError("not MP4")
                    rec["ok"] = True
                    recs.append(rec)
                    return host, recs
            except Exception as e:
                rec.update(ok=False, error_type=type(e).__name__,
                           error=str(e)[:160])
            recs.append(rec)
    return None, recs


def download(task):
    """单个源：有界下载 + 大小校验。

    镜像会 429 限流（实测首轮 64 源里 24 个 429）。429 是**传输层**错误，
    必须与"源不可用"区分：直接记为 failed 会把限流污染成数据可得性结论。
    故对 429/timeout 做指数退避重试；重试仍败才落 failed，并保留全部尝试记录。
    """
    row, host, outdir, deadline = task
    vid = row["video_id"]
    target = outdir / "videos" / f"{vid}.mp4"
    part = target.with_suffix(".part")
    rec = {"video_id": vid, "split": row["split"],
           "publisher_size_bytes": row["publisher_size_bytes"],
           "transport_host": host, "attempts": []}
    if target.exists() and target.stat().st_size == row["publisher_size_bytes"]:
        rec.update(status="already_present",
                   sha256=sha256(target), bytes=target.stat().st_size)
        return rec
    if time.monotonic() > deadline:
        rec.update(status="time_budget_reached")
        return rec
    size = int(row["publisher_size_bytes"])
    if size > MAX_FILE_BYTES:
        rec.update(status="skipped_over_file_budget", bytes=size)
        return rec
    url = row["video_url"].replace(f"https://{ORIGIN}/", f"https://{host}/", 1)
    start = time.monotonic()

    # 每个源最多 4 次；退避 4/12/28s，并加 0~3s 抖动打散两个 worker 的相位。
    last = None
    for k in range(4):
        if time.monotonic() > deadline:
            rec.update(status="time_budget_reached",
                       wall_s=round(time.monotonic() - start, 2))
            return rec
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "AIC-R22-instrument/1.0"})
            with urllib.request.urlopen(req, timeout=45) as r, part.open("xb") as f:
                got = 0
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    got += len(b)
                    if got > MAX_FILE_BYTES or time.monotonic() > deadline:
                        raise ValueError("byte/time budget exceeded")
                    f.write(b)
            if got != size:
                raise ValueError(f"publisher size mismatch {got} vs {size}")
            part.rename(target)
            rec.update(status="downloaded_size_verified", bytes=got,
                       sha256=sha256(target), retries=k,
                       wall_s=round(time.monotonic() - start, 2))
            rec["attempts"].append({"attempt": k + 1, "ok": True, "bytes": got})
            return rec
        except Exception as e:
            if part.exists():
                part.unlink()
            code = getattr(e, "code", None)
            transient = code in (429, 500, 502, 503, 504) or isinstance(
                e, (TimeoutError, urllib.error.URLError))
            last = e
            rec["attempts"].append({"attempt": k + 1, "ok": False,
                                    "error_type": type(e).__name__,
                                    "http_code": code, "error": str(e)[:120],
                                    "transient": bool(transient)})
            if not transient:
                break
            time.sleep(4 * (2 ** k) + random.uniform(0, 3))
    rec.update(status="failed", error_type=type(last).__name__,
               http_code=getattr(last, "code", None), error=str(last)[:180],
               wall_s=round(time.monotonic() - start, 2))
    return rec


def axis_audit(task):
    """完整解码 + PTS 轴审计 + 10 人原始秒标注栅格化。

    判据复现 r18_audit_media_axis.py：
      ready = |decoded_end - annotation_duration| <= max(0.5, 2/fps)
              且 原始区间均在 [0, annotation_duration] 内
              且 无区间越过实际媒体末尾
    """
    import av  # 延迟导入：CPU 任务，避免主进程依赖
    # 注意：需要带 av 的解释器（实测 /root/autodl-tmp/aha_env/bin/python, av 13.0.0）。
    # /root/miniconda3/bin/python 没有 av，会 ModuleNotFoundError。

    row, ann, outdir = task
    vid = row["video_id"]
    path = outdir / "videos" / f"{vid}.mp4"
    if not path.exists():
        return {"video_id": vid, "status": "MEDIA_MISSING"}
    start = time.monotonic()
    times = []
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        stream.codec_context.thread_count = 1
        fps = float(stream.average_rate or 0)
        declared = (float(stream.duration * stream.time_base)
                    if stream.duration is not None else None)
        container = c.duration / 1e6 if c.duration is not None else None
        shape = (stream.width, stream.height)
        for frame in c.decode(stream):
            if frame.pts is None:
                raise ValueError("decoded frame has no PTS")
            times.append(float(frame.pts * frame.time_base))
    ts = np.asarray(times, np.float64)
    if len(ts) < 2:
        return {"video_id": vid, "status": "DECODE_FAIL", "decoded_frames": len(ts)}
    if not np.all(np.diff(ts) >= 0):
        return {"video_id": vid, "status": "DECODE_FAIL", "reason": "nonmonotonic PTS"}
    origin = float(ts[0])
    ts = ts - origin
    step = float(np.median(np.diff(ts)))
    end = float(ts[-1] + step)

    # vsum: [段][10 rater] -> [lo,hi]
    intervals = [[] for _ in range(10)]
    for segment in ann["vsum"]:
        if len(segment) != 10:
            return {"video_id": vid, "status": "ANNOTATION_SCHEMA_FAIL",
                    "segment_len": len(segment)}
        for rater, pair in enumerate(segment):
            if len(pair) != 2:
                return {"video_id": vid, "status": "ANNOTATION_SCHEMA_FAIL"}
            intervals[rater].append([float(pair[0]), float(pair[1])])

    dur = float(ann["duration"])
    invalid = [{"rater": i, "interval": p} for i, ps in enumerate(intervals)
               for p in ps if not (0 <= p[0] < p[1] <= dur + 0.001)]
    outside = [{"rater": i, "interval": p} for i, ps in enumerate(intervals)
               for p in ps if p[1] > end + 0.001]

    raw = np.zeros((10, len(ts)), bool)
    for i, ps in enumerate(intervals):
        for lo, hi in ps:
            raw[i] |= (ts >= lo) & (ts <= hi)

    delta = end - dur
    tolerance = max(0.5, 2 / max(fps, 1))
    ready = (abs(delta) <= tolerance) and not invalid and not outside

    arrdir = outdir / "axis_arrays"
    arrdir.mkdir(exist_ok=True)
    arrpath = arrdir / f"{vid}.npz"
    np.savez_compressed(arrpath, timestamps=ts, raw_summary_membership=raw,
                        rater_count=np.int32(10))

    # 栅格化的整数秒 ten-rater 目标（官方口径：逐帧；此处按实际 PTS 帧）
    saved = np.asarray(ann["vsum_onehot"], bool)
    n = int(ann["sampled_frames"])
    raster_ok = saved.shape == (10, n)

    return {
        "video_id": vid, "split": row["split"],
        "status": "ID_DURATION_RAW_SECONDS_CONSISTENT" if ready else "AXIS_GATE_FAIL",
        "decoded_frames": len(ts),
        "decoded_span_with_median_frame_s": end,
        "decoded_first_pts_s": origin,
        "stream_duration_s": declared, "container_duration_s": container,
        "fps": fps, "width": shape[0], "height": shape[1],
        "annotation_duration_s": dur, "sampled_frames": n,
        "decoded_minus_annotation_duration_s": delta,
        "tolerance_s": tolerance,
        "raw_interval_invalid": invalid[:5], "raw_interval_invalid_n": len(invalid),
        "raw_interval_outside_media": outside[:5], "raw_interval_outside_media_n": len(outside),
        "vsum_onehot_shape_ok": bool(raster_ok),
        "onehot_positive_fraction": float(saved.mean()) if raster_ok else None,
        "raw_positive_fraction": float(raw.mean()),
        "media_sha256": sha256(path),
        "axis_array_sha256": sha256(arrpath),
        "axis_array_relpath": f"axis_arrays/{vid}.npz",
        "media_relpath": f"videos/{vid}.mp4",
        "wall_s": round(time.monotonic() - start, 2),
        "training": False, "fitting": False,
    }


def main(a):
    if a.output.exists() and not a.resume:
        raise SystemExit(f"{a.output} already exists; pass --resume to continue")
    a.output.mkdir(parents=True, exist_ok=True)
    cands = [json.loads(s) for s in a.candidates.read_text().splitlines()]
    ann = {r["video_id"]: r for r in json.loads(a.annotations.read_text())}
    missing = [r["video_id"] for r in cands if r["annotation_id"] not in ann]
    if missing:
        raise SystemExit(f"candidates without local annotation: {missing[:5]}")

    report = {
        "scope": "public TRAIN VideoXum media + 10-rater original-second summary; "
                 "no competition test data, no fitting, no prediction",
        "candidates_sha256": sha256(a.candidates),
        "annotations_sha256": sha256(a.annotations),
        "sources": len(cands),
        "split_counts": {s: sum(1 for r in cands if r["split"] == s)
                         for s in sorted({r["split"] for r in cands})},
        "byte_budget": MAX_TOTAL_BYTES, "wall_budget_s": MAX_WALL_S,
        "probes": [], "downloads": [], "axis": [],
        "training": False, "promotion": False,
    }
    (a.output / "videos").mkdir(exist_ok=True)
    save = lambda: (a.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n")
    prior = a.output / "report.json"
    if a.resume and prior.exists():
        old = json.loads(prior.read_text())
        # 不预载 downloads/axis：download() 与 axis_audit() 都是幂等的
        # （前者靠文件存在性返回 already_present，后者对同一份文件结果确定）。
        # 预载 + 本轮 append 会把同一批源算两遍 —— 上一轮就是这么把 64 源
        # 计成 99 条的字节量、误触 2GB 上限而中断的。
        report["resumed_from"] = old.get("status")
        # 失败原因本身是审计证据（如 429 限流），不能因重跑成功就丢掉。
        report["prior_failed_downloads"] = [
            {"video_id": r.get("video_id"), "status": r.get("status"),
             "error_type": r.get("error_type"), "http_code": r.get("http_code"),
             "error": r.get("error")}
            for r in old.get("downloads", []) if r.get("status") == "failed"]
        # 上一轮已判定的轴记录带过来（轴审计是纯 CPU 且耗解码，重跑很贵）；
        # 本轮只对不在其中的源做审计。
        report["axis"] = [r for r in old.get("axis", [])
                          if r.get("status") in
                          ("ID_DURATION_RAW_SECONDS_CONSISTENT", "AXIS_GATE_FAIL")]

    host, precs = probe(cands[0]["video_url"])
    report["probes"] = precs
    save()
    for prec in precs:
        print("PROBE", json.dumps(prec), flush=True)
    if not host:
        report["status"] = "NEED_MEDIA"
        save()
        return

    deadline = time.monotonic() + MAX_WALL_S
    tasks = [(r, host, a.output, deadline) for r in cands]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for rec in pool.map(download, tasks):
            report["downloads"].append(rec)
            print("MEDIA", json.dumps(rec), flush=True)
            save()

    total = sum(r.get("bytes", 0) for r in report["downloads"]
                if r["status"] in ("downloaded_size_verified", "already_present"))
    report["downloaded_bytes"] = total
    if total > MAX_TOTAL_BYTES:
        raise SystemExit("total byte budget exceeded")

    ok = {r["video_id"] for r in report["downloads"]
          if r["status"] in ("downloaded_size_verified", "already_present")}
    done_axis = {r["video_id"] for r in report["axis"]}
    atasks = [(r, ann[r["annotation_id"]], a.output) for r in cands
              if r["video_id"] in ok and r["video_id"] not in done_axis]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for rec in pool.map(axis_audit, atasks):
            report["axis"].append(rec)
            print("AXIS", json.dumps({k: rec.get(k) for k in
                  ("video_id", "status", "decoded_minus_annotation_duration_s")}), flush=True)
            save()

    good = [r for r in report["axis"]
            if r.get("status") == "ID_DURATION_RAW_SECONDS_CONSISTENT"]
    by_split = {}
    for r in good:
        by_split.setdefault(r["split"], []).append(r["video_id"])
    report["qc_summary"] = {
        "media_ok": len(ok),
        "axis_pass": len(good),
        "axis_fail": [r["video_id"] for r in report["axis"]
                      if r.get("status") == "AXIS_GATE_FAIL"],
        "by_split": {k: len(v) for k, v in sorted(by_split.items())},
        "pass_ids_by_split": {k: v for k, v in sorted(by_split.items())},
    }
    report["status"] = "complete"
    report["wall_s"] = round(time.monotonic() - (deadline - MAX_WALL_S), 2)
    save()
    print("R22_INSTRUMENT_DONE", json.dumps(report["qc_summary"]), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", action="store_true",
                   help="复用已存在的输出目录（已下载且大小正确的文件会跳过）")
    main(p.parse_args())
