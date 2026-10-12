#!/usr/bin/env python3
"""GDN/attn 补丁的真实生成语境探针(4B fp16, NPU) — 单臂进程隔离版.

背景(2026-10-12 主会话实测): attn+gdn 满配(nsq+affine)下 base 4B 输出 token 汤
(sparse_rate 0.000), 撤 gdn 留 attn 恢复 → GDN 补丁在 fp16 真实生成链路的
数值损坏; 我们全部验证都在 CPU 语义层/单次 forward, 真实 prefill→decode 链路
从没测过.

⚠️ 首跑(10-12 深夜)死于 TBE 编译 hang: 探针没带生产推理配方
   torch.npu.set_compile_mode(jit_compile=False) → GDN 大量小算子走 TBE
   在线编译. 生产 rollout 从不 hang 靠的就是这一行(r13/r14 脚本均有的同款).

臂表(6 臂二分):
  base            safe_chunk 基线(金丝雀, 应正常)
  attn            safe_chunk + attn 分块
  gdn_tri_seq     gdn tri+seq + attn
  gdn_nsq_seq     gdn nsquaring+seq + attn
  gdn_tri_affine  gdn tri+affine + attn
  gdn_nsq_affine  gdn nsquaring+affine + attn  (= 主会话复现配置)

跑法(每臂独立进程 = 三大坑③配方; 多卡时并行发):
  cd /root/AIC && set -a && source /data/aic/tools/ascend_teacher.env && set +a
  ASCEND_RT_VISIBLE_DEVICES=5 timeout 1500 python3 bench_gdn_real.py --arm base
  ASCEND_RT_VISIBLE_DEVICES=6 timeout 1500 python3 bench_gdn_real.py --arm gdn_nsq_affine
  ...每臂 JSON 落 /tmp/gdn_real_<arm>.json, 汇总: --summarize
"""
import argparse
import json
import os
import sys
import time

import torch

MODEL = "/data/aic/pretrained/Qwen3.5-4B"
PROMPTS = [
    "The capital of France is",
    "请用一句话介绍北京。",
]
ARMS = ["base", "attn", "gdn_tri_seq", "gdn_nsq_seq",
        "gdn_tri_affine", "gdn_nsq_affine"]


def soup_score(text):
    """汤分(0 正常 ~1 全汤): unique 词比率低 + 可读字符占比低."""
    if not text.strip():
        return 1.0
    toks = text.split()
    uniq = len(set(toks)) / max(len(toks), 1)
    good = sum(1 for ch in text if ch.isalnum() or "一" <= ch <= "鿿"
               or ch.isspace())
    return max(0.0, 1.0 - uniq * (good / max(len(text), 1)) * 2.0)


def set_arm(arm):
    """臂间补丁切换(进程内只跑一臂, 无需来回切)."""
    import qwen35_gdn_npu as gdn
    import qwen35_attn_npu as attn
    import r13_qwen35_safe_chunk as sc

    if arm == "base":
        sc.apply()
        return
    if arm == "attn":
        sc.apply()
        attn.apply()
        return
    solver, _, scan = arm[len("gdn_"):].partition("_")
    gdn.apply(solver=solver, scan=scan)
    attn.apply()


def run_arm(arm, max_new):
    import torch_npu  # noqa: F401
    torch.npu.set_device(0)
    # 生产同款推理配方: 关 TBE 在线编译(首跑 hang 根因), 必须 load 前
    torch.npu.set_compile_mode(jit_compile=False)
    dev = "npu:0"
    sys.path.insert(0, "/root/AIC/scripts")
    sys.path.insert(0, "/root/AIC/gdn_npu")

    set_arm(arm)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, attn_implementation="eager").to(dev)
    model.eval()
    print("model loaded", flush=True)

    texts = []
    for pi, prompt in enumerate(PROMPTS):
        ids = tok(prompt, return_tensors="pt").to(dev)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **ids, max_new_tokens=max_new, do_sample=False,
                pad_token_id=tok.eos_token_id)
        dt = time.time() - t0
        text = tok.decode(out[0][ids.input_ids.shape[1]:],
                          skip_special_tokens=True)
        score = soup_score(text)
        texts.append({"prompt": prompt, "text": text,
                      "soup": round(score, 3), "sec": round(dt, 1)})
        print(f"[{arm}] p{pi} soup={score:.2f} ({dt:.1f}s) {text!r}",
              flush=True)

    res = {"arm": arm, "texts": texts, "soup_max": max(t["soup"] for t in texts)}
    print(f"RESULT {json.dumps(res, ensure_ascii=False)}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=None, choices=ARMS,
                    help="单臂模式(独立进程, 生产编译配方)")
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--summarize", action="store_true",
                    help="汇总 /tmp/gdn_real_*.json")
    args = ap.parse_args()

    if args.summarize:
        rows = []
        for arm in ARMS:
            p = f"/tmp/gdn_real_{arm}.json"
            if os.path.exists(p):
                rows.append(json.load(open(p)))
        for r in rows:
            verdict = "OK" if r["soup_max"] < 0.5 else "SOUP"
            print(f"{r['arm']:18s} soup_max={r['soup_max']:.2f} {verdict}",
                  flush=True)
        return

    if args.arm is None:
        print(__doc__)
        return

    res = run_arm(args.arm, args.max_new)
    with open(f"/tmp/gdn_real_{args.arm}.json", "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    sys.stdout.flush()
    os._exit(0 if res["soup_max"] < 0.5 else 2)


if __name__ == "__main__":
    main()
