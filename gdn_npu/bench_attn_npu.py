#!/usr/bin/env python3
"""qwen35_attn_npu 的 NPU 基准: 显存峰值 + 速度 A/B(eager vs 分块+checkpoint).

⚠️ 本机 NPU 三大坑适用(见 gdn_npu/README): 新 shape 首编译可能 hang.
   跑法(外层 timeout + 输出落文件, 忌 pipe 长挂):
     cd /root/AIC && set -a && source /data/aic/tools/ascend_teacher.env && set +a
     timeout 900 python3 bench_attn_npu.py --task align    > /tmp/bt_align.log 2>&1
     timeout 900 python3 bench_attn_npu.py --task infer    > /tmp/bt_infer.log 2>&1
     timeout 900 python3 bench_attn_npu.py --task layer    > /tmp/bt_layer.log 2>&1
     timeout 900 python3 bench_attn_npu.py --task stack8   > /tmp/bt_stack8.log 2>&1
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/AIC/gdn_npu")
import qwen35_attn_npu as attn_mod  # noqa: E402


class _Mod:
    def __init__(self, n_rep, training=False):
        self.num_key_value_groups = n_rep
        self.training = training


def _npu():

    import torch_npu  # noqa: F401
    torch.npu.set_device(0)
    return "npu:0"


def _sync():
    torch.npu.synchronize()


def _rel_err(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-12)).item()


def _reset_peak():
    torch.npu.reset_peak_memory_stats()


def _peak_gib():
    return torch.npu.max_memory_allocated() / 2**30


def task_align(B, Hq, Hkv, S, D):
    """NPU fp16 数值对齐: 分块 vs eager, 前向+训练梯度."""
    dev = _npu()
    attn_mod._STATE.update(block=1024, use_ckpt=True, min_elems=0)
    mod = _Mod(Hq // Hkv, training=True)
    g = torch.Generator().manual_seed(0)
    q = (torch.randn(B, Hq, S, D, generator=g) * 0.3).half().to(dev).requires_grad_(True)
    k = (torch.randn(B, Hkv, S, D, generator=g) * 0.3).half().to(dev).requires_grad_(True)
    v = (torch.randn(B, Hkv, S, D, generator=g) * 0.3).half().to(dev).requires_grad_(True)
    scaling = D ** -0.5

    out_ref, _ = attn_mod.eager_attention_forward(mod, q, k, v, None, scaling)
    _sync()
    out_new, _ = attn_mod.chunked_attention_forward(mod, q, k, v, None, scaling)
    _sync()
    d_fwd = _rel_err(out_new, out_ref)

    wgt = torch.linspace(0.1, 1.0, S, device=dev).half().reshape(1, 1, 1, -1)
    (out_ref * wgt).sum().backward(retain_graph=True)
    gq_ref = q.grad.clone()
    for t in (q, k, v):
        t.grad = None
    (out_new * wgt).sum().backward()
    d_gq = _rel_err(q.grad, gq_ref)
    print(json.dumps({"task": "align", "S": S, "d_fwd": d_fwd, "d_gradq": d_gq,
                      "gate": "fwd<3e-3 grad<3e-3",
                      "pass": bool(d_fwd < 3e-3 and d_gq < 3e-3)}), flush=True)


def task_infer(B, Hq, Hkv, S, D):
    """推理 prefill(no_grad): 峰值显存 + 延迟, eager vs 分块."""
    dev = _npu()
    attn_mod._STATE.update(block=1024, use_ckpt=True, min_elems=0)
    mod = _Mod(Hq // Hkv, training=False)
    g = torch.Generator().manual_seed(0)
    q = (torch.randn(B, Hq, S, D, generator=g) * 0.3).half().to(dev)
    k = (torch.randn(B, Hkv, S, D, generator=g) * 0.3).half().to(dev)
    v = (torch.randn(B, Hkv, S, D, generator=g) * 0.3).half().to(dev)
    scaling = D ** -0.5

    res = {"task": "infer", "B": B, "S": S, "block": attn_mod._STATE["block"]}
    for name, fn in (("eager", lambda: attn_mod.eager_attention_forward(
                        mod, q, k, v, None, scaling)),
                     ("chunked", lambda: attn_mod.chunked_attention_forward(
                         mod, q, k, v, None, scaling))):
        with torch.no_grad():
            fn()
            _sync()
            _reset_peak()
            t0 = time.perf_counter()
            for _ in range(10):
                fn()
            _sync()
            dt = (time.perf_counter() - t0) / 10
            res[name] = {"ms": round(dt * 1e3, 3), "peak_gib": round(_peak_gib(), 3)}
        print(f"{name}: {res[name]}", flush=True)
    e, c = res["eager"], res["chunked"]
    res["speedup"] = round(e["ms"] / c["ms"], 2)
    res["peak_ratio"] = round(c["peak_gib"] / max(e["peak_gib"], 1e-9), 3)
    print(json.dumps(res), flush=True)


def task_layer(B, Hq, Hkv, S, D):
    """训练单层 forward+backward: 峰值显存 + 延迟."""
    dev = _npu()
    attn_mod._STATE.update(block=1024, use_ckpt=True, min_elems=0)
    mod = _Mod(Hq // Hkv, training=True)
    g = torch.Generator().manual_seed(0)
    q = (torch.randn(B, Hq, S, D, generator=g) * 0.3).half().to(dev).requires_grad_(True)
    k = (torch.randn(B, Hkv, S, D, generator=g) * 0.3).half().to(dev).requires_grad_(True)
    v = (torch.randn(B, Hkv, S, D, generator=g) * 0.3).half().to(dev).requires_grad_(True)
    scaling = D ** -0.5
    wgt = torch.linspace(0.1, 1.0, S, device=dev).half().reshape(1, 1, 1, -1)

    def run(fn):
        for t in (q, k, v):
            t.grad = None
        _sync()
        _reset_peak()
        t0 = time.perf_counter()
        out, _ = fn()
        out.backward(wgt.expand_as(out) if out.dim() == 4 else out)
        _sync()
        return (time.perf_counter() - t0) * 1e3, _peak_gib()

    res = {"task": "layer_train", "B": B, "S": S, "block": attn_mod._STATE["block"]}
    for name, fn in (("eager", lambda: attn_mod.eager_attention_forward(
                        mod, q, k, v, None, scaling)),
                     ("chunked_ckpt", lambda: attn_mod.chunked_attention_forward(
                         mod, q, k, v, None, scaling))):
        ms, peak = run(fn)
        res[name] = {"ms": round(ms, 1), "peak_gib": round(peak, 3)}
        print(f"{name}: {res[name]}", flush=True)
    e, c = res["eager"], res["chunked_ckpt"]
    res["peak_ratio"] = round(c["peak_gib"] / max(e["peak_gib"], 1e-9), 3)
    res["time_ratio"] = round(c["ms"] / e["ms"], 2)
    print(json.dumps(res), flush=True)


def task_stack8(B, Hq, Hkv, S, D):
    """8 层串联训练图(模拟 4B 的 8 个 full 层激活全保留场景): 峰值显存对比."""
    dev = _npu()
    attn_mod._STATE.update(block=1024, use_ckpt=True, min_elems=0)
    mod = _Mod(Hq // Hkv, training=True)
    g = torch.Generator().manual_seed(0)
    x = (torch.randn(B, Hq, S, D, generator=g) * 0.3).half().to(dev).requires_grad_(True)

    res = {"task": "stack8_train", "B": B, "S": S}
    for name, fn in (("eager", attn_mod.eager_attention_forward),
                     ("chunked_ckpt", attn_mod.chunked_attention_forward)):
        _reset_peak()
        t0 = time.perf_counter()
        h = x
        for _ in range(8):
            out, _ = fn(mod, h, h, h, None, D ** -0.5)
            h = out
        h.sum().backward()
        _sync()
        ms = (time.perf_counter() - t0) * 1e3
        peak = _peak_gib()
        res[name] = {"fwbw_ms": round(ms, 1), "peak_gib": round(peak, 3)}
        print(f"{name}: {res[name]}", flush=True)
    e, c = res["eager"], res["chunked_ckpt"]
    res["peak_ratio"] = round(c["peak_gib"] / max(e["peak_gib"], 1e-9), 3)
    print(json.dumps(res), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    choices=["align", "infer", "layer", "stack8"])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--seq", type=int, default=4900)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--block", type=int, default=1024)
    args = ap.parse_args()

    fn = {"align": task_align, "infer": task_infer,
          "layer": task_layer, "stack8": task_stack8}[args.task]
    attn_mod._STATE["block"] = args.block
    try:
        fn(args.batch, args.heads, args.kv_heads, args.seq, args.dim)
    finally:
        os._exit(0)  # 避开本机退出阶段 TBE daemon 死锁(输出已 flush)


if __name__ == "__main__":
    main()
