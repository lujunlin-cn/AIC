#!/usr/bin/env python3
"""qwen35_attn_npu.chunked_attention_forward 的自包含数值对齐测试.

对齐锚: 独立内嵌的 eager 参考(与被测文件内 eager 互为拷贝校验),
        causal/mask/pad/训练梯度全矩阵. CPU 默认(沙箱可跑), NPU 等空闲指令:
        python3 test_attn_correctness.py                 # CPU
        python3 test_attn_correctness.py --device npu    # 910A(空闲时)
        python3 test_attn_correctness.py --isolate       # 每 case 子进程隔离
"""
import argparse
import json
import os
import subprocess
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen35_attn_npu as attn_mod  # noqa: E402


# ---------------------------------------------------------------------------
# 独立 eager 参考(重打一份, 不 import 自被测模块, 防拷贝错误连坐)
# ---------------------------------------------------------------------------
def _repeat_kv_ref(hidden_states, n_rep):
    if n_rep == 1:
        return hidden_states
    bs, n_kv, slen, head_dim = hidden_states.shape
    return (hidden_states[:, :, None].expand(bs, n_kv, n_rep, slen, head_dim)
            .reshape(bs, n_kv * n_rep, slen, head_dim))


def _eager_ref(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key_states = _repeat_kv_ref(key, module.num_key_value_groups)
    value_states = _repeat_kv_ref(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class _Mod:
    """module mock: 只带 attention 函数契约用到的属性."""

    def __init__(self, n_rep, training=False):
        self.num_key_value_groups = n_rep
        self.training = training


def _gen(B, Hq, Hkv, Sq, Skv, D, dtype, seed, device="cpu"):
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(B, Hq, Sq, D, generator=g, device=device, dtype=dtype) * 0.3
    k = torch.randn(B, Hkv, Skv, D, generator=g, device=device, dtype=dtype) * 0.3
    v = torch.randn(B, Hkv, Skv, D, generator=g, device=device, dtype=dtype) * 0.3
    return q, k, v


def _make_mask(B, Sq, Skv, dtype, seed, device="cpu", pad_cols=0):
    """causal 加法 mask(0 可见 / min 不可见), 右侧 pad_cols 列 -inf.

    Sq==Skv: 标准下三角; Sq<Skv: 增量 prefill 语义(行 i 可见前 Skv-Sq+i+1 个 key).
    """
    m = torch.zeros(B, 1, Sq, Skv, device=device, dtype=dtype)
    if Sq == Skv:
        causal = torch.triu(
            torch.ones(Sq, Skv, device=device, dtype=torch.bool), diagonal=1
        )
        m.masked_fill_(causal, torch.finfo(dtype).min)
    else:
        qi = torch.arange(Sq, device=device)[:, None]
        ki = torch.arange(Skv, device=device)[None, :]
        allow = ki <= (Skv - Sq) + qi
        m.masked_fill_(~allow, torch.finfo(dtype).min)
    if pad_cols > 0:
        m[:, :, :, -pad_cols:] = torch.finfo(dtype).min
    return m


def _rel_err(a, b):
    a, b = a.float(), b.float()
    denom = b.abs().max().item()
    if denom == 0:
        return (a - b).abs().max().item()
    return ((a - b).abs().max() / denom).item()


def run_case(name, B, Hq, Hkv, Sq, Skv, D, dtype, tol, seed,
             mask_mode="none", pad_cols=0, training=False, grad=False,
             device="cpu"):
    block = attn_mod._STATE["block"]
    q, k, v = _gen(B, Hq, Hkv, Sq, Skv, D, dtype, seed, device)
    need_grad = grad or training
    if need_grad:
        q = q.requires_grad_(True)
        k = k.requires_grad_(True)
        v = v.requires_grad_(True)

    # mask_mode="none": 被测真传 None(走块内构造三角 mask 路径);
    # 参考与拷贝校验用显式 causal mask(裸 eager 无 mask = 全双向, 语义不同).
    if mask_mode == "none":
        mask_new = None
        mask_ref = (_make_mask(B, Sq, Skv, dtype, seed, device, 0)
                    if (Sq == Skv and Sq > 1) else None)
    else:
        mask_ref = _make_mask(B, Sq, Skv, dtype, seed, device, pad_cols)
        mask_new = mask_ref.clone()

    mod = _Mod(Hq // Hkv, training=training)
    scaling = D ** -0.5

    out_ref, _ = _eager_ref(mod, q, k, v, mask_ref, scaling)
    out_copy, _ = attn_mod.eager_attention_forward(mod, q, k, v, mask_ref, scaling)
    out_new, w_new = attn_mod.chunked_attention_forward(mod, q, k, v, mask_new, scaling)

    d_copy = _rel_err(out_copy, out_ref.detach())
    d_out = _rel_err(out_new, out_ref.detach())

    d_gq = d_gk = d_gv = None
    if need_grad:
        wgt = torch.linspace(0.1, 1.0, out_ref.shape[-1], device=device)
        wgt = wgt.reshape(1, 1, 1, -1).to(dtype)
        (out_ref * wgt).sum().backward(retain_graph=True)
        gq_ref, gk_ref, gv_ref = q.grad.clone(), k.grad.clone(), v.grad.clone()
        for t in (q, k, v):
            t.grad = None
        (out_new * wgt).sum().backward()
        d_gq = _rel_err(q.grad, gq_ref)
        d_gk = _rel_err(k.grad, gk_ref)
        d_gv = _rel_err(v.grad, gv_ref)

    ok = (d_copy < 1e-6) and (d_out < tol)
    if need_grad:
        ok = ok and (d_gq < tol) and (d_gk < tol) and (d_gv < tol)

    def fmt(x):
        return "-" if x is None else f"{x:.2e}"

    detail = (f"copy={d_copy:.2e} out={d_out:.2e} "
              f"gq={fmt(d_gq)} gk={fmt(d_gk)} gv={fmt(d_gv)}")
    return {"case": name, "pass": bool(ok), "detail": detail}


def build_cases():
    """block 统一由 _STATE['block'] 控制(每 case 自己设)."""
    cases = []
    # --- fp32 严格门(1e-5): 语义正确性 ---
    f32 = dict(dtype=torch.float32, tol=1e-5)
    cases.append(dict(name="fp32_4b_square_513_blk128", B=1, Hq=16, Hkv=4,
                      Sq=513, Skv=513, D=256, block=128, seed=11, **f32))
    cases.append(dict(name="fp32_4b_square_2048_blk512", B=1, Hq=16, Hkv=4,
                      Sq=2048, Skv=2048, D=256, block=512, seed=12, **f32))
    cases.append(dict(name="fp32_27b_square_517_blk100", B=1, Hq=24, Hkv=4,
                      Sq=517, Skv=517, D=256, block=100, seed=13, **f32))
    cases.append(dict(name="fp32_causal_mask_513", B=1, Hq=16, Hkv=4,
                      Sq=513, Skv=513, D=256, block=128, seed=14,
                      mask_mode="4d", **f32))
    cases.append(dict(name="fp32_causal_mask_pad64", B=2, Hq=16, Hkv=4,
                      Sq=400, Skv=400, D=256, block=128, seed=15,
                      mask_mode="4d", pad_cols=64, **f32))
    cases.append(dict(name="fp32_cross_100x300_fallback", B=1, Hq=16, Hkv=4,
                      Sq=100, Skv=300, D=256, block=128, seed=16,
                      mask_mode="4d", **f32))
    cases.append(dict(name="fp32_decode_s1_fallback", B=1, Hq=16, Hkv=4,
                      Sq=1, Skv=300, D=256, block=128, seed=17,
                      mask_mode="4d", **f32))
    cases.append(dict(name="fp32_grad_ckpt_600_blk128", B=1, Hq=16, Hkv=4,
                      Sq=600, Skv=600, D=256, block=128, seed=18,
                      training=True, grad=True, **f32))
    cases.append(dict(name="fp32_grad_nockpt_600_blk128", B=1, Hq=16, Hkv=4,
                      Sq=600, Skv=600, D=256, block=128, seed=19,
                      grad=True, **f32))
    # --- fp16 生产形状门(3e-3) ---
    f16 = dict(dtype=torch.float16, tol=3e-3)
    cases.append(dict(name="fp16_4b_513_blk1024", B=1, Hq=16, Hkv=4,
                      Sq=513, Skv=513, D=256, block=1024, seed=21, **f16))
    cases.append(dict(name="fp16_4b_2048_mask_blk1024", B=1, Hq=16, Hkv=4,
                      Sq=2048, Skv=2048, D=256, block=1024, seed=22,
                      mask_mode="4d", **f16))
    cases.append(dict(name="fp16_27b_517_pad_blk512", B=1, Hq=24, Hkv=4,
                      Sq=517, Skv=517, D=256, block=512, seed=23,
                      mask_mode="4d", pad_cols=17, **f16))
    cases.append(dict(name="fp16_grad_ckpt_1024_blk1024", B=1, Hq=16, Hkv=4,
                      Sq=1024, Skv=1024, D=256, block=1024, seed=24,
                      training=True, grad=True, **f16))
    return cases


def run_one(index):
    cases = build_cases()
    c = dict(cases[index])
    attn_mod._STATE["block"] = c["block"]
    attn_mod._STATE["use_ckpt"] = True
    attn_mod._STATE["min_elems"] = 0  # 测试全部走分块路径(除 fallback case)
    c.pop("block")
    try:
        return run_case(**c)
    except Exception as e:  # noqa: BLE001
        return {"case": c["name"], "pass": False,
                "detail": f"EXC {type(e).__name__}: {e}"}


def run_all(device="cpu"):
    if device == "npu":
        import torch_npu  # noqa: F401
        torch.npu.set_device(0)
    results = []
    for i in range(len(build_cases())):
        r = run_one(i)
        results.append(r)
        print(f"[{'PASS' if r['pass'] else 'FAIL'}] {r['case']}: {r['detail']}",
              flush=True)
    n_pass = sum(r["pass"] for r in results)
    print(f"== {n_pass}/{len(results)} PASS ({device}) ==", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "npu"])
    ap.add_argument("--isolate", action="store_true",
                    help="每 case 子进程隔离(NPU 防 TBE 连坐死锁)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.isolate:
        results = []
        here = os.path.abspath(__file__)
        for i, c in enumerate(build_cases()):
            code = (
                "import sys, json; "
                f"sys.path.insert(0, {os.path.dirname(here)!r}); "
                "import test_attn_correctness as t; "
                f"r = t.run_one({i}); "
                "print('__JSON__' + json.dumps(r))"
            )
            p = subprocess.run([sys.executable, "-c", code],
                               capture_output=True, text=True, timeout=1800)
            r = {"case": c["name"], "pass": False, "detail": "no-json"}
            for line in p.stdout.splitlines():
                if "__JSON__" in line:
                    r = json.loads(line.split("__JSON__", 1)[1])
            if not r.get("pass"):
                r["detail"] += (f" | tail={p.stdout[-300:]!r} "
                                f"err={p.stderr[-300:]!r}")
            results.append(r)
            print(f"[{'PASS' if r['pass'] else 'FAIL'}] {r['case']}: {r['detail']}",
                  flush=True)
        n_pass = sum(r["pass"] for r in results)
        print(f"== {n_pass}/{len(results)} PASS (isolate,{args.device}) ==",
              flush=True)
    else:
        results = run_all(args.device)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1, ensure_ascii=False)
    sys.exit(0 if all(r["pass"] for r in results) else 1)


if __name__ == "__main__":
    main()
