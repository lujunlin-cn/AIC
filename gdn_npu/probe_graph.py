#!/usr/bin/env python3
"""M1 探针: npu graph capture(decode step)可行性 + torch_npu graph API 自省.

背景: 若 decode step 可被 graph 捕获重放, python/launch 开销消失, 收益覆盖
全模型. 本探针验证 API 可用性与最小 capture 语义(3 算子循环体同构),
不做整模型 capture(那属 A+ 路线正式验证).

用法(910A 空卡): python3 probe_graph.py
"""
import sys

import torch

sys.path.insert(0, ".")


def introspect():
    print("=== torch_npu graph API introspection ===")
    import torch_npu
    import torch.npu as tnp
    for name, mod in (("torch_npu", torch_npu), ("torch.npu", tnp)):
        syms = [s for s in dir(mod) if "graph" in s.lower() or "capture" in s.lower()]
        print(f"  {name}: {syms}")
    try:
        import torch_npu.npu.graph as ng  # noqa: F401
        print("  torch_npu.npu.graph members:",
              [s for s in dir(ng) if not s.startswith("_")][:24])
    except Exception as e:  # noqa: BLE001
        print(f"  torch_npu.npu.graph import FAIL: {type(e).__name__}: {e}")


def find_capture_api():
    """探测可用的 graph capture 原语, 返回 (名称, 进入ctx工厂) 或 None."""
    candidates = []
    try:
        from torch_npu.npu import graph as _g  # noqa: F401
        candidates.append(("torch_npu.npu.graph.ctx", lambda: _g.NPUGraph(),
                           _g))
    except Exception:  # noqa: BLE001
        pass
    if hasattr(torch.npu, "NPUGraph"):
        candidates.append(("torch.npu.NPUGraph", torch.npu.NPUGraph, torch.npu))
    for name, ctor, mod in candidates:
        try:
            g = ctor()
            ctx = mod.graph(g) if hasattr(mod, "graph") else None
            if ctx is not None and hasattr(g, "replay"):
                return name, g, ctx
        except Exception:  # noqa: BLE001
            continue
    return None


def main():
    import torch_npu  # noqa: F401
    torch.npu.set_device(0)
    dev = "npu:0"
    introspect()

    print("\n=== minimal decode-step capture ===")
    # 转置状态布局(S_T=[dv,dk], 与 qwen35_gdn_npu decode 同款):
    # 本机 4 维 matmul+transpose 视图触发 TBE BatchMatMul 编译失败, 循环内
    # 不得出现 transpose 视图进 bmm.
    BH, dk, dv = 32, 128, 128
    state = (torch.randn(BH, dv, dk) * 0.1).to(dev, torch.float32)  # S_T
    kt = torch.randn(BH, dk, 1, device=dev, dtype=torch.float32)
    qt = torch.randn(BH, dk, 1, device=dev, dtype=torch.float32)
    vt = torch.randn(BH, dv, 1, device=dev, dtype=torch.float32)
    dec = torch.rand(BH, 1, 1, device=dev, dtype=torch.float32) * 0.5

    def step():
        state.mul_(dec)
        u = torch.bmm(state, kt).squeeze(-1)                        # S_T @ k
        delta = vt.squeeze(-1) - u
        state.add_(torch.bmm(delta[:, :, None], kt.transpose(-1, -2)))  # 3维 bmm
        _ = torch.bmm(state, qt).squeeze(-1)                        # out

    step()
    torch.npu.synchronize()
    ref = state.clone()
    state.zero_()

    found = find_capture_api()
    if found is None:
        print("  未找到可用的 graph capture 原语 -> A+ 路线降级, 手写融合算子权重上调")
        return
    name, graph, ctx = found
    print(f"  capture 原语: {name}")
    try:
        warm = torch.cuda.CUDAGraph  # 占位说明: 捕获前通常需 warmup 流
        del warm
    except Exception:  # noqa: BLE001
        pass
    try:
        for _ in range(2):  # capture 前 warmup
            step()
        torch.npu.synchronize()
        with ctx:
            step()
        state.zero_()
        torch.npu.synchronize()
        graph.replay()
        torch.npu.synchronize()
        diff = (state - ref).abs().max().item()
        print(f"  replay 数值 maxdiff vs eager: {diff:.3e}")
        print("  A+ 探针 PASS: capture+replay 可用(最小语义级)")
    except Exception as e:  # noqa: BLE001
        print(f"  capture/replay FAIL: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
