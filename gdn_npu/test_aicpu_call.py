#!/usr/bin/env python3
"""GdnDecodeStep om 的 torch 侧数值验证 + decode 基准(对照 torch 向量化实现).

用法(910A 空卡): python3 test_aicpu_call.py [--bench]
"""
import argparse
import sys
import time

import torch

sys.path.insert(0, "/root/AIC/gdn_npu")
from aicpu_call import GdnDecodeStepOm  # noqa: E402


def torch_ref(q, k, v, g, beta, state):
    """与 kernel 同数学的 torch 参考(BH 合维, 转置布局同 npu_recurrent)."""
    B, Hv, dk = q.shape
    dv = v.shape[-1]
    qf = q.float().reshape(B * Hv, dk)
    kf = k.float().reshape(B * Hv, dk)
    vf = v.float().reshape(B * Hv, dv)
    st = state.float().reshape(B * Hv, dk, dv).transpose(-1, -2).contiguous()
    st = st * g.float().reshape(B * Hv, 1, 1).exp()
    u = torch.bmm(st, kf[:, :, None]).squeeze(-1)
    delta = (vf - u) * beta.float().reshape(B * Hv, 1)
    st = st + torch.bmm(delta[:, :, None], kf[:, None, :])
    out = torch.bmm(st, qf[:, :, None]).squeeze(-1)
    return (out.reshape(B, Hv, dv).half(),
            st.transpose(-1, -2).reshape(B, Hv, dk, dv))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--om-dir", default="/root/AIC/gdn_npu/om")
    args = ap.parse_args()

    import torch_npu  # noqa: F401
    torch.npu.set_device(0)
    dev = "npu:0"
    print("npu_stream attr:", torch.npu.current_stream().npu_stream, flush=True)
    op = GdnDecodeStepOm(args.om_dir)  #尽早注册模型目录
    print("SetModelDir ok", flush=True)

    Hv, dk, dv = 32, 128, 128
    gen = torch.Generator().manual_seed(0)
    for B in (1, 8):
        q = (torch.randn(B, Hv, dk, generator=gen) * 0.3).half().to(dev)
        k = (torch.randn(B, Hv, dk, generator=gen) * 0.3).half().to(dev)
        v = (torch.randn(B, Hv, dv, generator=gen) * 0.3).half().to(dev)
        g = (-torch.rand(B, Hv, generator=gen) - 0.1).float().to(dev)
        beta = (torch.rand(B, Hv, generator=gen) * 0.8 + 0.1).float().to(dev)
        st = (torch.randn(B, Hv, dk, dv, generator=gen) * 0.1).float().to(dev)

        out, ns = op.run(q, k, v, g, beta, st)
        torch.npu.synchronize()

        o_ref, s_ref = torch_ref(q.cpu(), k.cpu(), v.cpu(), g.cpu(), beta.cpu(), st.cpu())
        d_out = (out.cpu().float() - o_ref.float()).abs().max().item()
        d_st = (ns.cpu() - s_ref).abs().max().item()
        print(f"B={B} out_diff={d_out:.3e} state_diff={d_st:.3e} "
              f"-> {'MATCH' if d_out < 5e-3 and d_st < 5e-5 else 'MISMATCH'}", flush=True)

        if args.bench and B == 1:
            sys.path.insert(0, "/root/AIC/gdn_npu")
            import qwen35_gdn_npu as gdn
            # npu_recurrent 接口为 [B,T,Hv,dk/dv] 布局(T=1 即 decode 单步)
            qt, kt = q.unsqueeze(1), k.unsqueeze(1)
            vt = v.unsqueeze(1)
            gt, bt = g.unsqueeze(1), beta.unsqueeze(1)
            kw = dict(initial_state=st, output_final_state=False,
                      use_qk_l2norm_in_kernel=False)

            def fn_vec():
                gdn.npu_recurrent_gated_delta_rule(qt, kt, vt, gt, bt, **kw)

            op.bind(q, k, v, g, beta, st)

            def fn_cached():
                op.run_cached()

            for name, fn, n in (("om-once", lambda: op.run(q, k, v, g, beta, st), 50),
                                ("om-cached", fn_cached, 200),
                                ("torch-vector", fn_vec, 200)):
                for _ in range(10):
                    fn()
                torch.npu.synchronize()
                t0 = time.perf_counter()
                for _ in range(n):
                    fn()
                torch.npu.synchronize()
                dt = (time.perf_counter() - t0) / n
                print(f"bench {name:12s} B={B}: {dt * 1e6:9.1f} us", flush=True)


if __name__ == "__main__":
    main()
