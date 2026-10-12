#!/usr/bin/env python3
"""GDN 真实链路 CPU 深度探针: prefill→decode 接力的误差累积(零 NPU).

模拟真实 rollout: chunk prefill(T=4900) 产出 state → recurrent decode 64 步.
此前验证盲区: 各 case 独立对齐过了, 但「mine 满配 prefill 的 state 喂给
mine decode」的真实组合从未测过; real4b + T=4900 的满配对齐也没测过
(CPU 82/82 里 real4b 只有 T=512).

用法: python3 probe_gdn_chain.py [solver scan]  (默认 nsquaring affine)
"""
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

import qwen35_gdn_npu as gdn  # noqa: E402
from test_gdn_correctness import (ref_chunk_gated_delta_rule,  # noqa: E402
                                  ref_recurrent_gated_delta_rule as ref_rec)


def make_inputs(B, T, Hk, Hv, dk, dv, seed=0):
    gen = torch.Generator().manual_seed(seed)
    rep = Hv // Hk
    q = (torch.randn(B, T, Hk, dk, generator=gen) * 1.0).half().repeat_interleave(rep, dim=2)
    k = (torch.randn(B, T, Hk, dk, generator=gen) * 1.0).half().repeat_interleave(rep, dim=2)
    v = (torch.randn(B, T, Hv, dv, generator=gen) * 1.0).half()
    g = -F.softplus(torch.randn(B, T, Hv, generator=gen) * 0.5) * 0.5
    beta = torch.sigmoid(torch.randn(B, T, Hv, generator=gen)).half()
    return q, k, v, g, beta


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-9)).item()


def main():
    solver = sys.argv[1] if len(sys.argv) > 1 else "nsquaring"
    scan = sys.argv[2] if len(sys.argv) > 2 else "affine"
    B, T, Hk, Hv, dk, dv = 1, 4900, 16, 32, 128, 128
    q, k, v, g, beta = make_inputs(B, T, Hk, Hv, dk, dv)
    kw = dict(output_final_state=True, use_qk_l2norm_in_kernel=True)
    gdn._CFG.update(solver=solver, scan=scan, chunk_batch=0,
                    conv_impl="conv1d", use_ckpt=True)

    # ---- 1) prefill 对齐(real4b T=4900 满配, 此前未测) ----
    o_r, s_r = ref_chunk_gated_delta_rule(q, k, v, g, beta, **kw)
    o_m, s_m = gdn.npu_chunk_gated_delta_rule(q, k, v, g, beta, **kw)
    print(f"prefill T={T} out rel={rel(o_m, o_r):.3e} state rel={rel(s_m, s_r):.3e}",
          flush=True)

    # ---- 2) decode 64 步接力(同一起始 state, mine vs ref 各自递推) ----
    qd, kd, vd, gd, bd = (t[:, -64:] for t in (q, k, v, g, beta))
    kw2 = dict(output_final_state=True, use_qk_l2norm_in_kernel=True)
    o_r2, s_r2 = ref_rec(qd, kd, vd, gd, bd, initial_state=s_r, **kw2)
    o_m2, s_m2 = gdn.npu_recurrent_gated_delta_rule(qd, kd, vd, gd, bd,
                                                    initial_state=s_m, **kw2)
    print(f"decode64 out rel={rel(o_m2, o_r2):.3e} state rel={rel(s_m2, s_r2):.3e}",
          flush=True)

    # ---- 3) 误差放大链: 一次 chunk vs 分两段接力(切点=4836, 与第二段严格相接) ----
    cut = T - 64
    o_all, s_all = gdn.npu_chunk_gated_delta_rule(q, k, v, g, beta, **kw)
    _, s1 = gdn.npu_chunk_gated_delta_rule(q[:, :cut], k[:, :cut], v[:, :cut],
                                           g[:, :cut], beta[:, :cut], **kw)
    o2, s2 = gdn.npu_chunk_gated_delta_rule(qd, kd, vd, gd, bd,
                                            initial_state=s1, **kw)
    print(f"chain state rel={rel(s2, s_all):.3e} tail-out rel={rel(o2, o_all[:, cut:]):.3e}",
          flush=True)

    # ---- 4) decode 步进误差累积曲线(state 每 16 步 vs ref) ----
    st_m, st_r = s_m.clone(), s_r.clone()
    for t in range(64):
        q_t = qd[:, t:t + 1]
        _, st_m = gdn.npu_recurrent_gated_delta_rule(
            q_t, kd[:, t:t + 1], vd[:, t:t + 1], gd[:, t:t + 1], bd[:, t:t + 1],
            initial_state=st_m, **kw2)
        _, st_r = ref_rec(q_t, kd[:, t:t + 1], vd[:, t:t + 1], gd[:, t:t + 1],
                          bd[:, t:t + 1], initial_state=st_r, **kw2)
        if (t + 1) % 16 == 0:
            print(f"  step {t + 1:3d}: state rel={rel(st_m, st_r):.3e}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
