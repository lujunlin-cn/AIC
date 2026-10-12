#!/usr/bin/env python3
"""M1 基准: chunk/decode/conv 三热点计时(等空卡后跑 NPU, 也可 CPU 冒烟).

用法:
    python3 bench_gdn.py --suite chunk --device npu --model 4b --T 8192
    python3 bench_gdn.py --suite decode --device npu --model 27b
    python3 bench_gdn.py --suite conv --device npu --model 4b
输出: 每配置 µs/调用 + (NPU)显存峰值.
"""
import argparse
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
import qwen35_gdn_npu as gdn  # noqa: E402
import test_gdn_correctness as t  # noqa: E402  (复用参考实现与造数)

MODELS = {
    "4b": dict(Hk=16, Hv=32, dk=128, dv=128),
    "27b": dict(Hk=16, Hv=48, dk=128, dv=128),
}

_DEV = "cpu"  # main() 里设置, suite 函数读全局


def sync(device):
    if device == "npu":
        torch.npu.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def bench(fn, n_warm=3, n_iter=10):
    for _ in range(n_warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    return (time.perf_counter() - t0) / n_iter


def ref_chunk_compat(query, key, value, g, beta, chunk_size=64,
                     initial_state=None, output_final_state=False,
                     use_qk_l2norm_in_kernel=False, **kwargs):
    """参考实现(safe_chunk 数学)的 NPU 兼容基线: 仅把 scan 里
    key^T @ v_new(4 维 transpose matmul, 触发 TBE BatchMatMul 编译 hang)
    改为 .contiguous() 后的等价 matmul. 每 Chunk 多一次拷贝, 对参考实现
    是轻微劣化(方向保守)."""
    import torch.nn.functional as F
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    recurrent_state_shape = (batch_size, num_v_heads, k_head_dim, v_head_dim)
    padded_output_shape = (batch_size, num_v_heads, -1, v_head_dim)
    decay = g
    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, decay)
    ]
    if use_qk_l2norm_in_kernel:
        query, key = t._ref_l2norm(query), t._ref_l2norm(key)
    query = query * (query.shape[-1] ** -0.5)
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query, key, value = (F.pad(x, (0, 0, 0, pad_size)) for x in (query, key, value))
    beta, decay = (F.pad(x, (0, pad_size)) for x in (beta, decay))
    num_chunks = (sequence_length + pad_size) // chunk_size
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, k_beta, v_beta)
    ]
    decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)
    mask = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(1)
    cum_decay = decay.cumsum(dim=3)
    pairwise_decay = (cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)).exp()
    pairwise_decay = pairwise_decay.masked_fill(mask, 0.0)
    ut_system = (k_beta @ key.transpose(-1, -2)) * pairwise_decay
    intra_chunk_attn = (query @ key.transpose(-1, -2)) * pairwise_decay
    decayed_k_beta = k_beta * cum_decay.exp().unsqueeze(-1)
    new_values = torch.linalg.solve_triangular(ut_system, v_beta, upper=False, unitriangular=True)
    k_cumdecay = torch.linalg.solve_triangular(ut_system, decayed_k_beta, upper=False, unitriangular=True)
    state = (torch.zeros(recurrent_state_shape, dtype=new_values.dtype, device=new_values.device)
             if initial_state is None else initial_state.to(new_values))
    core_attn_out = torch.zeros_like(new_values)
    query = query * cum_decay.exp().unsqueeze(-1)
    key = key * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
    chunk_decay = cum_decay[..., -1].exp()[..., None, None]
    for i in range(num_chunks):
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ state
        inter_chunk_attn = query[:, :, i] @ state
        core_attn_out[:, :, i] = inter_chunk_attn + intra_chunk_attn[:, :, i] @ v_new
        state = (state * chunk_decay[:, :, i]
                 + key[:, :, i].transpose(-1, -2).contiguous() @ v_new)  # 唯一改动
    state = state if output_final_state else None
    core_attn_out = core_attn_out.reshape(padded_output_shape)[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).to(
        initial_dtype, memory_format=torch.contiguous_format)
    return core_attn_out, state


def suite_chunk(args, dev, md):
    B, T = args.B, args.T
    q, k, v, g, beta = t.make_inputs(B, T, md["Hk"], md["Hv"], md["dk"], md["dv"], dev, args.seed)
    kw = dict(output_final_state=True, use_qk_l2norm_in_kernel=True)
    rows = []
    for solver in ("triangular", "nsquaring"):
        for scan in ("seq", "affine"):
            for cb in (0, 8, 16):
                gdn._CFG.update(solver=solver, scan=scan, chunk_batch=cb)
                try:
                    if _DEV == "npu":
                        torch.npu.reset_peak_memory_stats()
                    dt = bench(lambda: gdn.npu_chunk_gated_delta_rule(q, k, v, g, beta, **kw))
                    peak = torch.npu.max_memory_allocated() / 2**30 if _DEV == "npu" else None
                    rows.append((f"npu solver={solver[:3]} scan={scan} cb={cb}", dt, peak))
                except Exception as e:  # noqa: BLE001
                    rows.append((f"npu solver={solver[:3]} scan={scan} cb={cb}", None, f"{type(e).__name__}: {e}"))
    if not args.skip_ref:
        try:
            dt = bench(lambda: ref_chunk_compat(q, k, v, g, beta, **kw))
            rows.append(("ref(safe_chunk 兼容版)", dt, None))
        except Exception as e:  # noqa: BLE001
            rows.append(("ref(safe_chunk 兼容版)", None, f"{type(e).__name__}: {e}"))
    return rows


def suite_decode(args, dev, md):
    rows = []
    for B in (1, 4, 8):
        for T in (1, 7):
            q, k, v, g, beta = t.make_inputs(B, T, md["Hk"], md["Hv"], md["dk"], md["dv"], dev, args.seed)
            st = (torch.randn(B, md["Hv"], md["dk"], md["dv"]) * 0.1).to(dev, torch.float32)
            kw = dict(initial_state=st, output_final_state=True, use_qk_l2norm_in_kernel=True)
            dt_m = bench(lambda: gdn.npu_recurrent_gated_delta_rule(q, k, v, g, beta, **kw), 3, 20)
            rows.append((f"decode B={B} T={T} npu", dt_m, None))
            if not args.skip_ref:
                dt_r = bench(lambda: t.ref_recurrent_gated_delta_rule(q, k, v, g, beta, **kw), 3, 20)
                rows.append((f"decode B={B} T={T} ref", dt_r, None))
    return rows


def suite_conv(args, dev, md):
    C = md["Hk"] * md["dk"] * 2 + md["Hv"] * md["dv"]
    rows = []
    for B in (1,):
        for T in (1024, 4096, args.T if args.T > 4096 else 8192):
            gen = torch.Generator().manual_seed(1)
            x = torch.randn(B, C, T, generator=gen).to(dev, torch.float16)
            w = (torch.randn(C, 4, generator=gen) * 0.5).to(dev, torch.float16)
            b = (torch.randn(C, generator=gen) * 0.1).to(dev, torch.float16)
            gdn._CFG.update(conv_impl="conv1d")
            dt_c = bench(lambda: gdn.npu_causal_conv1d_fn(x, w, b, activation="silu"))
            gdn._CFG.update(conv_impl="unfold")
            dt_u = bench(lambda: gdn.npu_causal_conv1d_fn(x, w, b, activation="silu"))
            rows.append((f"conv_fn B={B} C={C} T={T} conv1d", dt_c, None))
            rows.append((f"conv_fn B={B} C={C} T={T} unfold", dt_u, None))
            if not args.skip_ref:
                dt_r = bench(lambda: t.ref_causal_conv1d_fn(x, w, b, activation="silu"))
                rows.append((f"conv_fn B={B} C={C} T={T} ref", dt_r, None))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, choices=["chunk", "decode", "conv", "all"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "npu"])
    ap.add_argument("--model", default="4b", choices=list(MODELS))
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--T", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-ref", action="store_true", help="只跑 mine(参考系在 NPU 有编译 hang 风险)")
    args = ap.parse_args()
    global _DEV
    _DEV = args.device
    if _DEV == "npu":
        import torch_npu  # noqa: F401
        torch.npu.set_device(0)
    device = args.device
    dev = device if device != "npu" else "npu"
    md = MODELS[args.model]
    print(f"model={args.model}({md}) B={args.B} T={args.T} device={device}", flush=True)

    suites = [args.suite] if args.suite != "all" else ["chunk", "decode", "conv"]
    for s in suites:
        print(f"\n=== {s} ===", flush=True)
        rows = globals()[f"suite_{s}"](args, dev, md)
        for name, dt, extra in rows:
            if dt is None:
                print(f"  {name:44s} FAILED {extra}")
            else:
                peak = f"  peak={extra:.2f}GiB" if extra is not None else ""
                print(f"  {name:44s} {dt * 1e6:12.1f} us{peak}", flush=True)


if __name__ == "__main__":
    main()
