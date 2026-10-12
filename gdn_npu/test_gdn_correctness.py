#!/usr/bin/env python3
"""GDN NPU 实现的数值对齐测试(自包含, CPU 可跑, 零 NPU 占用).

对照组 = transformers 5.19 modeling_qwen3_5 回退函数的逐字拷贝
        (chunk 为 r13_qwen35_safe_chunk 修复版, 即当前生产路径).

用法:
    python3 test_gdn_correctness.py                 # CPU 全量
    python3 test_gdn_correctness.py --quick         # 跳过长 case
    python3 test_gdn_correctness.py --device npu    # 910A 复测(等空卡)
退出码: 0 = 全 PASS, 1 = 有 FAIL.
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen35_gdn_npu as gdn  # noqa: E402

_ACT2FN = {"silu": F.silu, "gelu": F.gelu}


# ======================================================================
# 参考实现(transformers 5.19 逐字拷贝, 仅 ACT2FN 换映射、dynamo 分支删除)
# ======================================================================

def _ref_l2norm(x, dim=-1, eps=1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def ref_chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64,
                               initial_state=None, output_final_state=False,
                               use_qk_l2norm_in_kernel=False, **kwargs):
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
        query = _ref_l2norm(query, dim=-1, eps=1e-6)
        key = _ref_l2norm(key, dim=-1, eps=1e-6)
    scaling = query.shape[-1] ** -0.5
    query = query * scaling

    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query, key, value = (F.pad(x, (0, 0, 0, pad_size)) for x in (query, key, value))
    beta, decay = (F.pad(x, (0, pad_size)) for x in (beta, decay))

    total_sequence_length = sequence_length + pad_size
    num_chunks = total_sequence_length // chunk_size

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    query, key, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, k_beta, v_beta)
    ]
    decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)

    strictly_upper_mask = torch.ones(
        chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(1)
    cum_decay = decay.cumsum(dim=3)

    # safe_chunk 修复: exp 先行(finite), 再 masked_fill(0)
    pairwise_decay = (cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)).exp()
    pairwise_decay = pairwise_decay.masked_fill(strictly_upper_mask, 0.0)

    ut_system = (k_beta @ key.transpose(-1, -2)) * pairwise_decay
    intra_chunk_attn = (query @ key.transpose(-1, -2)) * pairwise_decay
    decayed_k_beta = k_beta * cum_decay.exp().unsqueeze(-1)

    new_values = torch.linalg.solve_triangular(
        ut_system, v_beta, upper=False, unitriangular=True)
    k_cumdecay = torch.linalg.solve_triangular(
        ut_system, decayed_k_beta, upper=False, unitriangular=True)

    if initial_state is None:
        last_recurrent_state = torch.zeros(
            recurrent_state_shape, dtype=new_values.dtype, device=new_values.device)
    else:
        last_recurrent_state = initial_state.to(new_values)
    core_attn_out = torch.zeros_like(new_values)

    query = query * cum_decay.exp().unsqueeze(-1)
    key = key * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
    chunk_decay = cum_decay[..., -1].exp()[..., None, None]

    for i in range(num_chunks):
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ last_recurrent_state
        inter_chunk_attn = query[:, :, i] @ last_recurrent_state
        core_attn_out[:, :, i] = inter_chunk_attn + intra_chunk_attn[:, :, i] @ v_new
        last_recurrent_state = (last_recurrent_state * chunk_decay[:, :, i]
                                + key[:, :, i].transpose(-1, -2) @ v_new)

    last_recurrent_state = None if not output_final_state else last_recurrent_state
    core_attn_out = core_attn_out.reshape(padded_output_shape)
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).to(
        initial_dtype, memory_format=torch.contiguous_format)
    return core_attn_out, last_recurrent_state


def ref_recurrent_gated_delta_rule(query, key, value, g, beta,
                                   initial_state=None, output_final_state=False,
                                   use_qk_l2norm_in_kernel=False, **kwargs):
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    decay = g

    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, decay)
    ]
    if use_qk_l2norm_in_kernel:
        query = _ref_l2norm(query, dim=-1, eps=1e-6)
        key = _ref_l2norm(key, dim=-1, eps=1e-6)
    query = query / (query.shape[-1] ** 0.5)

    if initial_state is None:
        recurrent_state_shape = (batch_size, num_v_heads, k_head_dim, v_head_dim)
        last_recurrent_state = torch.zeros(
            recurrent_state_shape, dtype=value.dtype, device=value.device)
    else:
        last_recurrent_state = initial_state.to(value)
    core_attn_out = torch.zeros_like(value)

    for i in range(sequence_length):
        q_t, k_t, v_t = query[:, :, i], key[:, :, i], value[:, :, i]
        decay_t = decay[:, :, i].exp()[..., None, None]
        last_recurrent_state = last_recurrent_state * decay_t
        beta_t = beta[:, :, i].unsqueeze(-1)
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    last_recurrent_state = None if not output_final_state else last_recurrent_state
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def ref_causal_conv1d_update(hidden_states, conv_state, weight, bias=None, activation=None):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = out[:, :, -seq_len:]
    if activation is not None:
        out = _ACT2FN[activation](out)
    return out.to(hidden_states.dtype)


def ref_causal_conv1d_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
    _, hidden_size, seq_len = hidden_states.shape
    padding = weight.shape[-1] - 1
    out = F.conv1d(
        hidden_states.to(weight.dtype), weight=weight.unsqueeze(1), bias=bias,
        padding=padding, groups=hidden_size,
    )[:, :, :seq_len]
    if activation is not None:
        out = _ACT2FN[activation](out)
    return out.to(hidden_states.dtype)


# ======================================================================
# 测试件
# ======================================================================

def make_inputs(B, T, Hk, Hv, dk, dv, device, seed=0):
    """按真实调用契约造数: q/k/v/beta fp16, g fp32(A_log.float()).

    契约: 调用点(Qwen3_5GatedDeltaNet.forward)在进 kernel 前已把 q/k
    repeat_interleave 到 Hv 个头, kernel 内 q/k/v 全部 Hv 头.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    def mk(*shape, dtype):
        return torch.randn(*shape, generator=gen).to(device=device, dtype=dtype)
    rep = Hv // Hk
    q = mk(B, T, Hk, dk, dtype=torch.float16).repeat_interleave(rep, dim=2)
    k = mk(B, T, Hk, dk, dtype=torch.float16).repeat_interleave(rep, dim=2)
    v = mk(B, T, Hv, dv, dtype=torch.float16)
    g = (-F.softplus(mk(B, T, Hv, dtype=torch.float32) * 0.5) * 0.5)
    beta = torch.sigmoid(mk(B, T, Hv, dtype=torch.float16))
    return q, k, v, g, beta


def metrics(a, b):
    if a.device != b.device:
        a = a.cpu()
    a32, b32 = a.float(), b.float()
    diff = a32 - b32
    rel = (diff.norm() / b32.norm().clamp_min(1e-12)).item()
    return rel, diff.abs().max().item()


RESULTS = []


def record(name, got, want, gate_rel, note=""):
    rel, mx = metrics(got, want)
    ok = rel <= gate_rel
    RESULTS.append((name, rel, mx, gate_rel, ok, note))
    return ok


def cfg(**kw):
    gdn._CFG.update(**kw)


# ---------------------------------------------------------------- chunk

def _to_dev(*xs, device="cpu"):
    """ref-on-cpu 模式: 参考实现固定 CPU 跑黄金标准, mine 上目标设备.
    本机 4 维 matmul+transpose 视图会触发 TBE BatchMatMul 编译失败,
    参考实现(含该调用形态)不得在 NPU 上执行."""
    if device == "cpu":
        return xs
    return tuple(x.to(device) for x in xs)


def case_chunk(tag, B, T, Hk, Hv, dk, dv, solver="triangular", scan="seq",
               chunk_batch=0, init=False, gate=3e-3, seed=0, device="cpu",
               l2norm=True):
    q, k, v, g, beta = make_inputs(B, T, Hk, Hv, dk, dv, "cpu", seed)
    st = torch.randn(B, Hv, dk, dv, generator=torch.Generator().manual_seed(seed + 1))
    st = (st * 0.1).to(dtype=torch.float32) if init else None
    kw = dict(output_final_state=True, use_qk_l2norm_in_kernel=l2norm)
    o_r, s_r = ref_chunk_gated_delta_rule(q, k, v, g, beta, initial_state=st, **kw)
    cfg(solver=solver, scan=scan, chunk_batch=chunk_batch)
    ins = _to_dev(q, k, v, g, beta, device=device)
    st_m = None if st is None else st.to(device)
    o_m, s_m = gdn.npu_chunk_gated_delta_rule(*ins, initial_state=st_m, **kw)
    ok = record(f"chunk/{tag}/out", o_m, o_r, gate)
    ok &= record(f"chunk/{tag}/state", s_m, s_r, 1e-3)
    return ok


def case_decode(tag, B, T, Hk, Hv, dk, dv, seed=0, device="cpu"):
    q, k, v, g, beta = make_inputs(B, T, Hk, Hv, dk, dv, "cpu", seed)
    st = torch.randn(B, Hv, dk, dv, generator=torch.Generator().manual_seed(seed + 1))
    st = (st * 0.1).to(dtype=torch.float32)
    kw = dict(output_final_state=True, use_qk_l2norm_in_kernel=True)
    o_r, s_r = ref_recurrent_gated_delta_rule(q, k, v, g, beta, initial_state=st, **kw)
    ins = _to_dev(q, k, v, g, beta, device=device)
    st_m = st.to(device)
    o_m, s_m = gdn.npu_recurrent_gated_delta_rule(*ins, initial_state=st_m, **kw)
    ok = record(f"decode/{tag}/out", o_m, o_r, 3e-3)
    ok &= record(f"decode/{tag}/state", s_m, s_r, 1e-3)
    return ok


def case_ref_crosscheck(device="cpu"):
    """对照组内部一致性: ref_recurrent vs ref_chunk(T 整除 64 时语义等价)."""
    B, T, Hk, Hv, dk, dv = 1, 128, 2, 4, 8, 16
    q, k, v, g, beta = make_inputs(B, T, Hk, Hv, dk, dv, device, seed=7)
    kw = dict(output_final_state=True, use_qk_l2norm_in_kernel=True)
    o_c, s_c = ref_chunk_gated_delta_rule(q, k, v, g, beta, **kw)
    o_r, s_r = ref_recurrent_gated_delta_rule(q, k, v, g, beta, **kw)
    # 门注: 对照组内部一致性检查; 910A 上 OMP 线程数不同会带来 ~5e-4 的 CPU
    # 求和顺序差异, 故门放宽到与其他 case 同级(非 mine 正确性依据).
    ok = record("ref_cross/out", o_r.float(), o_c.float(), 2e-3)
    ok &= record("ref_cross/state", s_r, s_c, 1e-3)
    return ok


def case_prefill_handoff(device="cpu", gate=3e-3):
    """prefill(T=512) 产出的 state 交给 decode 下一个 token, 应等于 chunk(T=513) 末位输出."""
    B, Hk, Hv, dk, dv = 1, 16, 32, 128, 128
    q, k, v, g, beta = make_inputs(B, 513, Hk, Hv, dk, dv, "cpu", seed=3)
    kw = dict(output_final_state=True, use_qk_l2norm_in_kernel=True)
    _, s_pre = ref_chunk_gated_delta_rule(q[:, :512], k[:, :512], v[:, :512],
                                          g[:, :512], beta[:, :512], **kw)
    o_dec, s_dec = ref_recurrent_gated_delta_rule(
        q[:, 512:], k[:, 512:], v[:, 512:], g[:, 512:], beta[:, 512:],
        initial_state=s_pre, **kw)
    cfg(solver="triangular", scan="seq", chunk_batch=0)
    qi, ki, vi, gi, bi = _to_dev(q, k, v, g, beta, device=device)
    _, s_pre_m = gdn.npu_chunk_gated_delta_rule(
        qi[:, :512], ki[:, :512], vi[:, :512], gi[:, :512], bi[:, :512], **kw)
    o_dec_m, s_dec_m = gdn.npu_recurrent_gated_delta_rule(
        qi[:, 512:], ki[:, 512:], vi[:, 512:], gi[:, 512:], bi[:, 512:],
        initial_state=s_pre_m, **kw)
    ok = record("handoff/out", o_dec_m, o_dec, gate)
    ok &= record("handoff/state_vs_ref_decode", s_dec_m, s_dec, 1e-3)
    return ok


def case_state_chain(device="cpu"):
    """chunk 两次(128+128, 传 state) 的终态应等于一次 chunk(256)."""
    B, T, Hk, Hv, dk, dv = 1, 128, 2, 4, 8, 16
    q, k, v, g, beta = make_inputs(B, 2 * T, Hk, Hv, dk, dv, "cpu", seed=5)
    kw = dict(output_final_state=True, use_qk_l2norm_in_kernel=True)
    _, s_all = ref_chunk_gated_delta_rule(q, k, v, g, beta, **kw)
    _, s1 = ref_chunk_gated_delta_rule(q[:, :T], k[:, :T], v[:, :T], g[:, :T],
                                       beta[:, :T], **kw)
    _, s2 = ref_chunk_gated_delta_rule(q[:, T:], k[:, T:], v[:, T:], g[:, T:],
                                       beta[:, T:], initial_state=s1, **kw)
    cfg(solver="triangular", scan="seq", chunk_batch=0)
    qi, ki, vi, gi, bi = _to_dev(q, k, v, g, beta, device=device)
    _, m1 = gdn.npu_chunk_gated_delta_rule(qi[:, :T], ki[:, :T], vi[:, :T], gi[:, :T],
                                           bi[:, :T], **kw)
    _, m2 = gdn.npu_chunk_gated_delta_rule(qi[:, T:], ki[:, T:], vi[:, T:], gi[:, T:],
                                           bi[:, T:], initial_state=m1, **kw)
    ok = record("chain/state_2x128_vs_256(mine)", m2, s_all, 1e-3)
    ok &= record("chain/state_2x128_vs_256(ref)", s2, s_all, 1e-5)
    return ok


# ---------------------------------------------------------------- conv

def case_conv_fn(tag, B, C, T, K, impl, device="cpu", gate=2e-3):
    gen = torch.Generator().manual_seed(11)
    x = torch.randn(B, C, T, generator=gen).to(torch.float16)
    w = (torch.randn(C, K, generator=gen) * 0.5).to(torch.float16)
    b = (torch.randn(C, generator=gen) * 0.1).to(torch.float16)
    o_r = ref_causal_conv1d_fn(x, w, b, activation="silu")  # CPU 黄金
    cfg(conv_impl=impl)
    xi, wi, bi = _to_dev(x, w, b, device=device)
    o_m = gdn.npu_causal_conv1d_fn(xi, wi, bi, activation="silu")
    return record(f"conv_fn/{tag}/{impl}", o_m, o_r, gate)


def case_conv_update(tag, B, C, K, SL, L, device="cpu", gate=2e-3):
    gen = torch.Generator().manual_seed(13)
    x = torch.randn(B, C, L, generator=gen).to(torch.float16)
    st = torch.randn(B, C, SL, generator=gen).to(torch.float16)
    w = (torch.randn(C, K, generator=gen) * 0.5).to(torch.float16)
    b = (torch.randn(C, generator=gen) * 0.1).to(torch.float16)
    st_r = st.clone()
    o_r = ref_causal_conv1d_update(x, st_r, w, b, activation="silu")  # CPU 黄金
    st_m = st.clone().to(device)
    xi, wi, bi = _to_dev(x, w, b, device=device)
    o_m = gdn.npu_causal_conv1d_update(xi, st_m, wi, bi, activation="silu")
    ok = record(f"conv_update/{tag}/out", o_m, o_r, gate)
    ok &= record(f"conv_update/{tag}/state", st_m.float(), st_r.float(), 0)
    return ok


# ======================================================================
# 主流程
# ======================================================================

def build_cases(device, quick):
    """case 注册表: (名字, 执行闭包)."""
    cases = [
        ("ref_crosscheck", lambda: case_ref_crosscheck(device)),
        ("chunk_small_seq", lambda: case_chunk(
            "b1t128", 1, 128, 2, 4, 8, 16, solver="triangular", scan="seq", device=device)),
        ("chunk_small_nsquaring", lambda: case_chunk(
            "b1t128", 1, 128, 2, 4, 8, 16, solver="nsquaring", scan="seq", device=device)),
        ("chunk_small_affine", lambda: case_chunk(
            "b1t128", 1, 128, 2, 4, 8, 16, solver="triangular", scan="affine", device=device)),
        ("chunk_small_affine_nsq", lambda: case_chunk(
            "b1t128", 1, 128, 2, 4, 8, 16, solver="nsquaring", scan="affine", device=device)),
        ("chunk_nonmul64", lambda: case_chunk(
            "b3t100", 3, 100, 2, 4, 8, 16, device=device)),
        ("chunk_init_state", lambda: case_chunk(
            "init", 2, 256, 2, 4, 8, 16, init=True, device=device)),
        ("chunk_batch4_affine", lambda: case_chunk(
            "b1t1000cb4", 1, 1000, 2, 4, 8, 16, scan="affine", chunk_batch=4, device=device)),
        ("chunk_batch4_seq", lambda: case_chunk(
            "b1t1000cb4", 1, 1000, 2, 4, 8, 16, scan="seq", chunk_batch=4, device=device)),
        ("chunk_real4b", lambda: case_chunk(
            "4b_t512", 1, 512, 16, 32, 128, 128, device=device)),
        ("chunk_real4b_affine", lambda: case_chunk(
            "4b_t512", 1, 512, 16, 32, 128, 128, scan="affine", chunk_batch=8, device=device)),
        ("chunk_real27b", lambda: case_chunk(
            "27b_t512", 1, 512, 16, 48, 128, 128, device=device)),
        ("chunk_real27b_affine", lambda: case_chunk(
            "27b_t512", 1, 512, 16, 48, 128, 128, scan="affine", chunk_batch=8, device=device)),
        ("decode_t1", lambda: case_decode("t1", 2, 1, 2, 4, 8, 16, device=device)),
        ("decode_t7", lambda: case_decode("t7", 2, 7, 2, 4, 8, 16, device=device)),
        ("decode_real4b", lambda: case_decode("4b_t1", 1, 1, 16, 32, 128, 128, device=device)),
        ("decode_real27b", lambda: case_decode("27b_t1", 1, 1, 16, 48, 128, 128, device=device)),
        ("prefill_decode_handoff", lambda: case_prefill_handoff(device)),
        ("state_chain", lambda: case_state_chain(device)),
        ("conv_fn_small_conv1d", lambda: case_conv_fn(
            "b2c48t100", 2, 48, 100, 4, "conv1d", device=device)),
        ("conv_fn_small_unfold", lambda: case_conv_fn(
            "b2c48t100", 2, 48, 100, 4, "unfold", device=device)),
        ("conv_fn_real4b_conv1d", lambda: case_conv_fn(
            "c8192t256", 1, 8192, 256, 4, "conv1d", device=device)),
        ("conv_fn_real4b_unfold", lambda: case_conv_fn(
            "c8192t256", 1, 8192, 256, 4, "unfold", device=device)),
        ("conv_update_L1", lambda: case_conv_update("l1", 2, 96, 4, 3, 1, device=device)),
        ("conv_update_L2", lambda: case_conv_update("l2", 2, 96, 4, 3, 2, device=device)),
    ]
    if not quick:
        cases[9:9] = [
            ("chunk_long_affine", lambda: case_chunk(
                "b1t8192", 1, 8192, 2, 4, 8, 16, scan="affine", chunk_batch=16, device=device)),
            ("chunk_long_seq", lambda: case_chunk(
                "b1t8192", 1, 8192, 2, 4, 8, 16, scan="seq", chunk_batch=16, device=device)),
        ]
    return cases


def run_one(args):
    """子进程入口: 只跑一个 case, 结果 JSON 落盘."""
    device = args.device
    if device == "npu":
        import torch_npu  # noqa: F401
        torch.npu.set_device(0)
    cases = dict(build_cases(device, args.quick))
    fn = cases[args.run_one]
    try:
        ok = bool(fn())
        note = ""
    except Exception as e:  # noqa: BLE001
        ok, note = False, f"EXC: {type(e).__name__}: {e}"
        RESULTS.append((args.run_one, float("nan"), float("nan"), float("nan"), False, note))
    payload = {"case": args.run_one, "ok": ok, "rows": RESULTS}
    with open(args.out, "w") as f:
        json.dump(payload, f)
    print(f"[one] {args.run_one} ok={ok} {note}", flush=True)
    # 本机 TBE daemon 会让 python 在退出阶段死锁(atexit join), 强退跳过清理.
    os._exit(0 if ok else 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "npu"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--isolate", action="store_true",
                    help="每个 case 独立进程跑(规避本机 TBE 编译服务连续编译死锁); NPU 默认开启")
    ap.add_argument("--no-isolate", action="store_true")
    ap.add_argument("--run-one", default=None)
    ap.add_argument("--out", default="")
    ap.add_argument("--case-timeout", type=int, default=150)
    args = ap.parse_args()

    if args.run_one is not None:
        return run_one(args)

    device = args.device
    isolate = args.isolate or (device == "npu" and not args.no_isolate)
    if not isolate:
        return run_inline(args, device)

    # 子进程隔离模式(NPU 默认): 超时 case 由下一轮重试(编译缓存已建).
    # 注: 子进程输出必须落文件而非 pipe——主进程 os._exit 后 TBE 孤儿仍持有
    # pipe 写端, capture_output 会死等管道到超时.
    import subprocess
    import tempfile
    cases = build_cases(device, args.quick)
    results = {}
    pending = [name for name, _ in cases]
    for round_id in (1, 2):
        if not pending:
            break
        retry = []
        for name in pending:
            out = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
            logf = tempfile.NamedTemporaryFile(suffix=".log", delete=False).name
            cmd = [sys.executable, "-u", __file__, "--run-one", name,
                   "--device", device, "--out", out]
            if args.quick:
                cmd.append("--quick")
            with open(logf, "w") as lf:
                proc = subprocess.Popen(cmd, stdout=lf, stderr=lf)
                killed = False
                try:
                    proc.wait(timeout=args.case_timeout)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                    killed = True
            payload = None
            if os.path.exists(out):
                try:
                    with open(out) as f:
                        payload = json.load(f)
                except Exception:  # noqa: BLE001
                    payload = None
            if payload is not None:
                note = "exit-hang(killed)" if killed else ""
                results[name] = (payload["ok"], payload["rows"], note)
                print(f"[{device} r{round_id}] {name}: "
                      f"{'PASS' if payload['ok'] else 'FAIL'}"
                      f"{'(exit-hang)' if killed else ''}", flush=True)
            else:
                retry.append(name)
                results[name] = (False, [(name, float("nan"), float("nan"),
                                          float("nan"), False,
                                          f"HANG(>{args.case_timeout}s 无结果)")], "")
                print(f"[{device} r{round_id}] {name}: HANG", flush=True)
            os.unlink(out)
            os.unlink(logf)
        pending = retry
        pending = retry
    all_rows = []
    failed = []
    for name, _ in cases:
        ok, rows, _ = results[name]
        all_rows.extend(rows)
        if not ok:
            failed.append(name)

    print(f"\n{'case':44s} {'rel':>10s} {'maxabs':>10s} {'gate':>8s}  result")
    print("-" * 88)
    for name, rel, mx, gate, ok, note in all_rows:
        mark = "PASS" if ok else "FAIL"
        print(f"{name:44s} {rel:10.3e} {mx:10.3e} {gate:8.1e}  {mark}  {note}")
    print("-" * 88)
    n_ok = sum(1 for r in all_rows if r[4])
    print(f"total {len(all_rows)}  pass {n_ok}  fail {len(all_rows) - n_ok}  device={device}")
    if failed:
        print("FAILED:", ", ".join(failed))
    return 1 if failed else 0


def run_inline(args, device):
    """CPU 常规顺序模式."""
    failed = []

    def run(name, fn):
        print(f"[run] {name} ...", flush=True)
        try:
            ok = fn()
            if not ok:
                failed.append(name)
        except Exception as e:  # noqa: BLE001
            RESULTS.append((name, float("nan"), float("nan"), float("nan"), False,
                            f"EXC: {type(e).__name__}: {e}"))
            failed.append(name)

    for name, fn in build_cases(device, args.quick):
        run(name, fn)

    print(f"\n{'case':44s} {'rel':>10s} {'maxabs':>10s} {'gate':>8s}  result")
    print("-" * 88)
    for name, rel, mx, gate, ok, note in RESULTS:
        mark = "PASS" if ok else "FAIL"
        print(f"{name:44s} {rel:10.3e} {mx:10.3e} {gate:8.1e}  {mark}  {note}")
    print("-" * 88)
    n_ok = sum(1 for r in RESULTS if r[4])
    print(f"total {len(RESULTS)}  pass {n_ok}  fail {len(RESULTS) - n_ok}  device={device}")
    if failed:
        print("FAILED:", ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
