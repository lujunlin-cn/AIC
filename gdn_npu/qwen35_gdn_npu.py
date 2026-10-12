"""Qwen3.5 GDN(Gated DeltaNet) + causal_conv1d 的 910A NPU 加速实现.

替换 transformers 5.19 modeling_qwen3_5 的 4 个 PyTorch 回退函数:
    causal_conv1d_update / causal_conv1d_fn /
    torch_chunk_gated_delta_rule / torch_recurrent_gated_delta_rule

用法(必须在模型构建之前, 机制同 r13_qwen35_safe_chunk.py):
    import sys; sys.path.insert(0, "/root/AIC/gdn_npu")
    import qwen35_gdn_npu as gdn
    gdn.apply()                       # 默认配置(数值对齐优先)
    gdn.apply(solver="nsquaring")     # 三角求解换迭代平方(纯 bmm, 可微)
    gdn.apply(chunk_batch=8)          # intra 量分批算, 压 pairwise_decay 峰值显存
    gdn.apply(scan="affine")          # chunk 间仿射前缀合成(去串行化, 数值门过后再默认)
    gdn.apply(conv_impl="unfold")     # prefill 深度卷积换 unfold+sum

    gdn.is_active()                   # True = 已接管
    gdn.unpatch()                     # 还原 transformers 原实现

应急回滚: 环境变量 GDN_NPU_DISABLE=1 时 apply() 直接跳过.

与 /data/aic/scripts/r13_qwen35_safe_chunk.py 的关系:
    本文件已吸收其 NaN 修复(exp 先行 + masked_fill(0)), apply() 之后
    不要再调用 safe_chunk.apply(), 否则 chunk 路径被覆盖回慢速版.

事实依据(2026-10-10, 详见 GDN_NPU算子调研报告_20261010.md):
    - 4B:  32 层 = 24 GDN + 8 full;  Hv=32, Hk=16, dk=dv=128, conv K=4
    - 27B: 64 层 = 48 GDN + 16 full; Hv=48, Hk=16, dk=dv=128, conv K=4
    - decode(rollout) 首要热点 = torch_recurrent_gated_delta_rule 逐 token 循环
    - solve_triangular 已在 910A 实证可用(现行 safe_chunk 训练在用)

已知疑点(对齐基准说明):
    transformers 5.19 的 torch_chunk 回退里, solve_triangular 分支解 (I+L)X=V,
    而 dynamo export 分支解 (I-L)X=V, 两分支符号相反. 生产路径(solve 分支)
    已被训练/打分验证, 本实现全部对齐 solve 分支; nsquaring 亦按 (I+L) 求逆.
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F

__all__ = ["apply", "unpatch", "is_active"]

# ----------------------------------------------------------------------
# 公共小件
# ----------------------------------------------------------------------

_MASK_CACHE: dict = {}


def _strict_upper_mask(chunk_size: int, device: torch.device) -> torch.Tensor:
    """chunk 内严格上三角 bool 掩码(缓存, 参考实现每次重建)."""
    key = (chunk_size, str(device))
    m = _MASK_CACHE.get(key)
    if m is None:
        m = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=device).triu(1)
        _MASK_CACHE[key] = m
    return m


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """与 transformers/fla 对齐的 l2norm(调用点在 fp32 上)."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


_ACT2FN = {"silu": F.silu, "gelu": F.gelu}


def _act(name):
    if name is None:
        return None
    fn = _ACT2FN.get(name)
    if fn is None:
        raise NotImplementedError(f"activation {name!r} not supported")
    return fn


# ----------------------------------------------------------------------
# causal_conv1d(深度一维卷积)
# ----------------------------------------------------------------------

def npu_causal_conv1d_update(hidden_states, conv_state, weight, bias=None, activation=None):
    """decode 单步卷积状态更新(等价 transformers 参考实现).

    hidden_states [B,C,L](L=1 主路径), conv_state [B,C,SL] in-place 更新,
    weight [C,K]. 用显式窗求和替代 cat+F.conv1d, L=1 时省 cat/conv/slice.
    """
    K = weight.shape[-1]
    L = hidden_states.shape[-1]
    SL = conv_state.shape[-1]
    x = hidden_states.to(weight.dtype)

    # 先用旧 conv_state 算输出(参考实现 conv 在状态写回之前), 再写回状态
    if L == 1:
        # out = w[K-1]*x_t + sum_{j=0}^{K-2} w[j]*state[..., SL-K+1+j]
        need = K - 1
        if SL >= need:
            tail = conv_state[:, :, SL - need:] if need else None
        else:  # SL < K-1: 左侧补零(理论上 Qwen3.5 SL=K-1, 不触发)
            tail = F.pad(conv_state, (need - SL, 0))
        out = x * weight[None, :, -1:]
        if need:
            out = out + (tail * weight[None, :, :-1]).sum(-1, keepdim=True)
        if bias is not None:
            out = out + bias[None, :, None]
    else:
        # 通用路径退回参考实现(预填充走 causal_conv1d_fn, 此分支少用)
        h = torch.cat([conv_state, x], dim=-1)[:, :, -(SL + L):]
        out = F.conv1d(h, weight.unsqueeze(1), bias, padding=0, groups=weight.shape[0])
        out = out[:, :, -L:]

    # conv_state 写回: cat(state, x) 的最后 SL 个位置(保持 in-place 语义)
    if L >= SL:
        conv_state.copy_(x[:, :, -SL:])
    else:
        conv_state[:, :, : SL - L].copy_(conv_state[:, :, L:])
        conv_state[:, :, SL - L:].copy_(x)

    out = _act(activation)(out) if activation is not None else out
    return out.to(hidden_states.dtype)


def _conv1d_causal_unfold(x, weight, bias):
    """causal 深度卷积的 unfold+bmm 实现(备选, 待 M1 bench).

    注: 不用 (xu * w).sum(-1) 的广播乘形式——大 numel fp16 广播乘在
    部分 torch CPU 后端会触发内核缺陷(实测被 OOM 杀), bmm 形式无广播且更快.
    """
    K = weight.shape[-1]
    B, C, T = x.shape
    xp = F.pad(x, (K - 1, 0))
    xw = xp.unfold(-1, K, 1).reshape(B * C, T, K)      # 视图合并合法, 零拷贝
    wk = weight[:, :, None].expand(B, C, K, 1).reshape(B * C, K, 1)
    out = torch.bmm(xw, wk).reshape(B, C, T)
    if bias is not None:
        out = out + bias[None, :, None]
    return out


def npu_causal_conv1d_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
    """prefill causal 深度卷积. conv_impl: "conv1d"(默认, 锚定) | "unfold"."""
    impl = _CFG.get("conv_impl", "conv1d")
    hidden_size = weight.shape[0]
    padding = weight.shape[-1] - 1
    x = hidden_states.to(weight.dtype)
    if impl == "unfold":
        out = _conv1d_causal_unfold(x, weight, bias)
    else:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=padding, groups=hidden_size)
    out = out[:, :, : hidden_states.shape[-1]]
    out = _act(activation)(out) if activation is not None else out
    return out.to(hidden_states.dtype)


# ----------------------------------------------------------------------
# 三角求解(可选 solver)
# ----------------------------------------------------------------------

def _nsquaring_inv_apply(neg_l, rhs):
    """inv(I+L)@rhs 的迭代平方(纯 bmm 可微), 独立成函数供 checkpoint 重算.

    严格下三角幂级数在 k=chunk_size 处精确终止; log2(C) 轮, 每轮
    inv@(eye+power) 与 power@power. 训练图不 checkpoint 时 autograd 会保留
    每轮的 ~3 个 [B,H,NC,C,C] 中间张量(10-12 训练 OOM 三连败之一).
    """
    C = neg_l.shape[-1]
    eye = torch.eye(C, dtype=neg_l.dtype, device=neg_l.device)
    power = neg_l
    inv = eye.expand_as(neg_l).contiguous()
    for _ in range(int(math.ceil(math.log2(C))) if C > 1 else 0):
        inv = inv @ (eye + power)
        power = power @ power
        if C > 1:
            power = power.tril(-1)  # 清浮点上三角噪声(数学上应为 0)
    return inv @ rhs


def _solve_unit_lower(ut_system, rhs, method, use_ckpt=True):
    """解 (I+L)X = rhs, L = tril(ut_system, -1), 即
    torch.linalg.solve_triangular(ut_system, rhs, upper=False, unitriangular=True).

    method="triangular": 直调 aclnn(910A 已实证可用, 数值锚定).
    method="nsquaring":  迭代平方全 bmm 可微; 训练图自动 checkpoint 重算
                         (use_ckpt), backward 只留输入输出不留 6 轮中间.
    """
    if method == "triangular":
        return torch.linalg.solve_triangular(ut_system, rhs, upper=False, unitriangular=True)
    if method != "nsquaring":
        raise ValueError(f"unknown solver {method!r}")

    neg_l = -ut_system.tril(-1)
    if use_ckpt and torch.is_grad_enabled() and (
            ut_system.requires_grad or rhs.requires_grad):
        return torch.utils.checkpoint.checkpoint(
            _nsquaring_inv_apply, neg_l, rhs, use_reentrant=False)
    return _nsquaring_inv_apply(neg_l, rhs)


# ----------------------------------------------------------------------
# chunk 路径(训练 / 预填充)
# ----------------------------------------------------------------------

def _intra_quantities(query, key, k_beta, v_beta, cum_decay, mask, solver, chunk_batch,
                      use_ckpt=True):
    """chunk 内并行量: new_values(u), k_cumdecay, intra_chunk_attn.

    chunk_batch>0 时沿 chunk 维分批, 压 pairwise_decay/ut_system 峰值显存
    (fp32 下 [B,Hv,NC,64,64], 4B/T=8192 约 2.1GB, 27B 约 3.2GB).
    """
    k_decay_head = k_beta * cum_decay.exp().unsqueeze(-1)
    NC = cum_decay.shape[2]

    if not chunk_batch or chunk_batch <= 0 or chunk_batch >= NC:
        pairwise = (cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)).exp()
        pairwise = pairwise.masked_fill(mask, 0.0)
        ut = (k_beta @ key.transpose(-1, -2)) * pairwise
        intra = (query @ key.transpose(-1, -2)) * pairwise
        new_values = _solve_unit_lower(ut, v_beta, solver, use_ckpt)
        k_cumdecay = _solve_unit_lower(ut, k_decay_head, solver, use_ckpt)
        return new_values, k_cumdecay, intra

    B, H, _, C, dv = v_beta.shape
    dk = key.shape[-1]
    # list+cat 组装: empty+slice 预填会触发 autograd inplace/version 冲突
    # (10-12 训练三连败之三)
    nv_list, kc_list, intra_list = [], [], []
    for s in range(0, NC, chunk_batch):
        sl = slice(s, min(s + chunk_batch, NC))
        cd = cum_decay[:, :, sl]
        pairwise = (cd.unsqueeze(4) - cd.unsqueeze(3)).exp()
        pairwise = pairwise.masked_fill(mask, 0.0)
        ut = (k_beta[:, :, sl] @ key[:, :, sl].transpose(-1, -2)) * pairwise
        nv_list.append(_solve_unit_lower(ut, v_beta[:, :, sl], solver, use_ckpt))
        kc_list.append(_solve_unit_lower(ut, k_decay_head[:, :, sl], solver, use_ckpt))
        intra_list.append((query[:, :, sl] @ key[:, :, sl].transpose(-1, -2)) * pairwise)
    return (torch.cat(nv_list, dim=2), torch.cat(kc_list, dim=2),
            torch.cat(intra_list, dim=2))


def _scan_seq(new_values, k_cumdecay, intra, query, key, chunk_decay, state, out):
    """串行 scan(对齐参考实现, bmm/baddbmm 3 维化 + in-place 减 op 版).

    注: bmm/baddbmm 只接受 3 维批矩阵, 切片后需 reshape(BH, n, m);
    参考 实现的 @ 运算符(matmul)支持任意批维, 语义相同.
    """
    B, H, NC, C, dv = new_values.shape
    dk = key.shape[-1]
    BH = B * H
    st = state.reshape(BH, dk, dv)
    # list+cat 组装输出: out3[:, i] = ... 的 slice 写入会撞 autograd
    # inplace/version 检查(10-12 训练三连败之三的主嫌)
    outs = []
    for i in range(NC):
        # v_new = u_i - k_cumdecay_i @ S   (baddbmm: alpha=-1 等价相减)
        v_new = torch.baddbmm(new_values[:, :, i].reshape(BH, C, dv),
                              k_cumdecay[:, :, i].reshape(BH, C, dk), st, alpha=-1, beta=1)
        # out_i = q_i @ S + intra_i @ v_new
        inter = torch.bmm(query[:, :, i].reshape(BH, C, dk), st)
        outs.append(torch.baddbmm(inter, intra[:, :, i].reshape(BH, C, C), v_new))
        # S = cd_i * S + key_i^T @ v_new
        st = st * chunk_decay[:, :, i].reshape(BH, 1, 1)
        st = torch.baddbmm(st, key[:, :, i].transpose(-1, -2).reshape(BH, dk, C), v_new)
    out.copy_(torch.cat(outs, dim=1).reshape(B, H, NC, C, dv))
    return st.reshape(B, H, dk, dv)


def _scan_affine(query, key, new_values, k_cumdecay, intra, chunk_decay, state0, out,
                 use_ckpt=True):
    """仿射前缀合成 scan: 把 NC 次串行小算子重构为 log2(NC) 层并行大 bmm.

    每 chunk 的状态转移是仿射的:
        S_{i+1} = M_i @ S_i + b_i,   M_i = cd_i*I - key_i^T @ kcd_i,  b_i = key_i^T @ u_i
    输出也可重组:
        out_i = P_i @ S_i + c_i,     P_i = q_i - intra_i @ kcd_i,      c_i = intra_i @ u_i
    故: 前缀对 (M', b') 用 stride-doubling 在 log2(NC) 层内并行合成,
    再一次性物化全部 S_i 并全量算 out. kernel 数 O(log NC) vs 串行 O(NC).
    """
    B, H, NC, C, dv = new_values.shape
    dk = key.shape[-1]
    eye = torch.eye(dk, dtype=new_values.dtype, device=new_values.device)

    flat = lambda t: t.reshape(B * H * NC, t.shape[-2], t.shape[-1])
    key_f = flat(key)                      # [BHNC, C, dk]
    kcd_f = flat(k_cumdecay)               # [BHNC, C, dk]
    u_f = flat(new_values)                 # [BHNC, C, dv]
    intra_f = flat(intra)                  # [BHNC, C, dv]
    q_f = flat(query)                      # [BHNC, C, dk]

    mk = torch.bmm(key_f.transpose(-1, -2), kcd_f)      # key^T @ kcd
    M = mk * (-1.0)
    M = M + chunk_decay.reshape(B * H * NC, 1, 1) * eye  # cd_i * I - key^T kcd
    b = torch.bmm(key_f.transpose(-1, -2), u_f)          # key^T @ u
    P = q_f - torch.bmm(intra_f, kcd_f)                  # q - intra @ kcd
    c = torch.bmm(intra_f, u_f)                          # intra @ u
    M = M.reshape(B, H, NC, dk, dk)
    b = b.reshape(B, H, NC, dk, dv)
    P = P.reshape(B, H, NC, C, dk)
    c = c.reshape(B, H, NC, C, dv)

    # stride-doubling 前缀合成: A_i <- A_{i-d} 后接 A_i.
    # 训练图包 checkpoint(use_ckpt): 合成链的 log2(NC) 层中间量(~636MB/层调用级)
    # 不 checkpoint 会被 autograd 全保留, 24 层模型直接爆显存(10-12 三连败之二);
    # slice 预填改 cat 构造(inplace/version 冲突同三连败之三).
    def _affine_prefix(M, b):
        Mp, bp = M, b
        d = 1
        while d < NC:
            Mp_d = Mp[:, :, d:].reshape(-1, dk, dk)
            Mn = torch.cat(
                [Mp[:, :, :d],
                 torch.bmm(Mp_d, Mp[:, :, :-d].reshape(-1, dk, dk)).reshape(
                     B, H, NC - d, dk, dk)],
                dim=2)
            bn = torch.cat(
                [bp[:, :, :d],
                 torch.baddbmm(bp[:, :, d:].reshape(-1, dk, dv),
                               Mp[:, :, d:].reshape(-1, dk, dk),
                               bp[:, :, :-d].reshape(-1, dk, dv)).reshape(
                     B, H, NC - d, dk, dv)],
                dim=2)
            Mp, bp = Mn, bn
            d *= 2
        return Mp, bp

    if use_ckpt and torch.is_grad_enabled() and (M.requires_grad or b.requires_grad):
        Mp, bp = torch.utils.checkpoint.checkpoint(
            _affine_prefix, M, b, use_reentrant=False)
    else:
        Mp, bp = _affine_prefix(M, b)

    # S_i 全量物化: S_0 = state0; S_{i} = Mp_{i-1} @ S_0 + bp_{i-1}
    # (全部 3 维 bmm; 本机 4 维 matmul 带 transpose 视图会触发 TBE BatchMatMul
    #  在线编译并失败, 见调研报告 P0 探针记录; cat 组装避免 slice 预填)
    if NC > 1:
        Mp_f = Mp[:, :, :-1].reshape(-1, dk, dk)
        S0_f = state0.reshape(B * H, 1, dk, dv).expand(B * H, NC - 1, dk, dv).reshape(-1, dk, dv)
        S_rest = torch.bmm(Mp_f, S0_f).reshape(B, H, NC - 1, dk, dv) + bp[:, :, :-1]
        S_all = torch.cat([state0.unsqueeze(2), S_rest], dim=2)
    else:
        S_all = state0.unsqueeze(2)

    # out 全量: out_i = P_i @ S_i + c_i
    out_flat = torch.baddbmm(c.reshape(-1, C, dv), P.reshape(-1, C, dk), S_all.reshape(-1, dk, dv))
    out.copy_(out_flat.reshape(B, H, NC, C, dv))
    final = torch.baddbmm(bp[:, :, -1].reshape(B * H, dk, dv),
                          Mp[:, :, -1].reshape(B * H, dk, dk),
                          state0.reshape(B * H, dk, dv))
    return final.reshape(B, H, dk, dv)


def npu_chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64,
                               initial_state=None, output_final_state=False,
                               use_qk_l2norm_in_kernel=False, cu_seqlens=None, **kwargs):
    """Gated DeltaRule chunk 形式(接口同 transformers 参考实现).

    相对参考实现的改动:
      1. NaN 修复: pairwise_decay exp 先行再 masked_fill(0)(吸收 safe_chunk).
      2. solver: "triangular"(默认) | "nsquaring"(纯 bmm).
      3. chunk_batch: intra 量分批, 压峰值显存.
      4. scan: "seq"(串行, 对齐参考) | "affine"(仿射前缀, 去串行化).
    """
    if cu_seqlens is not None:
        raise NotImplementedError("cu_seqlens packed path not supported")
    if query.shape[-2] != value.shape[-2] or key.shape[-2] != value.shape[-2]:
        raise ValueError(
            "q/k must be repeat_interleave'd to num_v_heads before the kernel "
            f"(call-site contract): q={query.shape}, k={key.shape}, v={value.shape}")
    solver = _CFG.get("solver", "triangular")
    scan_mode = _CFG.get("scan", "seq")
    chunk_batch = _CFG.get("chunk_batch", 0)
    use_ckpt = _CFG.get("use_ckpt", True)

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
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
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
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, k_beta, v_beta)
    ]
    decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)

    mask = _strict_upper_mask(chunk_size, query.device)
    cum_decay = decay.cumsum(dim=3)

    new_values, k_cumdecay, intra = _intra_quantities(
        query, key, k_beta, v_beta, cum_decay, mask, solver, chunk_batch,
        use_ckpt=use_ckpt)

    if initial_state is None:
        state = torch.zeros(recurrent_state_shape, dtype=new_values.dtype, device=new_values.device)
    else:
        state = initial_state.to(new_values)
    core_attn_out = torch.empty_like(new_values)

    # 预衰减(对齐参考: intra 计算之后才做); 非 inplace: query/key 已在
    # autograd 图上, mul_ 触发 version 冲突(10-12 训练 inplace bug 共犯)
    query = query * cum_decay.exp().unsqueeze(-1)
    key = key * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
    chunk_decay = cum_decay[..., -1].exp()[..., None, None]

    if scan_mode == "affine":
        final = _scan_affine(query, key, new_values, k_cumdecay, intra,
                             chunk_decay, state, core_attn_out, use_ckpt=use_ckpt)
    else:
        final = _scan_seq(new_values, k_cumdecay, intra, query, key,
                          chunk_decay, state, core_attn_out)

    last_recurrent_state = final if output_final_state else None
    core_attn_out = core_attn_out.reshape(padded_output_shape)
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).to(
        initial_dtype, memory_format=torch.contiguous_format)
    return core_attn_out, last_recurrent_state


# ----------------------------------------------------------------------
# decode 路径(rollout 首要热点)
# ----------------------------------------------------------------------

def npu_recurrent_gated_delta_rule(query, key, value, g, beta,
                                   initial_state=None, output_final_state=False,
                                   use_qk_l2norm_in_kernel=False, **kwargs):
    """Gated DeltaRule 逐 token 递推(接口同参考实现).

    向量化改动: B*H 合维、g 预 exp、in-place 状态更新、empty 预分配输出,
    单 token 从约 11 次算子(含每步分配)降到 7 次.
    """
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    decay = g

    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, decay)
    ]
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query = query / (query.shape[-1] ** 0.5)

    BH = batch_size * num_v_heads
    qf = query.reshape(BH, sequence_length, k_head_dim)
    kf = key.reshape(BH, sequence_length, k_head_dim)
    vf = value.reshape(BH, sequence_length, v_head_dim)
    gf = decay.reshape(BH, sequence_length)
    bf = beta.reshape(BH, sequence_length)
    g_exp = gf.exp()  # 预付: 循环外一次算完整条 decay 序列

    if initial_state is None:
        st_t = torch.zeros(BH, v_head_dim, k_head_dim, dtype=vf.dtype, device=vf.device)
    else:
        st_t = initial_state.to(vf.dtype).reshape(
            BH, k_head_dim, v_head_dim).transpose(-1, -2).contiguous()

    # 转置状态布局: S_T=[dv,dk], 循环内全部 bmm 零转置视图(避免 aclnn 内部拷贝)
    #   u = S^T k = S_T @ k ;  S_T += delta (x) k ;  out^T = S^T q = S_T @ q
    out = torch.empty_like(vf)
    for t in range(sequence_length):
        st_t.mul_(g_exp[:, t, None, None])
        u = torch.bmm(st_t, kf[:, t, :, None]).squeeze(-1)                # [BH, dv]
        delta = (vf[:, t] - u) * bf[:, t, None]
        st_t.add_(torch.bmm(delta[:, :, None], kf[:, t, None, :]))        # S_T += delta (x) k
        out[:, t] = torch.bmm(st_t, qf[:, t, :, None]).squeeze(-1)        # out^T = S_T @ q

    last_recurrent_state = st_t.transpose(-1, -2).contiguous() \
        .reshape(batch_size, num_v_heads, k_head_dim, v_head_dim) \
        if output_final_state else None
    core_attn_out = out.reshape(batch_size, num_v_heads, sequence_length, v_head_dim)
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


# ----------------------------------------------------------------------
# patch 管理
# ----------------------------------------------------------------------

_CFG: dict = {"solver": "triangular", "scan": "seq", "chunk_batch": 0, "conv_impl": "conv1d",
              "use_ckpt": True}
_ORIG: dict = {}
_TARGETS = (
    "causal_conv1d_update",
    "causal_conv1d_fn",
    "torch_chunk_gated_delta_rule",
    "torch_recurrent_gated_delta_rule",
)
_NPU_IMPLS = {
    "causal_conv1d_update": npu_causal_conv1d_update,
    "causal_conv1d_fn": npu_causal_conv1d_fn,
    "torch_chunk_gated_delta_rule": npu_chunk_gated_delta_rule,
    "torch_recurrent_gated_delta_rule": npu_recurrent_gated_delta_rule,
}


def apply(solver: str = "triangular", scan: str = "seq", chunk_batch: int = 0,
          conv_impl: str = "conv1d", use_ckpt: bool = True):
    """接管 transformers 的 4 个 GDN/conv 回退函数(模型构建前调用).

    use_ckpt: 训练图对 nsquaring solve 链与 affine 前缀合成链走 checkpoint
              重算(10-12 训练三连败的 OOM 修复); no_grad 推理无影响.
    """
    if os.environ.get("GDN_NPU_DISABLE", "").strip() in ("1", "true", "True"):
        print("[gdn_npu] GDN_NPU_DISABLE=1, skip patch")
        return "disabled"
    import transformers.models.qwen3_5.modeling_qwen3_5 as mm

    _CFG.update(solver=solver, scan=scan, chunk_batch=chunk_batch, conv_impl=conv_impl,
                use_ckpt=use_ckpt)
    if not _ORIG:  # 首次接管才备份
        for name in _TARGETS:
            _ORIG[name] = getattr(mm, name)
    for name in _TARGETS:
        fn = _NPU_IMPLS[name]
        fn._gdn_npu_orig_name = name
        setattr(mm, name, fn)
    print(f"[gdn_npu] patched 4 fns: solver={solver} scan={scan} "
          f"chunk_batch={chunk_batch} conv_impl={conv_impl} use_ckpt={use_ckpt}")
    return "patched"


def unpatch():
    """还原 transformers 原实现."""
    if not _ORIG:
        return "not-patched"
    import transformers.models.qwen3_5.modeling_qwen3_5 as mm
    for name, fn in _ORIG.items():
        setattr(mm, name, fn)
    _ORIG.clear()
    print("[gdn_npu] unpatched")
    return "unpatched"


def is_active() -> bool:
    """当前模块函数是否已被本补丁接管."""
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5 as mm
    except ImportError:
        return False
    return all(getattr(mm, n, None) is _NPU_IMPLS[n] for n in _TARGETS)
