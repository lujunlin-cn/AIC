#!/usr/bin/env python3
"""R15 CUDA port of r13_qwen35_safe_chunk.py (exp-then-mask GDN chunk).

The root cause fixed here was torch_npu-specific: masked_fill(-inf) followed
by exp() NaN-poisoned the whole tensor on NPU. The fix (exp first on finite
values, then zero the masked triangle) is backend/dtype-agnostic and stays
numerically safe on CUDA.

When you need it on RTX PRO 6000:
  - flash-linear-attention (fla) installed -> transformers uses fla triton
    kernels; this patch is UNUSED (preferred fast path).
  - fla missing -> transformers falls back to its torch chunk
    implementation; apply() this patch anyway (harmless, mathematically
    identical) to keep the 910A-validated code path.
Import BEFORE building the model.
"""
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import transformers.models.qwen3_5.modeling_qwen3_5 as mm  # noqa: E402


def safe_chunk_gated_delta_rule(
    query, key, value, g, beta, chunk_size=64,
    initial_state=None, output_final_state=False,
    use_qk_l2norm_in_kernel=False, cu_seqlens=None, **kwargs,
):
    if cu_seqlens is not None:
        raise NotImplementedError("packed path not needed for our inference")
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
        query = mm.l2norm(query, dim=-1, eps=1e-6)
        key = mm.l2norm(key, dim=-1, eps=1e-6)
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

    # safe form: exp first (finite everywhere), then zero the masked triangle
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


def apply():
    mm.torch_chunk_gated_delta_rule = safe_chunk_gated_delta_rule
    return "patched"
