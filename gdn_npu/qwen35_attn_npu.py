"""Qwen3.5 full-attention 分块实现(训练激活显存解) — 910A 可用版.

判定依据(2026-10-12, 文件级证据, 零 NPU 消耗):
  CANN 9.0.0 opp/built-in 全树无 FlashAttention 类 kernel(仅 detect_* 检测域变体);
  ascend910 家族预编译融合算子仅 4 个 conv2d 融合; tbe impl 仅 9 个文件无 attention;
  libopapi 有 FlashAttentionScore host 符号(203 hits)但无 kernel 文件 = 空壳.
  → npu_fusion_attention / npu_prompt_flash_attention / npu_incre_flash_attention
    在本机全部不可用, 原生融合算子路判死.
  同法验证: RecurrentGatedDeltaRule host 符号 18 hits 同样无 kernel → GDN 手写路线
  (qwen35_gdn_npu.py)仍是唯一路, 两模块互不冲突.

本文件: 纯 torch 分块 attention + 块级 gradient checkpoint.
  - 前向 score 峰值 O(S^2) → O(block*S); 训练时 score 不进 autograd 保存区(重算).
  - 数学与 eager 逐块等价: causal 下每行 query 的 key 范围完整落在块内,
    softmax 行内独立精确, 无 online-softmax 近似.
  - Qwen3.5-4B 实测账(S=4900, B=1, fp16): eager 每层保留
    score 0.77 + repeat_kv 0.77 + softmax 输出 0.77 ≈ 2.3GB × 8 full 层 ≈ 18.6GB
    → 分块+checkpoint 后 ≈ 1GB 内(边界+重算峰值).

用法(与 gdn 补丁独立, 可叠加):
    import sys; sys.path.insert(0, "/root/AIC/gdn_npu")
    import qwen35_attn_npu as attn
    attn.apply()                    # 默认 block=1024, ckpt 开
    # 回滚: attn.unpatch() 或环境变量 QWEN35_ATTN_NPU_DISABLE=1

契约: 锚定 transformers 5.19 modeling_qwen3_5.Qwen3_5Attention.forward;
      q/k/v 已 rope+q_norm/k_norm; mask 为 4D 加法 mask([B,1,Sq,Skv])或 None.
"""
import os

import torch
import torch.nn.functional as F

_DEFAULT_BLOCK = 1024
# q_len*kv_len*num_q_heads 超过此元素数才分块(低于则 eager 更省事)
_DEFAULT_MIN_ELEMS = 64 * 1024 * 1024  # 64M 元素(≈score fp16 0.13GB)起分块

_STATE = {
    "patched": False,
    "block": _DEFAULT_BLOCK,
    "use_ckpt": True,
    "min_elems": _DEFAULT_MIN_ELEMS,
}


def is_active():
    return _STATE["patched"]


# ---------------------------------------------------------------------------
# eager 参考(transformers 5.19 modeling_qwen3_5.eager_attention_forward 逐字拷贝)
# ---------------------------------------------------------------------------
def eager_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    scaling,
    dropout=0.0,
    **kwargs,
):
    key_states = _repeat_kv(key, module.num_key_value_groups)
    value_states = _repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def _repeat_kv(hidden_states, n_rep):
    """transformers.modeling_flash_attention_utils.repeat_kv 等价(仅 3/4 维通用形)."""
    if n_rep == 1:
        return hidden_states
    bs, n_kv, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :] \
        .expand(bs, n_kv, n_rep, slen, head_dim) \
        .reshape(bs, n_kv * n_rep, slen, head_dim)
    return hidden_states


# ---------------------------------------------------------------------------
# 分块实现
# ---------------------------------------------------------------------------
def _block_compute(q_blk, key, value, mask_blk, scaling, dropout, n_rep, training):
    """单 query 块前向(可被 checkpoint 重入).

    q_blk: [B,Hq,bs,D]; key/value: [B,Hkv,q_end,D](已截到本块可见末尾);
    mask_blk: [B,1,bs,q_end] 加法 mask 或 None.
    """
    k = _repeat_kv(key, n_rep)
    v = _repeat_kv(value, n_rep)
    w = torch.matmul(q_blk, k.transpose(2, 3)) * scaling
    if mask_blk is not None:
        w = w + mask_blk
    w = F.softmax(w, dim=-1, dtype=torch.float32).to(q_blk.dtype)
    if dropout > 0.0 and training:
        w = F.dropout(w, p=dropout, training=True)
    out = torch.matmul(w, v)
    return out  # [B,Hq,bs,D]


def chunked_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    scaling,
    dropout=0.0,
    **kwargs,
):
    """eager_attention_forward 的分块 drop-in(同签名同返回).

    仅优化 Sq==Skv 主场景(训练/prefill); Sq!=Skv(chunked prefill 增量)一律
    回退 eager — 因果可见范围由 mask 决定, 块边界假设失效, 安全第一.
    attn_weights 返回 None(与 sdpa/flash 一致; output_attentions 不支持).
    """
    if query.shape[2] != key.shape[2]:
        return eager_attention_forward(
            module, query, key, value, attention_mask, scaling, dropout, **kwargs
        )

    st = _STATE
    b, hq, sq, d = query.shape
    n_rep = module.num_key_value_groups
    elems = sq * sq * hq
    if sq <= 1 or elems < st["min_elems"]:
        return eager_attention_forward(
            module, query, key, value, attention_mask, scaling, dropout, **kwargs
        )

    block = min(st["block"], sq)
    use_ckpt = (
        st["use_ckpt"]
        and torch.is_grad_enabled()
        and query.requires_grad
    )

    outs = []
    for q0 in range(0, sq, block):
        q1 = min(q0 + block, sq)
        q_blk = query[:, :, q0:q1, :]
        k_all = key[:, :, :q1, :]          # 本块可见 key 前缀(causal)
        v_all = value[:, :, :q1, :]
        if attention_mask is not None:
            # [B,1,Sq,Skv] → 本块行范围; 广播形(dim2==1, 无行差异)只截 key 维
            if attention_mask.shape[2] == 1:
                mask_blk = attention_mask[:, :, :, :q1]
            else:
                mask_blk = attention_mask[:, :, q0:q1, :q1]
        else:
            mask_blk = None

        if use_ckpt:
            out = torch.utils.checkpoint.checkpoint(
                _block_compute, q_blk, k_all, v_all, mask_blk,
                scaling, dropout, n_rep, bool(module.training),
                use_reentrant=False,
            )
        else:
            out = _block_compute(
                q_blk, k_all, v_all, mask_blk, scaling, dropout, n_rep,
                bool(module.training),
            )
        outs.append(out)

    attn_output = torch.cat(outs, dim=2)               # [B,Hq,Sq,D]
    attn_output = attn_output.transpose(1, 2).contiguous()  # [B,Sq,Hq,D] 同 eager
    return attn_output, None


# ---------------------------------------------------------------------------
# 类级 forward patch(锚定 transformers 5.19 Qwen3_5Attention.forward)
# ---------------------------------------------------------------------------
def _make_patched_forward(allowed_fn):
    """复制原 forward, 仅把 attention_interface 调用替换为 allowed_fn."""

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        **kwargs,
    ):
        # 以下与 transformers 5.19 原实现逐行一致(仅 attention 计算换实现)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        dropout = 0.0 if not self.training else self.attention_dropout
        attn_output, attn_weights = allowed_fn(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    return forward


def apply(block=_DEFAULT_BLOCK, use_ckpt=True, min_elems=_DEFAULT_MIN_ELEMS):
    """接管 Qwen3_5Attention(文本 full attention). 模型构建前后均可."""
    if os.environ.get("QWEN35_ATTN_NPU_DISABLE") == "1":
        return False
    import transformers.models.qwen3_5.modeling_qwen3_5 as m

    if getattr(m.Qwen3_5Attention, "_qwen35_attn_npu", False):
        _STATE.update(block=block, use_ckpt=use_ckpt, min_elems=min_elems)
        return True

    fwd = _make_patched_forward(chunked_attention_forward)
    m.Qwen3_5Attention._qwen35_attn_npu_orig = m.Qwen3_5Attention.forward
    m.Qwen3_5Attention.forward = fwd
    m.Qwen3_5Attention._qwen35_attn_npu = True
    _STATE.update(patched=True, block=block, use_ckpt=use_ckpt, min_elems=min_elems)
    return True


def unpatch():
    """还原原 forward."""
    import transformers.models.qwen3_5.modeling_qwen3_5 as m

    if getattr(m.Qwen3_5Attention, "_qwen35_attn_npu", False):
        m.Qwen3_5Attention.forward = m.Qwen3_5Attention._qwen35_attn_npu_orig
        del m.Qwen3_5Attention._qwen35_attn_npu
        del m.Qwen3_5Attention._qwen35_attn_npu_orig
    _STATE["patched"] = False
