#!/usr/bin/env python3
"""R13 T1d: bisect the GDN fault on torch_npu.

Instantiate ONE Qwen3_5GatedDeltaNet layer (fp32), feed identical random
input through CPU and NPU, and wrap the two reference kernels
(causal_conv1d_fn / torch_chunk_gated_delta_rule) so each call reports the
NPU-vs-CPU output diff on the same inputs.
"""
import json
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch  # noqa: E402

MODEL = "/data/aic/pretrained/Qwen3.5-4B"
rep = {}

from transformers import AutoConfig  # noqa: E402
import transformers.models.qwen3_5.modeling_qwen3_5 as mm  # noqa: E402

cfg = AutoConfig.from_pretrained(MODEL)
text_cfg = getattr(cfg, "text_config", cfg)

orig_conv = mm.causal_conv1d_fn
orig_chunk = mm.torch_chunk_gated_delta_rule
stats = {"conv_calls": 0, "conv_maxdiff": 0.0, "chunk_calls": 0, "chunk_maxdiff": 0.0}


def to_cpu(*t):
    return tuple(x.detach().float().cpu() if torch.is_tensor(x) else x for x in t)


def conv_wrapper(x, weight, bias=None, activation=None, **kw):
    out_npu = orig_conv(x, weight, bias, activation, **kw)
    a = to_cpu(x, weight, bias)
    with torch.no_grad():
        out_cpu = orig_conv(*a, activation, **kw)
    d = float((out_npu.detach().float().cpu() - out_cpu).abs().max())
    stats["conv_calls"] += 1
    stats["conv_maxdiff"] = max(stats["conv_maxdiff"], d)
    return out_npu


def chunk_wrapper(query, key, value, g=None, beta=None, initial_state=None,
                  output_final_state=False, use_qk_l2norm_in_kernel=False,
                  cu_seqlens=None, **kw):
    out_npu = orig_chunk(query, key, value, g=g, beta=beta,
                         initial_state=initial_state,
                         output_final_state=output_final_state,
                         use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                         cu_seqlens=cu_seqlens, **kw)
    a = to_cpu(query, key, value)
    st = initial_state.detach().float().cpu() if torch.is_tensor(initial_state) else initial_state
    g_cpu = g.detach().float().cpu() if torch.is_tensor(g) else g
    b_cpu = beta.detach().float().cpu() if torch.is_tensor(beta) else beta
    with torch.no_grad():
        out_cpu = orig_chunk(*a, g=g_cpu, beta=b_cpu, initial_state=st,
                             output_final_state=output_final_state,
                             use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                             cu_seqlens=cu_seqlens, **kw)
    d = float((out_npu[0].detach().float().cpu() - out_cpu[0]).abs().max())
    stats["chunk_calls"] += 1
    stats["chunk_maxdiff"] = max(stats["chunk_maxdiff"], d)
    return out_npu


mm.causal_conv1d_fn = conv_wrapper
mm.torch_chunk_gated_delta_rule = chunk_wrapper

torch.manual_seed(0)
layer = mm.Qwen3_5GatedDeltaNet(text_cfg, layer_idx=0).eval()

# CPU reference first
x = torch.randn(2, 64, 2560)
with torch.no_grad():
    y_cpu = layer(x)

# NPU
import torch_npu  # noqa: F401
torch.npu.set_compile_mode(jit_compile=False)
layer_npu = mm.Qwen3_5GatedDeltaNet(text_cfg, layer_idx=0).eval().to("npu:0")
layer_npu.load_state_dict(layer.state_dict())
with torch.no_grad():
    y_npu = layer_npu(x.to("npu:0"))

rep["layer_maxdiff"] = float((y_npu.cpu() - y_cpu).abs().max())
rep["layer_out_std_cpu"] = float(y_cpu.std())
rep.update(stats)
print(json.dumps(rep, ensure_ascii=False, indent=1))
print("T1D_DONE")
