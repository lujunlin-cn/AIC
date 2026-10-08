#!/usr/bin/env python3
"""R13 T1i: clean per-segment diff inside torch_chunk_gated_delta_rule.

Step 1: run the REAL GDN layer once, capture the exact (query,key,value,g,beta)
        shapes passed into the chunk kernel.
Step 2: rebuild random tensors with those exact shapes, run the chunk function
        body segment by segment identically on CPU and NPU (no dispatch mode),
        report per-segment max diff for both the solve path and the fallback.
"""
import json
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

torch.npu.set_compile_mode(jit_compile=False)

import transformers.models.qwen3_5.modeling_qwen3_5 as mm  # noqa: E402
from transformers import AutoConfig  # noqa: E402

MODEL = "/data/aic/pretrained/Qwen3.5-4B"
cfg = AutoConfig.from_pretrained(MODEL)
text_cfg = getattr(cfg, "text_config", cfg)

# ---- step 1: capture real shapes ----
captured = {}
orig_chunk = mm.torch_chunk_gated_delta_rule


def capture_chunk(q, k, v, g=None, beta=None, **kw):
    captured["shapes"] = [tuple(t.shape) for t in (q, k, v, g, beta)]
    captured["dtypes"] = [str(t.dtype) for t in (q, k, v, g, beta)]
    return orig_chunk(q, k, v, g=g, beta=beta, **kw)


mm.torch_chunk_gated_delta_rule = capture_chunk
torch.manual_seed(0)
layer = mm.Qwen3_5GatedDeltaNet(text_cfg, layer_idx=0).eval()
with torch.no_grad():
    layer(torch.randn(2, 64, 2560))
mm.torch_chunk_gated_delta_rule = orig_chunk
print("captured shapes:", captured["shapes"], captured["dtypes"], flush=True)

# ---- step 2: per-segment clean diff ----
SQ, SK, SV, SG, SB = captured["shapes"]
torch.manual_seed(1)
q = torch.randn(SQ)
k = torch.randn(SK)
v = torch.randn(SV)
g = -torch.rand(SG).cumsum(1) * 0.05 - 1e-4
beta = torch.rand(SB)


def run_segments(device, solver):
    qd, kd, vd, gd, bd = [t.to(device) for t in (q, k, v, g, beta)]
    seg = {}

    def S(name, t_cpu, t_dev):
        seg[name] = float((t_dev.detach().float().cpu() - t_cpu).abs().max())

    qq = qd.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
    kc = kd.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
    vv = vd.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
    bb = bd.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
    gg = gd.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)

    qn = mm.l2norm(qq, dim=-1, eps=1e-6)
    kn = mm.l2norm(kc, dim=-1, eps=1e-6)
    qn_ref = q.transpose(1, 2).float()
    kn_ref = k.transpose(1, 2).float()
    qn_ref = qn_ref / torch.sqrt((qn_ref * qn_ref).sum(-1, keepdim=True) + 1e-6)
    kn_ref = kn_ref / torch.sqrt((kn_ref * kn_ref).sum(-1, keepdim=True) + 1e-6)
    S("l2norm_q", qn_ref, qn)
    S("l2norm_k", kn_ref, kn)

    scaling = qn.shape[-1] ** -0.5
    qn = qn * scaling
    seq = qn.shape[2]
    pad = (64 - seq % 64) % 64
    if pad:
        qn = torch.nn.functional.pad(qn, (0, 0, 0, pad))
        kn = torch.nn.functional.pad(kn, (0, 0, 0, pad))
        vv = torch.nn.functional.pad(vv, (0, 0, 0, pad))
        bb = torch.nn.functional.pad(bb, (0, pad))
        gg = torch.nn.functional.pad(gg, (0, pad))
    v_beta = vv * bb.unsqueeze(-1)
    k_beta = kn * bb.unsqueeze(-1)
    f = lambda x: x.reshape(x.shape[0], x.shape[1], -1, 64, x.shape[-1])
    qn, kn, k_beta, v_beta = f(qn), f(kn), f(k_beta), f(v_beta)
    decay = gg.reshape(gg.shape[0], gg.shape[1], -1, 64)

    up_mask = torch.ones(64, 64, dtype=torch.bool, device=device).triu(1)
    cum = decay.cumsum(dim=3)
    cum_ref = gg.detach().float().cpu().reshape(gg.shape[0], gg.shape[1], -1, 64).cumsum(dim=3)
    S("cumsum", cum_ref, cum)

    pw = cum.unsqueeze(4) - cum.unsqueeze(3)
    pw = pw.masked_fill(up_mask, float("-inf"))
    pw = pw.exp()
    pw_ref = cum_ref.unsqueeze(4) - cum_ref.unsqueeze(3)
    pw_ref = pw_ref.masked_fill(torch.ones(64, 64, dtype=torch.bool).triu(1), float("-inf"))
    pw_ref = pw_ref.exp()
    S("pairwise_decay", pw_ref, pw)

    ut = (k_beta @ kn.transpose(-1, -2)) * pw
    kn4 = kn_ref.reshape(kn_ref.shape[0], kn_ref.shape[1], -1, 64, kn_ref.shape[-1])
    k_beta_ref = kn4 * bb.detach().float().cpu().reshape(bb.shape[0], bb.shape[1], -1, 64).unsqueeze(-1)
    ut_ref = (k_beta_ref @ kn4.transpose(-1, -2)) * pw_ref
    S("ut_matmul", ut_ref, ut)

    dk = k_beta * cum.exp().unsqueeze(-1)
    dk_ref = k_beta.detach().float().cpu() * cum_ref.exp().unsqueeze(-1)
    if solver == "solve":
        nv = torch.linalg.solve_triangular(ut, v_beta, upper=False, unitriangular=True)
        nv_ref = torch.linalg.solve_triangular(ut_ref, v_beta.detach().float().cpu(),
                                               upper=False, unitriangular=True)
    else:
        u2 = -ut.tril(-1)
        u2r = -ut_ref.tril(-1)
        for i in range(1, 64):
            row = u2[..., i, :i].clone()
            sub = u2[..., :i, :i].clone()
            u2[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
            rowr = u2r[..., i, :i].clone()
            subr = u2r[..., :i, :i].clone()
            u2r[..., i, :i] = rowr + (rowr.unsqueeze(-1) * subr).sum(-2)
        u2 = u2 + torch.eye(64, dtype=u2.dtype, device=device)
        u2r = u2r + torch.eye(64)
        nv, nv_ref = u2 @ v_beta, u2r @ v_beta.detach().float().cpu()
    S("new_values", nv_ref, nv)
    return seg


rep = {"shapes": captured["shapes"], "dtypes": captured["dtypes"]}
rep["solve_path"] = run_segments("npu:0", "solve")
rep["fallback_path"] = run_segments("npu:0", "fallback")
print(json.dumps(rep, ensure_ascii=False, indent=1))
print("T1I_DONE")
