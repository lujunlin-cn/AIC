#!/usr/bin/env python3
"""R13 T1g: TorchDispatchMode — re-execute every aten op on CPU during the NPU
GDN layer forward and record the first ops whose results diverge."""
import json
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

torch.npu.set_compile_mode(jit_compile=False)

import transformers.models.qwen3_5.modeling_qwen3_5 as mm  # noqa: E402
from transformers import AutoConfig  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402
from torch._subclasses.fake_tensor import FakeTensorMode  # noqa: E402


def clone_cpu(t):
    return t.detach().to("cpu", torch.float32) if torch.is_tensor(t) else t


class CpuCheck(TorchDispatchMode):
    def __init__(self, tol=1e-2):
        super().__init__()
        self.tol = tol
        self.bad = []
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        self.count += 1
        try:
            if any("FakeTensor" in type(a).__name__ or (torch.is_tensor(a) and a.is_meta) for a in args):
                return out
            cpu_args = torch.utils._pytree.tree_map_only(torch.Tensor, clone_cpu, args)
            cpu_kwargs = torch.utils._pytree.tree_map_only(torch.Tensor, clone_cpu, kwargs)
            if "masked_fill" in str(func) and cpu_args and torch.is_tensor(cpu_args[1]):
                cpu_args = list(cpu_args)
                cpu_args[1] = cpu_args[1].bool()
                cpu_args = tuple(cpu_args)
            with torch._C._DisableTorchDispatch():
                cpu_out = func(*cpu_args, **cpu_kwargs)
            out_t = out[0] if isinstance(out, (list, tuple)) else out
            cpu_t = cpu_out[0] if isinstance(cpu_out, (list, tuple)) else cpu_out
            if torch.is_tensor(out_t) and out_t.is_floating_point():
                d = (out_t.detach().float().cpu() - cpu_t.float()).abs().max().item()
                if d > self.tol:
                    self.bad.append({"op": str(func), "maxdiff": round(d, 4),
                                     "shape": list(out_t.shape)})
        except Exception as e:  # noqa: BLE001
            self.bad.append({"op": str(func), "check_error": str(e)[:120]})
        return out


cfg = AutoConfig.from_pretrained("/data/aic/pretrained/Qwen3.5-4B")
text_cfg = getattr(cfg, "text_config", cfg)
torch.manual_seed(0)
x = torch.randn(2, 64, 2560)

layer_cpu = mm.Qwen3_5GatedDeltaNet(text_cfg, layer_idx=0).eval()
with torch.no_grad():
    y_cpu = layer_cpu(x)

layer_npu = mm.Qwen3_5GatedDeltaNet(text_cfg, layer_idx=0).eval().to("npu:0")
layer_npu.load_state_dict(layer_cpu.state_dict())

chk = CpuCheck()
with chk, torch.no_grad():
    y = layer_npu(x.to("npu:0"))
y = y if isinstance(y, torch.Tensor) else y[0]

rep = {"ops_checked": chk.count, "n_bad": len(chk.bad),
       "layer_maxdiff": float((y.detach().cpu() - y_cpu).abs().max()),
       "first_bad": chk.bad[:12]}
print(json.dumps(rep, ensure_ascii=False, indent=1))
print("T1G_DONE")
