"""GdnDecodeStep 离线 om 的 ctypes 直调封装(免编译).

姿势(抄自 msopst 生成代码, 本机已验证):
    aclopSetModelDir(om_dir)   一次性注册离线单算子模型(static map)
    aclopExecuteV2(...)        按输入 shape 自动匹配 om 并执行
aclop 的动态模型生成与 graph capture 在本机均不可用, 但 static map 路径可用.
"""
import ctypes
import os

import torch

_LIB = None
_ACL_FLOAT = 0
_ACL_FLOAT16 = 1
_ACL_ND = 2


def _lib():
    global _LIB
    if _LIB is None:
        # 必须解析到 torch_npu 已加载的同一份副本(dlopen 按名复用)
        _LIB = ctypes.CDLL("libascendcl.so")
    return _LIB


def _check(rc, what):
    if rc != 0:
        raise RuntimeError(f"acl {what} failed rc={rc}")


class GdnDecodeStepOm:
    """GdnDecodeStep 静态模型执行器(om_dir 下全部 B 档 om 按 shape 自动匹配)."""

    def __init__(self, om_dir: str):
        lib = _lib()
        _check(lib.aclopSetModelDir(om_dir.encode()), "SetModelDir")
        self.lib = lib
        self.attr = lib.aclopCreateAttr()
        self._descs = {}  # (name, shape_tuple, dtype) -> desc

    def _desc(self, name, t):
        key = (name, tuple(t.shape), t.dtype)
        d = self._descs.get(key)
        if d is None:
            dims = (ctypes.c_int64 * len(t.shape))(*t.shape)
            dt = _ACL_FLOAT16 if t.dtype == torch.float16 else _ACL_FLOAT
            d = self.lib.aclCreateTensorDesc(dt, len(t.shape), dims, _ACL_ND)
            _check(self.lib.aclSetTensorDescName(d, name.encode()), "SetTensorDescName")
            self._descs[key] = d
        return d

    def _dataset(self, tensors, descs):
        lib = self.lib
        ds = lib.aclmdlCreateDataset()
        buffers = []
        for t, d in zip(tensors, descs):
            buf = lib.aclCreateDataBuffer(ctypes.c_void_p(t.data_ptr()),
                                          t.numel() * t.element_size())
            rc = lib.aclmdlAddDatasetBuffer(ds, buf)
            if rc != 0:
                for b in buffers:
                    lib.aclDestroyDataBuffer(b)
                lib.aclmdlDestroyDataset(ds)
                _check(rc, "AddDatasetBuffer")
            buffers.append(buf)
        return ds, buffers

    @staticmethod
    def _free_dataset(lib, ds, buffers):
        for b in buffers:
            lib.aclDestroyDataBuffer(b)
        lib.aclmdlDestroyDataset(ds)

    def bind(self, q, k, v, g, beta, state):
        """预绑定固定张量: dataset/buffer 全缓存, 之后 run_cached 零 host 构造."""
        names = ("q", "k", "v", "g", "beta", "state")
        self.tensors = {"q": q, "k": k, "v": v, "g": g, "beta": beta, "state": state}
        ins = [q, k, v, g, beta, state]
        in_descs = [self._desc(n, t) for n, t in zip(names, ins)]
        self.out = torch.empty_like(v)
        self.new_state = torch.empty_like(state)
        out_descs = [self._desc("out", self.out), self._desc("new_state", self.new_state)]
        lib = self.lib
        self.in_ds, self.in_bufs = self._dataset(ins, in_descs)
        self.out_ds, self.out_bufs = self._dataset([self.out, self.new_state], out_descs)
        self._in_darr = (ctypes.c_void_p * 6)(*[ctypes.cast(d, ctypes.c_void_p) for d in in_descs])
        self._in_barr = (ctypes.c_void_p * 6)(*[ctypes.cast(b, ctypes.c_void_p) for b in self.in_bufs])
        self._out_darr = (ctypes.c_void_p * 2)(*[ctypes.cast(d, ctypes.c_void_p) for d in out_descs])
        self._out_barr = (ctypes.c_void_p * 2)(*[ctypes.cast(b, ctypes.c_void_p) for b in self.out_bufs])
        self._bound = True

    def run_cached(self):
        """bind() 后的零构造执行路径: 每 token 每层仅一次 ExecuteV2 调用."""
        if not getattr(self, "_bound", False):
            raise RuntimeError("call bind() first")
        stream = torch.npu.current_stream().npu_stream
        _check(self.lib.aclopExecuteV2(b"GdnDecodeStep", 6, self._in_darr,
                                      self._in_barr, 2, self._out_darr,
                                      self._out_barr, ctypes.c_void_p(self.attr),
                                      ctypes.c_void_p(stream)), "ExecuteV2")
        return self.out, self.new_state

    def run(self, q, k, v, g, beta, state, out=None, new_state=None):
        """一次性调用(desc 缓存, dataset 临时构造)."""
        lib = self.lib
        if out is None:
            out = torch.empty_like(v)
        if new_state is None:
            new_state = torch.empty_like(state)
        names = ("q", "k", "v", "g", "beta", "state")
        ins = [q, k, v, g, beta, state]
        in_descs = [self._desc(n, t) for n, t in zip(names, ins)]
        out_descs = [self._desc("out", out), self._desc("new_state", new_state)]

        in_ds, in_bufs = self._dataset(ins, in_descs)
        out_ds, out_bufs = self._dataset([out, new_state], out_descs)
        try:
            stream = torch.npu.current_stream().npu_stream
            _check(lib.aclopExecuteV2(b"GdnDecodeStep", 6, (ctypes.c_void_p * 6)(
                *[ctypes.cast(d, ctypes.c_void_p) for d in in_descs]),
                (ctypes.c_void_p * 6)(*[ctypes.cast(b, ctypes.c_void_p) for b in in_bufs]),
                2, (ctypes.c_void_p * 2)(
                    *[ctypes.cast(d, ctypes.c_void_p) for d in out_descs]),
                (ctypes.c_void_p * 2)(*[ctypes.cast(b, ctypes.c_void_p) for b in out_bufs]),
                ctypes.c_void_p(self.attr), ctypes.c_void_p(stream)), "ExecuteV2")
        finally:
            self._free_dataset(lib, in_ds, in_bufs)
            self._free_dataset(lib, out_ds, out_bufs)
        return out, new_state


def init(om_dir: str) -> GdnDecodeStepOm:
    """进程内一次性初始化(torch_npu import 并 set_device 之后)."""
    return GdnDecodeStepOm(os.path.abspath(om_dir))
