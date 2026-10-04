"""NPU patch for the Siglip2 NaFlex positional-embedding resize bug.

Root cause (found 2026-10-04, round-5 B1 parity investigation):
transformers 5.18 Siglip2VisionEmbeddings.resize_positional_embeddings
returns ALL ZEROS on torch_npu/CANN 9.0.0 (measured: NPU output norm 0.0
vs CPU 566.2; downstream features corr 0.16 vs CPU, champion F1 drops
0.77 -> 0.52, 32/32 masks differ).  The module builds its result with
torch.empty + row-wise writes; on this stack the written result is lost.
The interpolate kernel itself is CORRECT on NPU (verified corr 1.0,
maxdiff 0.0) - only the module-level empty+write pattern fails.

Patch: compute the resize on CPU (256x768 source grid - negligible cost)
and move the result to the model's device.  After the patch, NPU fp32 vs
CPU fp32 features agree at corr 1.0, maxdiff 5.7e-4 (float noise).

Usage:
    from v11_npu_siglip2_patch import patch_siglip2_npu
    model = Lfm2VlForConditionalGeneration.from_pretrained(...).to('npu')
    patch_siglip2_npu(model)

Scope: transformers 5.18, torch 2.9, torch_npu 2.9.0.post2, CANN 9.0.0.
Re-verify with a parity run after ANY stack change
(scripts/v11_round5_cpu_npu_parity.py).
"""


def patch_siglip2_npu(model):
    vt = getattr(getattr(model, 'model', model), 'vision_tower', None)
    if vt is None or not hasattr(vt, 'embeddings'):
        return False
    emb = vt.embeddings
    if getattr(emb, '_npu_pos_patch', False):
        return True
    orig = emb.resize_positional_embeddings

    def resize_on_cpu(positional_embeddings, spatial_shapes, max_length):
        src_dtype = positional_embeddings.dtype
        out = orig(positional_embeddings.float().cpu(),
                   spatial_shapes.detach().cpu(),
                   max_length=max_length)
        # keep the caller's dtype (fp16 model -> fp16 pos embeds), else the
        # patch_embeds + pos add silently upcasts and breaks attention dtypes
        return out.to(device=positional_embeddings.device, dtype=src_dtype)

    emb.resize_positional_embeddings = resize_on_cpu
    emb._npu_pos_patch = True
    return True
