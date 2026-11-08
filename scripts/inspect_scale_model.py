"""Inspect real Video-ORA config/adapter schema on CPU without allocating base weights.

This metadata probe is explicitly NOT a real-weight forward/backward smoke test.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys


def inspect(model_dir, runtime_dir, adapter_root):
    import torch
    from accelerate import init_empty_weights
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoProcessor
    from aic.scale_runtime import verify_tokenizer_roundtrip

    sys.path.insert(0, str(runtime_dir))
    from compact_model import CompactQwen3_5ForConditionalGeneration
    config = AutoConfig.from_pretrained(str(model_dir), local_files_only=True)
    with init_empty_weights():
        base = CompactQwen3_5ForConditionalGeneration(config)
        model = get_peft_model(base, LoraConfig.from_pretrained(str(adapter_root / 'portrait')))
        model.add_adapter('other_geometry', LoraConfig.from_pretrained(str(adapter_root / 'vertical')))
        targets = [name for name, module in base.named_modules()
                   if name.endswith(('.self_attn.q_proj', '.self_attn.v_proj', '.self_attn.o_proj'))]
        model.add_adapter('scale', LoraConfig(r=1, lora_alpha=2, lora_dropout=0.,
                                             target_modules=targets, bias='none', task_type='CAUSAL_LM'))
    scale_names = {name:list(p.shape) for name,p in model.named_parameters() if '.scale.' in name}
    parent_names = {name:list(p.shape) for name,p in model.named_parameters() if 'lora_' in name and '.scale.' not in name}
    parameters = sum(p.numel() for p in model.parameters())
    buffers = sum(p.numel() for p in model.buffers())
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
    roundtrip = verify_tokenizer_roundtrip(processor.tokenizer, 8)
    # Test stacked parent + Scale activation supported by the installed PEFT.
    model.base_model.set_adapter(['default','scale'])
    active = list(model.base_model.active_adapters)
    inventory_sha = hashlib.sha256(json.dumps(scale_names,sort_keys=True).encode()).hexdigest()
    return {'status':'metadata_probe_passed', 'real_weights_forward_backward':False,
            'torch_version':torch.__version__, 'base_allocated':False,
            'parameters':parameters, 'buffers':buffers, 'complete_inference_elements':parameters+buffers,
            'cap_9B_passed':parameters+buffers <= 9_000_000_000,
            'scale_parameters':sum(p.numel() for n,p in model.named_parameters() if '.scale.' in n),
            'scale_target_modules':targets, 'scale_allowlist':scale_names,
            'parent_adapter_inventory':parent_names, 'target_inventory_sha256':inventory_sha,
            'active_adapters':active, 'tokenizer_roundtrip':roundtrip}


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--runtime-dir',type=Path,required=True)
    parser.add_argument('--adapter-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=inspect(args.model_dir,args.runtime_dir,args.adapter_root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in {'scale_allowlist','parent_adapter_inventory','scale_target_modules'}},indent=2))

if __name__=='__main__':main()
