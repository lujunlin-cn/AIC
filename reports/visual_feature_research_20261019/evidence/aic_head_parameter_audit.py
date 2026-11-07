#!/usr/bin/env python3
"""Pure integer audit of local Head ASTs; no torch, model, weights, or GT."""
import argparse
import ast
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ENV = {"d": 2305, "ch": 128, "heads": 4, "ff": 256, "nc": 129}
EXPECTED = {"proj", "qkv", "proj_o", "ln1", "ln2", "ff1", "ff2", "out"}


def integer(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, bool)):
        return node.value
    if isinstance(node, ast.Name) and node.id in ENV:
        return ENV[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [integer(item) for item in node.elts]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return integer(node.left) * integer(node.right)
    raise ValueError(f"Unsupported integer expression: {ast.dump(node)}")


def count_module(node):
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "nn"):
        raise ValueError(f"Not a supported nn constructor: {ast.dump(node)}")
    name = node.func.attr
    kw = {item.arg: integer(item.value) for item in node.keywords}
    if name == "Linear":
        din, dout = map(integer, node.args)
        return din * dout + (dout if kw.get("bias", True) else 0)
    if name == "LayerNorm":
        shape = integer(node.args[0])
        n = math.prod(shape) if isinstance(shape, list) else shape
        return n * (int(kw.get("elementwise_affine", True))
                    + int(kw.get("elementwise_affine", True) and kw.get("bias", True)))
    if name in ("GELU", "Dropout"):
        return 0
    if name == "Sequential":
        return sum(count_module(child) for child in node.args)
    if name == "ModuleList":
        children = node.args[0]
        if isinstance(children, ast.List):
            return sum(count_module(child) for child in children.elts)
        if isinstance(children, ast.ListComp) and len(children.generators) == 1:
            gen = children.generators[0]
            if gen.ifs or gen.is_async:
                raise ValueError("Conditional/asynchronous module lists unsupported")
            if not (isinstance(gen.iter, ast.Call) and isinstance(gen.iter.func, ast.Name)
                    and gen.iter.func.id == "range" and len(gen.iter.args) == 1):
                raise ValueError("Only constant range(n) module lists supported")
            return integer(gen.iter.args[0]) * count_module(children.elt)
    raise ValueError(f"Unsupported module declaration: {ast.dump(node)}")


def audit(path, relative):
    raw = path.read_bytes()
    tree = ast.parse(raw.decode())
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Head"]
    if len(classes) != 1:
        raise ValueError(f"Expected one Head class in {path}")
    constructors = [n for n in ast.walk(classes[0])
                    if isinstance(n, ast.FunctionDef) and n.name == "__init__"]
    if len(constructors) != 1:
        raise ValueError(f"Expected one Head constructor in {path}")
    groups, expressions = {}, {}
    for stmt in constructors[0].body:
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1):
            continue
        target = stmt.targets[0]
        if not (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                and target.value.id == "self"):
            continue
        value = stmt.value
        if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                and isinstance(value.func.value, ast.Name) and value.func.value.id == "nn"):
            continue  # Scalar metadata such as self.h = heads.
        if target.attr in groups:
            raise ValueError(f"Repeated module assignment: {target.attr}")
        groups[target.attr] = count_module(value)
        expressions[target.attr] = ast.unparse(value)
    if set(groups) - {"drop"} != EXPECTED:
        raise ValueError(f"Unreviewed architecture fields in {relative}: {set(groups)}")
    return {
        "source": relative, "source_sha256": hashlib.sha256(raw).hexdigest(),
        "architecture_arguments": ENV,
        "input_d_basis": "3 * 768 pooled features + 1 position scalar; local feature concatenation",
        "declared_parameters": sum(groups.values()),
        "parameter_groups": groups, "source_module_expressions": expressions,
        "model_instantiated": False, "forward_executed": False,
        "checkpoint_loaded": False,
        "historical_B3_checkpoint_identity_verified": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records = [audit(args.repo / relative, relative) for relative in
               ("scripts/v9_vision_only_points.py", "scripts/r70_spatial_utility_pilot.py")]
    if any(row["declared_parameters"] != 1511425 for row in records):
        raise ValueError("Declared Head count differs from reviewed architecture")
    result = {
        "date_shanghai": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "scope": "Pure integer count from local AST module declarations; no torch, model construction, weights or GT",
        "records": records,
        "historical_encoder_declared_parameters": 85843200,
        "historical_yunet_declared_parameters": 53104,
        "historical_encoder_yunet_not_independently_loaded": True,
        "conditional_vision_head_yunet_total": 85843200 + 1511425 + 53104,
        "total_condition": "Applicable only if historical assets match the audited 2305-input Head and declared encoder/YuNet counts",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
