"""Static constructor arithmetic only. No torch imports, weights, forward, or training.
Supports precisely the layer declarations in author u2net.py/isnet.py; unknown AST fails.
Conv2d count = out * (in/groups * kh*kw + bias); BatchNorm affine = 2*channels.
"""
import ast
import argparse
import hashlib
import json
import operator
from pathlib import Path

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Pow: operator.pow}

def value(n, env):
    if isinstance(n, ast.Constant): return n.value
    if isinstance(n, ast.Name): return env[n.id]
    if isinstance(n, (ast.Tuple, ast.List)): return [value(x, env) for x in n.elts]
    if isinstance(n, ast.BinOp): return OPS[type(n.op)](value(n.left, env), value(n.right, env))
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub): return -value(n.operand, env)
    raise ValueError(ast.dump(n))

def audit(path, names, expected_git_blob):
    blob=path.read_bytes()
    git_blob=hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest()
    if git_blob != expected_git_blob:
        raise ValueError(f'Author source Git blob mismatch: {path}: {git_blob}')
    raw=blob.decode('utf-8')
    classes={c.name:c for c in ast.parse(raw).body if isinstance(c, ast.ClassDef)}
    def count(name, args=(), kwargs=None):
        kwargs=kwargs or {}
        if name.startswith('nn.'):
            layer=name[3:]
            if layer=='Conv2d':
                inp,out=args[:2]
                k=args[2] if len(args)>2 else kwargs['kernel_size']
                kh,kw=(k,k) if isinstance(k,int) else k
                return int(out*(inp//kwargs.get('groups',1)*kh*kw+int(kwargs.get('bias',True)))),0
            if layer=='BatchNorm2d':
                n=args[0]
                return 2*n*int(kwargs.get('affine',True)), (2*n+1)*int(kwargs.get('track_running_stats',True))
            if layer in {'ReLU','MaxPool2d','AvgPool2d','Upsample'}: return 0,0
            raise ValueError(name)
        init=next(f for f in classes[name].body if isinstance(f,ast.FunctionDef) and f.name=='__init__')
        argnames=[x.arg for x in init.args.args][1:]
        env={k:value(v,{}) for k,v in zip(argnames[-len(init.args.defaults):],init.args.defaults)}
        env.update(zip(argnames,args)); env.update(kwargs)
        p=b=0
        for stmt in init.body:
            if isinstance(stmt,ast.Expr) and isinstance(stmt.value,ast.Call) and isinstance(stmt.value.func,ast.Attribute) and stmt.value.func.attr=='__init__':
                continue
            if not isinstance(stmt,ast.Assign) or len(stmt.targets)!=1 or not isinstance(stmt.targets[0],ast.Attribute) or not isinstance(stmt.targets[0].value,ast.Name) or stmt.targets[0].value.id!='self':
                raise ValueError((name,ast.dump(stmt)))
            if not isinstance(stmt.value,ast.Call):
                # These declarations record scalar channel/dilation metadata, not layers.
                value(stmt.value,env)
                continue
            call=stmt.value
            if isinstance(call.func,ast.Name): cname=call.func.id
            elif isinstance(call.func,ast.Attribute) and isinstance(call.func.value,ast.Name) and call.func.value.id=='nn': cname='nn.'+call.func.attr
            else: raise ValueError(ast.dump(call.func))
            pp,bb=count(cname,[value(a,env) for a in call.args],{k.arg:value(k.value,env) for k in call.keywords})
            p+=pp; b+=bb
        return p,b
    rows=[]
    for name in names:
        p,b=count(name)
        rows.append({'model':name,'declared_parameter_elements':p,'batchnorm_buffer_elements':b})
    return {'file':str(path),'source_git_blob':git_blob,'sha256':hashlib.sha256(blob).hexdigest(),'models':rows}

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-dir',type=Path,required=True,help='Directory containing exact author u2net.py and isnet.py blobs listed in sources.json')
parser.add_argument('--output',type=Path,default=Path(__file__).with_name('parameter_counts.json'))
args=parser.parse_args()
root=args.source_dir
report={'method':'static AST constructor arithmetic; not tensor/checkpoint verification',
        'torch_used':False,'weights_downloaded':False,'inference_or_training_run':False,
        'sources':[audit(root/'u2net.py',['U2NET','U2NETP'],'5b85f138f3af4e2ceae1ff07dee514c859a831af'),audit(root/'isnet.py',['ISNetDIS','ISNetGTEncoder'],'8be74bb03954496e6cb8af6e1c8e1cd2c0bbb80c')]}
args.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
