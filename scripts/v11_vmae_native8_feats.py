"""Native-framerate VideoMAE features for PHD2 fragments.

For each fragment: decode the SOURCE video over [t0, t0+L], split into 4
equal sub-windows, 16 native frames each (~8 fps at 30 fps sources, 4x
the time resolution of the 1 fps keyframe grid), and encode each
sub-window under TWO conditions:
  ord: the 16 native frames in time order
  rep: the sub-window middle frame repeated 16x (same decode, no motion)
Output per fragment: {ord4 [4,8,768], rep4 [4,8,768]}.
"""
import argparse, json, os, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--index', type=Path,
                default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/index_clean.jsonl'))
ap.add_argument('--media-dir', type=Path,
                default=Path('/data/aic/external_datasets/PHD2/raw/youtube'))
ap.add_argument('--weights', default='/data/aic/pretrained/videomaev2')
ap.add_argument('--bridge-mod',
                default=Path('/data/aic/experiments_910a/LFM_V11/v11_m01_conv3d_bridge.py'))
ap.add_argument('--out', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/phd2_vmae_native8'))
ap.add_argument('--selections', type=Path,
                default=Path('/data/aic/external_datasets/PHD2/annotations/selections/train.json'))
ap.add_argument('--cards', default='0')
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshards', type=int, default=1)
ap.add_argument('--nsub', type=int, default=4)
args = ap.parse_args()

os.environ['ASCEND_RT_VISIBLE_DEVICES'] = args.cards
import numpy as np
import torch
import torch_npu
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False
assert torch.npu.device_count() > 0, 'no NPU'
from safetensors.torch import load_file
import av
import sys
sys.path.insert(0, str(Path(args.weights).parent))
from videomaev2.modeling_videomaev2 import VisionTransformer
import importlib.util as _ilu
spec = _ilu.spec_from_file_location('v11_bridge', args.bridge_mod)
bridge_mod = _ilu.module_from_spec(spec)
spec.loader.exec_module(bridge_mod)

cfg = json.load(open(Path(args.weights) / 'config.json'))['model_config']
model = VisionTransformer(**cfg)
sd = load_file(Path(args.weights) / 'model.safetensors')
sd = {k[len('model.'):] if k.startswith('model.') else k: v for k, v in sd.items()}
model.load_state_dict(sd)
bridge_mod.patch_videomae_conv3d(model)
model = model.float().eval().to('npu')
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def decode_window(c, st, t0, subL, n=16):
    """16 native frames equally spaced over [t0, t0+subL]."""
    times = t0 + subL * (np.arange(n) + 0.5) / n
    out, last_t = [], None
    for t in times:
        c.seek(int(t * av.time_base), stream=st, backward=True)
        got = None
        for fr in c.decode(st):
            if fr.time is None or fr.time >= t - 0.25:
                got = fr
                break
        if got is not None:
            out.append(got.reformat(width=224, height=224, format='rgb24').to_ndarray())
    if not out:
        return None
    while len(out) < n:
        out.append(out[-1])
    return np.stack(out)


def encode(x):  # x: np.uint8 NHWC stack
    x = torch.from_numpy(x)[None].float().permute(0, 4, 1, 2, 3).contiguous().to('npu')
    with torch.autocast('npu', dtype=torch.float16):
        h = model.patch_embed(x)
        h = h + model.pos_embed.expand(1, -1, -1).type_as(h).to(h.device).clone().detach()
        h = model.pos_drop(h)
        for blk in model.blocks:
            h = blk(h)
    L = h.shape[1]
    return h.float().view(1, 8, L // 8, 768).mean(2)[0].cpu().numpy().astype(np.float32)


rows = [json.loads(l) for l in args.index.read_text().splitlines() if l.strip()]
sel = json.loads(args.selections.read_text())
mixed = set()
for r in rows:
    ivs = []
    for recs in sel.get(r['src'], {}).values():
        for rec in recs:
            if float(rec['t1']) > float(rec['t0']):
                ivs.append(rec)
    if ivs:
        mixed.add(r['video_id'])
rows = [r for r in rows if r['video_id'] in mixed]
done_ids = set()
if args.out.exists():
    done_ids = {p.stem for p in args.out.glob('*.npz')}
todo = [r for r in rows if r['video_id'] not in done_ids]
todo.sort(key=lambda r: r['video_id'])
mine = [r for i, r in enumerate(todo) if i % args.nshards == args.shard]
print(f'NAT shard {args.shard}/{args.nshards}: {len(mine)} fragments', flush=True)
t0, n = time.time(), 0
args.out.mkdir(parents=True, exist_ok=True)
with torch.no_grad():
    for vi, r in enumerate(mine):
        src = args.media_dir / f"{r['src']}.mp4"
        if not src.exists():
            continue
        try:
            with av.open(str(src)) as c:
                st = c.streams.video[0]
                subL = float(r['L']) / args.nsub
                ord_ws, rep_ws = [], []
                ok = True
                for wi in range(args.nsub):
                    sub_t0 = float(r['t0']) + wi * subL
                    frames = decode_window(c, st, sub_t0, subL)
                    if frames is None:
                        ok = False
                        break
                    ord_ws.append(encode(frames))
                    rep_ws.append(encode(np.repeat(frames[7:8], 16, 0)))
                if not ok:
                    continue
            np.savez(args.out / f"{r['video_id']}.npz",
                     ord4=np.stack(ord_ws).astype(np.float32),
                     rep4=np.stack(rep_ws).astype(np.float32),
                     src=r['src'], t0=float(r['t0']), L=float(r['L']))
            n += 1
            if n % 100 == 0:
                print(f'NAT {n} frags {time.time() - t0:.0f}s', flush=True)
        except Exception as e:
            if n == 0:
                import traceback
                traceback.print_exc()
            continue
print('NAT_SUMMARY', json.dumps({'shard': args.shard, 'frags': n,
                                 'wall_s': round(time.time() - t0, 1)}), flush=True)
