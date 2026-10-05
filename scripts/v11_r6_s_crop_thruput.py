"""R6 S-REREAD step 2: 100-crop NPU encoding throughput (preregistered scale check).

Frame source: the SAME protocol as v8_live_extract (the B3 feature builder):
spatial_crop_v1 release -> manifest media_path -> decode to the packed
frame_idx counters -> long-side 640 BILINEAR.  Crop boxes (manifest W,H
space) are scaled by s = 640/max(W,H).  Encoder: the deployment Siglip2
NaFlex contract (min_tiles=1, max_tiles=1, max_image_tokens=256),
valid-token mean -> [D] fp32.

Also encodes each DISTINCT frame's full 640px image (S-ZERO cosine and the
S1-control repeated-full-image arm).

Run on ONE NPU: ASCEND_RT_VISIBLE_DEVICES=2.  Siglip2 patch applied.
Output: r6_s_thruput.json (+ r6_s_crop_feats.pt cache for later steps)
"""
import argparse, csv, json, os, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '16')
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
from pathlib import Path
import numpy as np
import sys
import torch
import torch_npu                                            # noqa: F401
torch.npu.set_compile_mode(jit_compile=False)
import av                                                   # noqa: E402
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('--crop-csv', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/r6_spatial_shortlist_oracle_crops.csv'))
ap.add_argument('--release', default='/data/aic/external_datasets/_releases/spatial_crop_v1')
ap.add_argument('--model-dir', default='/data/aic/pretrained/lfm2_5_vl_450m')
ap.add_argument('--n-thruput', type=int, default=100)
ap.add_argument('--out', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/r6_s_thruput.json'))
args = ap.parse_args()

sys.path.insert(0, '/root/AIC')
sys.path.insert(0, '/root/AIC/ext_data')
from v11_npu_siglip2_patch import patch_siglip2_npu         # noqa: E402
from aicext.release import Release                          # noqa: E402
from transformers import AutoProcessor, Lfm2VlForConditionalGeneration


def build():
    proc = AutoProcessor.from_pretrained(args.model_dir, min_tiles=1, max_tiles=1,
                                         max_image_tokens=256)
    model = Lfm2VlForConditionalGeneration.from_pretrained(
        args.model_dir, dtype=torch.float16,
        attn_implementation='eager').eval().to('npu')
    assert patch_siglip2_npu(model), 'siglip2 npu patch failed'
    return proc, model


def embed(proc, model, im):
    msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': im},
                                         {'type': 'text', 'text': 'describe'}]}]
    x = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                 return_dict=True, return_tensors='pt')
    x = {k: v.to('npu') for k, v in x.items() if isinstance(v, torch.Tensor)}
    with torch.no_grad():
        out = model.model.vision_tower(pixel_values=x['pixel_values'],
                                       spatial_shapes=x['spatial_shapes'],
                                       pixel_attention_mask=x['pixel_attention_mask'],
                                       return_dict=True)
    valid = int(x['pixel_attention_mask'][0].sum())
    fh, fw = [int(v) for v in x['spatial_shapes'][0]]
    return out.last_hidden_state[0, :valid].float().reshape(fh, fw, -1).mean((0, 1)).cpu()


def main():
    rows = list(csv.DictReader(args.crop_csv.open()))[:args.n_thruput]
    rel = Release('spatial_crop_v1', path=Path(args.release))
    media, WH = {}, {}
    for sp in ('train', 'dev', 'confirmation'):
        for m in rel.manifest(sp):
            vid = m['unit_id'].split(':', 1)[1]
            media[vid] = m['media_path']
            WH[vid] = (float(m['width']), float(m['height']))
    # frames to decode per (vid): decode counters from the crop csv 'frame'
    need = {}
    for r in rows:
        need.setdefault(r['vid'], set()).add(int(r['frame']))
    packed = {v: np.load(Path(args.release) / 'packed' / 'LIVE_YT_VC' / f'{v}.npz')
              for v in need}
    # decode-count -> frame counter mapping is identity in v8_live_extract
    # (packed frame_idx lists the counters saved); we re-decode to the same
    # counters directly.
    print(f'crops {len(rows)}; vids {len(need)}', flush=True)
    proc, model = build()

    imgs = {}
    t_dec = 0.0
    for vid, counters in need.items():
        t0 = time.time()
        s = 640.0 / max(*WH[vid])
        want = set(counters)
        got = {}
        with av.open(media[vid]) as cont:
            stream = cont.streams.video[0]
            fi = 0
            for frame in cont.decode(stream):
                if fi in want:
                    im = frame.to_image().convert('RGB')
                    got[fi] = im.resize(
                        (max(1, round(im.width * s)), max(1, round(im.height * s))),
                        Image.BILINEAR)
                fi += 1
                if fi > max(want):
                    break
        missing = want - set(got)
        assert not missing, f'{vid}: missing decode counters {sorted(missing)}'
        imgs[vid] = got
        t_dec += time.time() - t0

    W640 = {v: imgs[v][list(imgs[v])[0]].size for v in imgs}
    feats, fulls = {}, {}
    t0 = time.time()
    for i, r in enumerate(rows):
        vid, fr, ratio, cand = r['vid'], int(r['frame']), r['ratio'], int(r['cand'])
        im = imgs[vid][fr]
        iw, ih = im.size
        x1, y1, x2, y2 = (float(r['x1']), float(r['y1']),
                          float(r['x2']), float(r['y2']))
        # crop csv boxes are in the MANIFEST W,H of the oracle script's row
        # (already the release W,H); scale by the same s used for resize:
        Wm, Hm = WH[vid]
        s = 640.0 / max(Wm, Hm)
        box = (max(0, int(round(x1 * s))), max(0, int(round(y1 * s))),
               min(iw, max(3, int(round(x2 * s)))), min(ih, max(3, int(round(y2 * s)))))
        feats[f'{vid}|{fr}|{ratio}|{cand}'] = embed(proc, model, im.crop(box))
        if (vid, fr) not in fulls:
            fulls[(vid, fr)] = embed(proc, model, im)
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f'{i+1}/{len(rows)} crops, {el:.1f}s, {(i + 1) / el:.2f} crop/s',
                  flush=True)
    el = time.time() - t0
    out = {
        'protocol': 'R6 S-REREAD 100-crop throughput; decode+resize identical to '
                    'v8_live_extract (long side 640); deployment Siglip2 contract',
        'n_crops': len(rows), 'n_distinct_frames': len(fulls),
        'decode_wall_s': round(t_dec, 1),
        'encode_wall_s': round(el, 1),
        'crops_per_s': round(len(rows) / el, 3),
        'crops_plus_full_per_s': round((len(rows) + len(fulls)) / el, 3),
        'frames_640_size_sample': sorted(set(W640.values()))[:4],
    }
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    torch.save({'crops': feats,
                'fulls': {f'{v}|{f}': t for (v, f), t in fulls.items()}},
              args.out.with_name('r6_s_crop_feats.pt'))
    print(json.dumps(out, indent=1))
    print('WROTE', args.out, flush=True)


if __name__ == '__main__':
    main()
