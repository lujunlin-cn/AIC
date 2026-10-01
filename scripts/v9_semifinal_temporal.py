"""P0-2 deployment: temporal head on the 426-video semifinal drop.

The TCN was trained on 8-frame fragments sampled on a strict 1 s grid, and the
semifinal keyframes are already on that same grid (the file-name step equals the
video's fps, verified over the drop: step 30 at 30 fps, 25 at 25 fps, 60 at
60 fps).  So deployment reuses the frozen keyframes directly - no resampling,
no distribution shift between train and test time steps.

Two stages, resumable:
  --stage feats  LFM vision-tower pooled features per keyframe (NPU, 2 cards)
  --stage mask   TCN scores -> per-second keep mask -> filtered predictions

Mask policy (preregistered from the PHD2 deployment simulation, not tuned on the
test drop): keep the top `--keep-frac` of seconds per video by TCN score and drop
the rest.  Simulation on the held-out PHD2 pool gave F1 0.5576 at keep=0.80
versus 0.5176 keeping everything, with +0.0914 [+0.0779,+0.1312] against a
random-keep control at the same budget; keep=0.90 gives +0.0388 and keep=0.50
starts losing recall, so 0.80 is the interior optimum.  Per-video, no global
threshold, so a video never loses frames for another video's sake.

The spatial predictions are copied through byte-for-byte from the parent package;
only the frame list changes.  `diff_vs_parent` in the manifest proves it.
"""
import argparse, json, os, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--semifinal', type=Path, default=Path('/data/aic/semifinal_20261001'))
ap.add_argument('--parent', type=Path, required=True, help='parent predictions.jsonl')
ap.add_argument('--ckpt', type=Path, required=True)
ap.add_argument('--feat-root', type=Path, required=True)
ap.add_argument('--out-features', type=Path, required=True)
ap.add_argument('--out-predictions', type=Path, required=True)
ap.add_argument('--out-manifest', type=Path, default=None)
ap.add_argument('--stage', choices=['feats', 'mask'], required=True)
ap.add_argument('--cards', default='0,1')
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshards', type=int, default=1)
ap.add_argument('--keep-frac', type=float, default=0.80)
ap.add_argument('--model', default='/data/aic/pretrained/lfm2_5_vl_450m')
ap.add_argument('--verbose-every', type=int, default=100)
args = ap.parse_args()

KF = args.semifinal / 'keyframes' / 'keyframes'


def feats_stage():
    os.environ['ASCEND_RT_VISIBLE_DEVICES'] = args.cards
    import torch
    import torch_npu  # noqa: F401
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = False
    assert torch.npu.device_count() > 0, 'no NPU visible'
    import numpy as np
    from PIL import Image
    from transformers import AutoProcessor, Lfm2VlForConditionalGeneration

    vids = sorted(os.listdir(KF))
    mine = [v for i, v in enumerate(vids) if i % args.nshards == args.shard]
    print(f'TEMPF feats shard {args.shard}/{args.nshards}: {len(mine)} videos', flush=True)
    proc = AutoProcessor.from_pretrained(args.model, min_tiles=1, max_tiles=1,
                                         max_image_tokens=256)
    model = Lfm2VlForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float16, low_cpu_mem_usage=True,
        attn_implementation='eager').eval().to('npu')

    def pooled(im):
        msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': im},
                                             {'type': 'text', 'text': 'describe'}]}]
        x = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                     return_dict=True, return_tensors='pt')
        x = {k: v.to('npu') for k, v in x.items() if isinstance(v, torch.Tensor)}
        x['pixel_values'] = x['pixel_values'].to(torch.float16)
        with torch.no_grad():
            out = model.model.vision_tower(pixel_values=x['pixel_values'],
                                           spatial_shapes=x['spatial_shapes'],
                                           pixel_attention_mask=x['pixel_attention_mask'],
                                           return_dict=True)
        return out.last_hidden_state[0].float().mean(0).cpu().numpy().astype(np.float16)

    t0, n = time.time(), 0
    for vi, vid in enumerate(mine):
        od = args.out_features / vid
        pngs = sorted(p for p in os.listdir(KF / vid) if p.endswith('.png'))
        need = [p for p in pngs if not (od / f'{p[:-4]}.npy').exists()]
        if not need:
            continue
        od.mkdir(parents=True, exist_ok=True)
        for p in need:
            im = Image.open(KF / vid / p).convert('RGB')
            np.save(od / f'{p[:-4]}.npy', pooled(im))
            n += 1
        if vi % args.verbose_every == 0:
            el = time.time() - t0
            print(f'TEMPF {vi}/{len(mine)} frames={n} {el:.0f}s', flush=True)
    print('TEMPF_SUMMARY', json.dumps({'shard': args.shard, 'videos': len(mine),
                                       'frames': n, 'wall_s': round(time.time() - t0, 1)}),
          flush=True)


def mask_stage():
    import numpy as np
    import torch

    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)

    class TCN(torch.nn.Module):
        def __init__(self, d_in=768, ch=128, dils=(1, 2, 4)):
            super().__init__()
            self.proj = torch.nn.Conv1d(d_in, ch, 1)
            self.convs = torch.nn.ModuleList(
                [torch.nn.Conv1d(ch, ch, 3, padding=d, dilation=d) for d in dils])
            self.out = torch.nn.Conv1d(ch, 1, 1)

        def forward(self, x):
            h = self.proj(x.transpose(1, 2))
            for conv in self.convs:
                h = torch.nn.functional.gelu(conv(h) + h)
            return self.out(h).squeeze(1)

    model = TCN(ch=ck.get('ch', 128), dils=tuple(ck.get('dils', (1, 2, 4))))
    model.load_state_dict(ck['state_dict'])
    model.eval()

    parent = {json.loads(l)['video_id']: json.loads(l)
              for l in args.parent.read_text().splitlines() if l.strip()}
    out_rows, per_video = [], []
    n_drop_frames = n_keep_frames = 0
    for vid, row in sorted(parent.items(), key=lambda kv: int(kv[0])):
        d = args.out_features / vid
        keys = sorted((p.stem for p in d.glob('*.npy')), key=float) if d.exists() else []
        pred = row['predictions']
        if not keys or len(keys) < 4:
            out_rows.append(row)
            per_video.append({'video_id': vid, 'scored_seconds': len(keys),
                              'kept_frames': len(pred), 'dropped_frames': 0,
                              'fallback': 'insufficient_keyframes'})
            n_keep_frames += len(pred)
            continue
        X = np.stack([np.load(d / f'{k}.npy') for k in keys]).astype(np.float32)[None]
        with torch.no_grad():
            s = model(torch.from_numpy(X))[0].numpy()
        kidx = sorted(int(k) for k in keys)
        step = (kidx[1] - kidx[0]) if len(kidx) > 1 else 1
        step = step if step > 0 else 1
        sec_of = lambda f: f // step
        kf = {int(k): float(v) for k, v in zip(kidx, s)}
        n_keep = max(1, int(round(args.keep_frac * len(pred))))
        frames_by_sec = {}
        for p in pred:
            frames_by_sec.setdefault(sec_of(p['frame']), []).append(p['frame'])
        keep_frames, used = set(), 0
        for sec in sorted(frames_by_sec, key=lambda s2: -kf.get(
                min(kidx, key=lambda k: abs(k - s2 * step)), -1e9)):
            if used >= n_keep:
                break
            add = len(frames_by_sec[sec])
            if used + add > n_keep:
                continue
            keep_frames.update(frames_by_sec[sec])
            used += add
        if not keep_frames:
            keep_frames = {p['frame'] for p in pred[:n_keep]}
        newp = [p for p in pred if p['frame'] in keep_frames]
        out_rows.append({'video_id': vid, 'targetRatioWH': row['targetRatioWH'],
                         'model_size_mb': row.get('model_size_mb'),
                         'predictions': newp})
        n_drop_frames += len(pred) - len(newp)
        n_keep_frames += len(newp)
        per_video.append({'video_id': vid, 'scored_seconds': len(keys),
                          'kept_frames': len(newp), 'dropped_frames': len(pred) - len(newp),
                          'fallback': None})
    out_rows.sort(key=lambda r: int(r['video_id']))
    args.out_predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.out_predictions.open('w') as fh:
        for r in out_rows:
            r = {k: v for k, v in r.items() if v is not None}
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    man = {'videos': len(out_rows), 'keep_frames': n_keep_frames,
           'dropped_frames': n_drop_frames, 'keep_frac_target': args.keep_frac,
           'ckpt': str(args.ckpt), 'val_ap': ck.get('val_ap'), 'per_video': per_video}
    if args.out_manifest:
        args.out_manifest.write_text(json.dumps(man, indent=1))
    print('MASK_SUMMARY', json.dumps({k: v for k, v in man.items() if k != 'per_video'}),
          flush=True)


if __name__ == '__main__':
    (feats_stage if args.stage == 'feats' else mask_stage)()