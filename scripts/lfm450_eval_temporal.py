"""LFM2.5-VL-450M zero-shot temporal probe on TVSum dev (16 source videos).

Frozen protocol (splits/local_protocol_v1.json dev partition; lockbox untouched):
  - per video 16 segments of equal length; one frame at each segment centre
    (fixed sampling, timestamp MM:SS given to the model);
  - segment target = mean TVSum 20-rater score (labels_v3 proxy-v2, the
    manifest-registered labels) over that segment;
  - conditions, all through the repo's ranking_report:
      UNIFORM  constant score (the "keep everything / uniform" baseline;
               ranking metrics degenerate by construction, reported for scale)
      RANDOM   fixed-seed random scores (chance level, 200 draws averaged)
      MULTI    one query with all 16 frames in order -> 'Segment k' answer:
               top-3 segment numbers, ranked (MULTI_TOP3), and per-segment
               0-9 ratings (MULTI_RATE)
      SINGLE   16 independent single-frame queries, 0-9 highlight rating each
               (single-frame visual salience control -- NOT highlight
               understanding by itself)
  - multi-image sanity probes (per video): count the frames; identify which
    segment shows a specific frame (the model gets 16 frames and then one of
    them repeated: 'Which segment number is identical to this last image?').
Input budget: processor max_tiles pinned to 1 is not allowed (min_tiles 2), so
multi-image queries use do_image_splitting=False (one 256-token-max image per
frame; recorded).  NPU fp16 eager, jit_compile False, greedy.
"""
import argparse, json, os, re, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='/data/aic/pretrained/lfm2_5_vl_450m')
ap.add_argument('--protocol', type=Path, default=Path('/root/AIC/splits/local_protocol_v1.json'))
ap.add_argument('--labels-dir', type=Path, default=Path('/data/aic/datasets/TVSum/labels_v3'))
ap.add_argument('--videos-root', type=Path, default=Path('/data/aic/datasets/TVSum/raw/ydata-tvsum50-v1_1/videos/video'))
ap.add_argument('--cards', default='5')
ap.add_argument('--dtype', default='float16')
ap.add_argument('--n-seg', type=int, default=16)
ap.add_argument('--min-image-tokens', type=int, default=64)
ap.add_argument('--max-image-tokens', type=int, default=64)
ap.add_argument('--limit-videos', type=int, default=0)
ap.add_argument('--output-dir', type=Path, required=True)
args = ap.parse_args()

os.environ['ASCEND_RT_VISIBLE_DEVICES'] = args.cards

import torch, torch_npu  # noqa: E402
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False
import numpy as np  # noqa: E402
import av  # noqa: E402
from transformers import AutoProcessor, Lfm2VlForConditionalGeneration  # noqa: E402

import sys  # noqa: E402
sys.path.insert(0, '/root/AIC')
from aic.temporal_metrics import ranking_report  # noqa: E402

N = args.n_seg
P_TOP3 = (f'The {N} images are frames from {N} consecutive segments of one video, in time order '
          f'(Segment 1 to Segment {N}). Which 3 segments are the most important highlights of this video? '
          f'Answer with only the 3 segment numbers, most important first, separated by commas.')
P_RATE = (f'The {N} images are frames from {N} consecutive segments of one video, in time order '
          f'(Segment 1 to Segment {N}). Rate how much each segment is a highlight of this video from 0 (boring) '
          f'to 9 (key highlight). Answer with exactly {N} integers separated by commas, one per segment in order.')
P_SINGLE = ('This is one frame from a video. Rate how likely this moment is a highlight of the video, '
            'from 0 (boring) to 9 (key highlight). Answer with one integer only.')
P_COUNT = f'How many images are shown above? Answer with one integer only.'
P_MATCH = (f'The first {N} images are Segment 1 to Segment {N} of a video. The last image is an exact copy of one '
           f'of those segments. Which segment number is it? Answer with one integer only.')


def ints(reply):
    return [int(x) for x in re.findall(r'\d+', reply)]


def segment_frames(path, n):
    """One RGB frame at each of n equal-segment centres, plus frame indices / fps."""
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        fps = float(st.average_rate)
        nb = st.frames or int(round(float(c.duration) / 1e6 * fps))
        edges = np.linspace(0, nb, n + 1)
        centres = np.clip(np.round((edges[:-1] + edges[1:]) / 2).astype(int), 0, nb - 1)
        want = {int(i): k for k, i in enumerate(centres)}
        imgs = [None] * n
        for i, fr in enumerate(c.decode(st)):
            if i in want:
                imgs[want[i]] = fr.to_image()
            if i > centres.max():
                break
    return imgs, centres, edges, fps, nb


def mmss(sec):
    return f'{int(sec) // 60:02d}:{int(sec) % 60:02d}'


t0 = time.time()
proc = AutoProcessor.from_pretrained(args.model, do_image_splitting=False,
                                     min_image_tokens=args.min_image_tokens,
                                     max_image_tokens=args.max_image_tokens)
model = Lfm2VlForConditionalGeneration.from_pretrained(
    args.model, dtype=getattr(torch, args.dtype), low_cpu_mem_usage=True,
    attn_implementation='eager').eval().to('npu')
load_s = time.time() - t0
proto = json.loads(args.protocol.read_text())
vids = proto['dev_video_ids'][:args.limit_videos or None]
args.output_dir.mkdir(parents=True, exist_ok=True)
tq = 0.0
nq = 0


def ask(content, max_new):
    global tq, nq
    x = proc.apply_chat_template([{'role': 'user', 'content': content}], tokenize=True,
                                 add_generation_prompt=True, return_dict=True, return_tensors='pt').to('npu')
    x['pixel_values'] = x['pixel_values'].to(model.dtype)
    t = time.time()
    with torch.no_grad():
        out = model.generate(**x, max_new_tokens=max_new, do_sample=False)
    torch.npu.synchronize()
    tq += time.time() - t
    nq += 1
    return proc.batch_decode(out[:, x['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip(), int(x['input_ids'].shape[1])


def seg_list(imgs, stamps):
    c = []
    for k, (im, s) in enumerate(zip(imgs, stamps)):
        c += [{'type': 'text', 'text': f'Segment {k + 1} ({s}):'}, {'type': 'image', 'image': im}]
    return c


rng = np.random.default_rng(0)
rows_f = (args.output_dir / 'rows.jsonl').open('w')
res = []
for vid in vids:
    lab = np.load(args.labels_dir / f'{vid}.proxy-v2.npz', allow_pickle=True)
    labels, mask = lab['labels'].astype(float), lab['mask'].astype(bool)
    imgs, centres, edges, fps, nb = segment_frames(args.videos_root / f'{vid}.mp4', N)
    if any(im is None for im in imgs):
        rows_f.write(json.dumps({'vid': vid, 'status': 'decode_short'}) + '\n')
        continue
    L = len(labels)
    tgt = np.array([labels[int(a * L / nb):max(int(b * L / nb), int(a * L / nb) + 1)][mask[int(a * L / nb):max(int(b * L / nb), int(a * L / nb) + 1)]].mean()
                    for a, b in zip(edges[:-1], edges[1:])])
    stamps = [mmss(i / fps) for i in centres]
    seg = seg_list(imgs, stamps)

    r_top3, ntok = ask(seg + [{'type': 'text', 'text': P_TOP3}], 16)
    top3 = [k for k in dict.fromkeys(ints(r_top3)) if 1 <= k <= N][:3]
    s_top3 = np.zeros(N)
    for rank, k in enumerate(top3):
        s_top3[k - 1] = 3 - rank
    r_rate, _ = ask(seg + [{'type': 'text', 'text': P_RATE}], 64)
    rate = [min(v, 9) for v in ints(r_rate)]
    rate_ok = len(rate) >= N
    s_rate = np.asarray((rate + [0] * N)[:N], float)
    s_single, r_single = [], []
    for im in imgs:
        r, _ = ask([{'type': 'image', 'image': im}, {'type': 'text', 'text': P_SINGLE}], 4)
        v = ints(r)
        r_single.append(r)
        s_single.append(float(min(v[0], 9)) if v else np.nan)
    s_single = np.asarray(s_single)
    single_ok = int(np.isfinite(s_single).sum())
    s_single = np.nan_to_num(s_single, nan=float(np.nanmean(s_single)) if single_ok else 0.)
    r_count, _ = ask(seg + [{'type': 'text', 'text': P_COUNT}], 6)
    probe_k = int(rng.integers(0, N))
    r_match, _ = ask(seg + [{'type': 'text', 'text': 'Last image:'}, {'type': 'image', 'image': imgs[probe_k]},
                            {'type': 'text', 'text': P_MATCH}], 6)
    rand = [ranking_report(rng.random(N), tgt) for _ in range(200)]
    rr = {'MULTI_TOP3': ranking_report(s_top3, tgt) if top3 else None,
          'MULTI_RATE': ranking_report(s_rate, tgt) if rate_ok else None,
          'SINGLE': ranking_report(s_single, tgt),
          'UNIFORM': ranking_report(np.zeros(N), tgt),
          'RANDOM': {k: float(np.mean([d[k] for d in rand if d[k] is not None])) for k in rand[0]}}
    row = {'vid': vid, 'n_frames_video': int(nb), 'fps': fps, 'centres': centres.tolist(), 'stamps': stamps,
           'target': tgt.round(4).tolist(), 'prompt_tokens_multi': ntok,
           'top3': top3, 'top3_reply': r_top3, 'rate': rate, 'rate_ok': rate_ok, 'rate_reply': r_rate[:160],
           'rate_std': float(np.std(s_rate)), 'single': s_single.tolist(), 'single_ok': single_ok,
           'single_std': float(np.std(s_single)), 'count_reply': r_count, 'count_ok': ints(r_count)[:1] == [N],
           'match_target': probe_k + 1, 'match_reply': r_match,
           'match_ok': ints(r_match)[:1] == [probe_k + 1], 'metrics': rr,
           'top3_hit_gt_top3': len(set(top3) & set((np.argsort(-tgt)[:3] + 1).tolist()))}
    rows_f.write(json.dumps(row) + '\n')
    rows_f.flush()
    res.append(row)
    print(f'{vid} top3={top3} rate_ok={rate_ok} count={r_count!r} match={r_match!r}/{probe_k + 1} '
          f'avg {tq / max(nq, 1):.2f}s', flush=True)

keys = ['spearman', 'kendall_tau_b', 'ndcg', 'ndcg_at_15pct', 'top15_mean_relevance']
summ = {'n_videos': len(res), 'n_seg': N, 'n_queries': nq, 'avg_s_per_query': round(tq / max(nq, 1), 3),
        'peak_mem_gib': round(torch.npu.max_memory_allocated() / 2**30, 3), 'load_s': round(load_s, 1),
        'image_tokens': [args.min_image_tokens, args.max_image_tokens], 'do_image_splitting': False,
        'prompts': {'TOP3': P_TOP3, 'RATE': P_RATE, 'SINGLE': P_SINGLE, 'COUNT': P_COUNT, 'MATCH': P_MATCH},
        'parse': {'top3_nonempty': sum(bool(r['top3']) for r in res), 'rate_ok': sum(r['rate_ok'] for r in res),
                  'single_all16': sum(r['single_ok'] == N for r in res)},
        'probes': {'count_ok': sum(r['count_ok'] for r in res), 'match_ok': sum(r['match_ok'] for r in res),
                   'match_chance': 1 / N},
        'rate_constant_videos': sum(r['rate_std'] == 0 for r in res),
        'single_constant_videos': sum(r['single_std'] == 0 for r in res),
        'top3_overlap_mean': float(np.mean([r['top3_hit_gt_top3'] for r in res])) if res else None,
        'top3_overlap_chance': 9 / N,
        'mean': {}}
for cond in ('MULTI_TOP3', 'MULTI_RATE', 'SINGLE', 'UNIFORM', 'RANDOM'):
    ok = [r['metrics'][cond] for r in res if r['metrics'][cond]]
    summ['mean'][cond] = {'n': len(ok), **{k: (float(np.mean([d[k] for d in ok if d[k] is not None]))
                                               if any(d[k] is not None for d in ok) else None) for k in keys}}
(args.output_dir / 'summary.json').write_text(json.dumps(summ, indent=1, ensure_ascii=False) + '\n')
print('SUMMARY', json.dumps(summ['mean']), json.dumps(summ['probes']), json.dumps(summ['parse']), flush=True)
