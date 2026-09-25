"""Bounded public-video pHash screen, without labels or model selection."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np


def fingerprint(path, count=9):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f'Cannot open {path}')
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    targets = np.unique(np.rint(np.linspace(0, n - 1, count)).astype(int))
    hashes = []
    for i in targets:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f'Cannot decode {path}:{i}')
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 32)).astype(np.float32)
        coef = cv2.dct(gray)[:8, :8].ravel()[1:]
        bits = coef > np.median(coef)
        hashes.append(bits)
    cap.release()
    return targets.tolist(), np.stack(hashes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tvsum-manifest', required=True)
    ap.add_argument('--summe-results', action='append', required=True)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    cv2.setNumThreads(2)
    start = time.perf_counter()
    tv = [json.loads(x) for x in Path(a.tvsum_manifest).read_text().splitlines()]
    sm = []
    for path in a.summe_results:
        sm.extend(r for r in json.loads(Path(path).read_text())['manifest'] if r['included'])
    rows = {'TVSum': tv, 'SumMe': sm}
    cache = {}
    inventory = []
    for dataset, records in rows.items():
        for r in records:
            p = Path(r['path'])
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            if digest != r['sha256']:
                raise ValueError('Source hash drift: ' + str(p))
            indices, h = fingerprint(p)
            cache[dataset, r['video_id']] = h
            inventory.append({'dataset': dataset, 'video_id': r['video_id'], 'sha256': digest,
                              'frame_indices': indices, 'phash_63bits': [''.join(map(str, x.astype(int))) for x in h]})
    pairs = []
    for s in sm:
        h = cache['SumMe', s['video_id']]
        for t in tv:
            other = cache['TVSum', t['video_id']]
            distances = (h[:, None] != other[None]).sum(-1)
            close = distances.min(1)
            pairs.append({'summe_video': s['video_id'], 'tvsum_video': t['video_id'],
                          'exact_file_match': s['sha256'] == t['sha256'],
                          'min_hamming': int(distances.min()), 'median_nearest_hamming': float(np.median(close)),
                          'summe_frames_with_near_match': int((close <= 6).sum())})
    pairs.sort(key=lambda r: (-r['summe_frames_with_near_match'], r['median_nearest_hamming'], r['min_hamming']))
    candidates = [p for p in pairs if p['exact_file_match'] or p['summe_frames_with_near_match'] >= 3]
    result = {'protocol': 'CROSS_DATASET_SPARSE_PHASH_V1', 'frames_per_video': 9,
              'rule_fixed_before_result': 'review pairs with >=3 SumMe sampled frames having nearest pHash Hamming <=6 among 9 TVSum frames',
              'n_tvsum': len(tv), 'n_summe': len(sm), 'n_pairs': len(pairs),
              'exact_duplicates': sum(p['exact_file_match'] for p in pairs),
              'review_candidates': candidates, 'nearest_pairs': pairs[:20], 'inventory': inventory,
              'elapsed_seconds': time.perf_counter() - start,
              'limitations': 'Sparse pHash only; missed short/reframed clips possible. A clean screen is not proof against all pretraining or video overlap.',
              'official_f_video': None, 'competition_score': None}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['n_pairs', 'exact_duplicates', 'review_candidates', 'elapsed_seconds']}))


if __name__ == '__main__':
    main()
