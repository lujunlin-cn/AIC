"""Map the 174 preliminary evaluation videos back to PHD2 official intervals.

Premise under test: each preliminary mp4 is a media cut of ONE user-selected
interval from the PHD2 official testing split (testing.csv ships youtubeId,
start, duration, user_id, is_last).  If the mapping holds, the official
interval annotations give us per-video highlight ground truth for all 174
videos, and "real F on official-domain GT" becomes computable locally.

Stage 1 (this run): duration matching between mp4 metadata and csv rows.
Stage 2 (later): perceptual frame verification of matched pairs.

CPU only.
"""
import argparse, csv, json, collections
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--enriched', type=Path,
                default=Path('/data/aic/official_test_20260926/intake/index.enriched.jsonl'))
ap.add_argument('--testing-csv', type=Path,
                default=Path('/data/aic/external_datasets/PHD2/annotations/upstream_repo/testing.csv'))
ap.add_argument('--out', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/map174_stage1.json'))
args = ap.parse_args()

durs, metas = {}, {}
for line in args.enriched.read_text().splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    durs[r['video_id']] = round(r['duration'], 2)
    metas[r['video_id']] = {'duration': r['duration'], 'frame_count': r['frame_count'],
                            'fps': round(r['fps'], 3), 'w_h': [r.get('width'), r.get('height')]}

rows = list(csv.DictReader(args.testing_csv.open()))
by_dur = collections.defaultdict(list)
for r in rows:
    by_dur[round(float(r['duration']), 2)].append(r)

mp4_vals = sorted(durs.values())
exact, ambiguous, unmatched = [], 0, []
for vid, v in sorted(durs.items(), key=lambda x: int(x[0])):
    cands = by_dur.get(v, [])
    if not cands:
        unmatched.append(vid)
        continue
    users = sorted(set(c['user_id'] for c in cands))
    if len(cands) > 1:
        ambiguous += len(cands) > 1
    exact.append({'video_id': vid, 'dur': v, 'n_csv_rows': len(cands),
                  'yt_ids': sorted(set(c['youtubeId'] for c in cands))[:4],
                  'users': users[:4]})

res = {
    'n_mp4': len(durs),
    'mp4_dur_range': [mp4_vals[0], mp4_vals[-1]],
    'testing_rows': len(rows),
    'testing_is_last_true': sum(1 for r in rows if r['is_last'] == 'True'),
    'testing_unique_yt': len(set(r['youtubeId'] for r in rows)),
    'exact_dur_match_videos': len(exact),
    'exact_dur_match_share': round(len(exact) / len(durs), 4),
    'unmatched_videos': unmatched[:20],
    'n_unmatched': len(unmatched),
    'sample_matches': exact[:8],
    'multi_candidate_videos': sum(1 for e in exact if e['n_csv_rows'] > 1),
}
print(json.dumps({k: res[k] for k in ('n_mp4', 'testing_rows', 'testing_is_last_true',
                                      'testing_unique_yt', 'exact_dur_match_videos',
                                      'exact_dur_match_share', 'n_unmatched',
                                      'multi_candidate_videos')}, indent=1))
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps({'summary': res, 'metas': metas}, indent=1))
