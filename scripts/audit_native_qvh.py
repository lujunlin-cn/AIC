"""Bounded native QVH media/annotation audit. No target conversion or training."""
import argparse
import hashlib
import json
from pathlib import Path
import av
from scripts.audit_qvh_source_split import original_source


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--videos', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int, default=5)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    media = {f.stem: f for f in a.videos.rglob('*.mp4')}
    by_split = {s: [json.loads(x) for x in (a.annotations / (s + '.jsonl')).read_text().splitlines()]
                for s in ('train', 'val')}
    source_sets = {s: {original_source(r['vid']) for r in rows} for s, rows in by_split.items()}
    overlap = source_sets['train'] & source_sets['val']
    summary = {'protocol': 'QVH_NATIVE_MEDIA_AUDIT_V1', 'official_aic_gt': False,
               'annotation_scope': 'query-conditioned human saliency and relevant windows; no crop GT',
               'cross_split_original_sources': len(overlap), 'splits': {}}
    for split, rows in by_split.items():
        matches = sorted({r['vid'] for r in rows} & media.keys())
        # Selection occurs before decoding or looking at metric values.
        selected = [v for v in matches if original_source(v) not in overlap][:a.limit]
        records = []
        for vid in selected:
            ann = [r for r in rows if r['vid'] == vid]
            first = last = previous = None
            count = 0; monotonic = True
            with av.open(str(media[vid])) as container:
                stream = container.streams.video[0]
                fps = float(stream.average_rate)
                width, height = stream.width, stream.height
                for frame in container.decode(stream):
                    count += 1
                    if frame.pts is None:
                        monotonic = False
                        continue
                    t = float(frame.pts * frame.time_base)
                    if previous is not None and t <= previous: monotonic = False
                    if first is None: first = t
                    previous = last = t
            duration = (last - first + 1 / fps) if first is not None else 0
            aligned = all(abs(float(r['duration']) - duration) < max(.1, 2/fps) for r in ann)
            schema = all(len(r['relevant_clip_ids']) == len(r['saliency_scores'])
                         and all(len(x) == 3 and all(0 <= z <= 4 for z in x) for x in r['saliency_scores'])
                         and all(0 <= x < duration / 2 for x in r['relevant_clip_ids'])
                         and all(0 <= b <= e <= float(r['duration']) for b,e in r['relevant_windows']) for r in ann)
            rec = {'vid':vid, 'source_group':original_source(vid), 'split':split, 'video_path':str(media[vid]),
                   'decoded_frames':count, 'fps':fps, 'duration':duration, 'width':width, 'height':height,
                   'pts_monotonic':monotonic, 'duration_matches_annotation':aligned,
                   'annotation_schema_valid':schema, 'query_count':len(ann),
                   'status':'validated' if monotonic and aligned and schema else 'failed'}
            records.append(rec); print(json.dumps(rec), flush=True)
        (a.output / (split + '.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in records))
        summary['splits'][split] = {'query_rows':len(rows), 'unique_videos':len({r['vid'] for r in rows}),
                                  'raw_id_matches':len(matches), 'audited':len(records),
                                  'passed':sum(r['status']=='validated' for r in records),
                                  'annotation_sha256':hashlib.sha256((a.annotations/(split+'.jsonl')).read_bytes()).hexdigest()}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__ == '__main__': main()
