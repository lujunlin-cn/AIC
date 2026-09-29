"""V6 E0: render keyframe PNGs missing from the official keyframes cache.

The official keyframes cache (QWEN_SUBJECT_POINT_OFFICIAL_V1) was rendered at
1 fps (stride 60 @59.94fps), while the dense teacher points are at 0.5s stride
(every 30th frame).  V5 visual-head training (rv/live) had one png per dense
key (57/57 verified on rv_dev) — the official deployment must match that
convention, so this script renders exactly the missing frames from the source
mp4s listed in the intake index, scaled to the same half-resolution convention
as the existing renders, with ffmpeg autorotate disabled (index convention:
coded_pixels_no_autorotate).

Existing cache files are never modified; only missing files are added, and a
provenance manifest (which files were rendered by this script, with source
sha256 from the index) is written to --manifest.
"""
import argparse, json, pickle, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def png_size(p):
    from PIL import Image
    with Image.open(p) as im:
        return im.size


def render_one(vid, key, mp4, dest, tw, th):
    tmp = str(dest) + '.tmp.png'
    vf = f"select=eq(n\\,{key}),scale={tw}:{th}"
    r = subprocess.run(['ffmpeg', '-noautorotate', '-i', mp4, '-vf', vf,
                        '-frames:v', '1', '-y', tmp],
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0 or not Path(tmp).exists():
        return f'{vid}/{key}: ffmpeg rc={r.returncode} {r.stderr[-200:]}'
    if png_size(tmp) != (tw, th):
        return f'{vid}/{key}: size {png_size(tmp)} != {(tw, th)}'
    Path(tmp).replace(dest)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--units', default='/data/aic/experiments/V6_E0/off_e2e/off_units.pkl')
    ap.add_argument('--index', default='/data/aic/official_test_20260926/intake/index.enriched.jsonl')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--manifest', default='/data/aic/experiments/V6_E0/render_missing_manifest.json')
    a = ap.parse_args()
    units = pickle.load(open(a.units, 'rb'))
    index = {}
    for line in Path(a.index).read_text().splitlines():
        if line.strip():
            r = json.loads(line); index[r['video_id']] = r
    todo = {}  # (vid,key) -> (mp4,dest,tw,th)
    for u in units:
        vid = u['vid']; fr = Path(u['frames_dir']) / vid
        row = index.get(vid)
        if row is None:
            sys.exit(f'video {vid} not in index')
        existing = {int(f.stem) for f in fr.glob('*.png')}
        ref = next(fr.glob('*.png'), None)
        tw, th = png_size(ref) if ref else (u['W'] // 2, u['H'] // 2)
        for k in u['keys']:
            if int(k) not in existing:
                todo[(vid, int(k))] = (row['video_path'], fr / f'{k}.png', tw, th)
    print(json.dumps({'videos': len({v for v, _ in todo}), 'missing': len(todo)}), flush=True)
    errs = []; done = []
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(render_one, vid, key, mp4, dest, tw, th): (vid, key)
                for (vid, key), (mp4, dest, tw, th) in todo.items()}
        for f in futs:
            e = f.result()
            (errs if e else done).append(e or futs[f])
    prov = {'rendered': len(done), 'errors': errs,
            'convention': 'ffmpeg -noautorotate select=eq(n,K),scale=half-of-coded; provenance per index video_path/source_sha256',
            'by_video': {vid: sorted(k for v, k in done if v == vid) for vid in sorted({v for v, _ in done})}}
    Path(a.manifest).write_text(json.dumps(prov, indent=1) + '\n')
    print(json.dumps({'rendered': len(done), 'errors': len(errs), 'manifest': a.manifest}), flush=True)
    sys.exit(1 if errs else 0)


if __name__ == '__main__':
    main()
