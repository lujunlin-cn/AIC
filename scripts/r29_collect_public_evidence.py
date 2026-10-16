#!/usr/bin/env python3
"""Export only the already-exposed R24 TRAIN/DEV evidence, never protected sources."""
import hashlib
import io
import json
import tarfile
from pathlib import Path


def main():
    root = Path('/root/autodl-tmp/experiments')
    bindings = json.loads((root / 'R24_EXEC_20261015/bindings.json').read_text())
    assert len(bindings) == 40
    assert {r['split'] for r in bindings} == {'train', 'dev'}
    assert len({r['video_id'] for r in bindings}) == 40
    out = root / 'R29_BREAKTHROUGH'
    out.mkdir(exist_ok=True)
    hashes = {}
    with tarfile.open(out / 'exposed_public_evidence.tar.gz', 'w:gz') as tar:
        for row in bindings:
            vid = row['video_id']
            assert vid and '/' not in vid and '..' not in vid
            for dirname, prefix, suffix in [
                ('R22_INSTRUMENT_20261015', 'axis_arrays/', '.npz'),
                ('R26_DIAG_20261016', 'scores_native_shape/', '.json'),
                ('R24_EXEC_20261015', '', '.npz'),
            ]:
                src = root / dirname / (prefix + vid + suffix)
                name = dirname + '/' + prefix + vid + suffix
                tar.add(src, arcname=name, recursive=False)
                hashes[name] = hashlib.sha256(src.read_bytes()).hexdigest()
        selected = []
        cfg = json.loads((root / 'R28_SPATIAL_AUDIT/R28_SPATIAL_PUBLIC_PILOT.json').read_text())
        for vid in cfg['sources']:
            cached = {int(p.stem): p for p in (Path(cfg['public_keyframe_root']) / vid).glob('*.png')}
            # Read exactly the configured public annotations; no new source selection.
            for rw, rh in cfg['ratios']:
                raw = [(Path(cfg['annotation_root']) / f'annotator_{a}' / f'{vid}_{rw}-{rh}.txt').read_text().splitlines() for a in range(1, 7)]
                assert len(set(map(len, raw))) == 1
                n = len(raw[0])
                for fraction in cfg['frame_fractions']:
                    frame = min(cached, key=lambda k: (abs(k - int(fraction*n)), k))
                    boxes = [[float(x) for x in a[frame].split(',')] for a in raw]
                    png = cached[frame]
                    selected.append(dict(video_id=vid, frame=frame, ratio=[rw,rh], gt_boxes=boxes, png_sha256=hashlib.sha256(png.read_bytes()).hexdigest()))
                    image_name = f'public_rv_images/{vid}/{frame}.png'
                    if image_name not in hashes:
                        tar.add(png, arcname=image_name, recursive=False)
                        hashes[image_name] = hashlib.sha256(png.read_bytes()).hexdigest()
        for name, data in [('sampled_rv_gt.json', selected), ('bindings.json', bindings), ('source_hashes.json', hashes)]:
            content = (json.dumps(data, indent=2)+'\n').encode()
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    print(json.dumps({'archive': str(out/'exposed_public_evidence.tar.gz'), 'bytes': (out/'exposed_public_evidence.tar.gz').stat().st_size, 'sources':40, 'protected_access':False}))


if __name__ == '__main__':
    main()
