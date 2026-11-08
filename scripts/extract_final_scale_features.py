"""Label-blind CPU RGB descriptors for a bounded, immutable P1 cache cohort."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.run_final_p1_cache import read_rows, sha, write


def rgb_descriptors(payload):
    with Image.open(io.BytesIO(payload)) as image:
        rgb = np.asarray(image.convert('RGB').resize((64, 64)), dtype=float)/255
    gray = rgb.mean(2)
    hist = np.histogram(gray, bins=16, range=(0, 1))[0]/gray.size
    # Transpose-invariant content summaries, deliberately not called subject
    # detections or a complete measure of semantic Scale predictability.
    return [float(gray.mean()), float(gray.std()),
        float((rgb.max(2)-rgb.min(2)).mean()),
        float(abs(np.diff(gray, axis=0)).mean()+abs(np.diff(gray, axis=1)).mean()),
        float(-np.sum(hist*np.log(hist+1e-12))),
        *[float(np.mean(rgb[:, :, i])) for i in range(3)]]


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--records', type=Path, required=True)
    p.add_argument('--image-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    records = []
    for path in sorted(a.records.glob('*.json')):
        records.extend(json.loads(path.read_text()))
    wanted = {r['image_sha256'] for r in records}
    features, inputs = {}, {}
    for r in read_rows(a.labels):
        h = r['image_sha256']
        if h not in wanted or h in features:
            continue
        path = (a.image_root/r['input_image']).resolve()
        if not path.is_relative_to(a.image_root.resolve()):
            raise ValueError('image outside approved root')
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != h:
            raise ValueError('visual input SHA mismatch')
        features[h] = rgb_descriptors(payload); inputs[h] = r['input_image']
    if set(features) != wanted:
        raise ValueError('not all P1 sample image hashes have features')
    write(a.output, {'features_by_image_sha256': features, 'input_paths': inputs,
        'descriptors': ['gray_mean', 'gray_std', 'color_range', 'edge_mean', 'entropy', 'R_mean', 'G_mean', 'B_mean'],
        'labels_sha256': sha(a.labels), 'descriptor_code_sha256': sha(__file__),
        'CUDA_initialized': False, 'GT_not_used_for_feature_extraction': True,
        'view_contract': 'transpose-invariant RGB summaries; A/B fitted separately',
        'limitation': 'low-level descriptors, not subject count/size or a VLM semantic feature bank'})


if __name__ == '__main__':
    main()
