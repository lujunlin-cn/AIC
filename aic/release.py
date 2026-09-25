"""Run a frozen, hash-verified engineering candidate without parameter search."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def verify_weights(manifest: dict, candidate: dict, weights_dir: Path) -> tuple[dict, int]:
    """Audit all and only weights used by this candidate before loading models."""
    inventory = {}
    names = [candidate['temporal_weight']]
    if candidate.get('detector_weight'):
        names.append(candidate['detector_weight'])
    for name in names:
        expected = manifest['weights'][name]
        path = (weights_dir / expected['filename']).resolve()
        if not path.is_relative_to(weights_dir.resolve()):
            raise ValueError('Weight path must stay inside weights directory')
        size = path.stat().st_size
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if size != expected['bytes'] or digest != expected['sha256']:
            raise ValueError(f'Weight integrity mismatch: {name}')
        inventory[name] = {'path': str(path), 'bytes': size, 'sha256': digest}
    total = sum(v['bytes'] for v in inventory.values())
    if total != candidate['expected_loaded_bytes']:
        raise ValueError('Candidate total weight bytes do not match manifest')
    return inventory, total


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--weights-dir', required=True)
    parser.add_argument('--index', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--video-root')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--stage', choices=['preliminary', 'final'], default='final')
    args = parser.parse_args(argv)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    candidate = manifest['candidates'][args.candidate]
    inventory, total = verify_weights(manifest, candidate, Path(args.weights_dir))
    # Refuse to silently overwrite a previously generated submission/evidence pair.
    if Path(args.output).exists() or Path(args.report).exists():
        raise FileExistsError('Use fresh output and report paths for each run')
    import torch
    from .inference import run_inference
    from .contract import load_jsonl
    torch.set_num_threads(4)
    cuda = torch.device(args.device).type == 'cuda'
    if cuda:
        torch.cuda.set_device(torch.device(args.device))
        torch.cuda.reset_peak_memory_stats(args.device)
    start = time.perf_counter()
    report = run_inference(
        args.index, args.output,
        model_path=inventory[candidate['temporal_weight']]['path'],
        detector_path=inventory[candidate['detector_weight']]['path'] if candidate.get('detector_weight') else None,
        video_root=args.video_root, device=args.device, stage=args.stage,
        sample_fps=candidate['sample_fps'], threshold=candidate['threshold'],
        spatial_mode=candidate['spatial_mode'], spatial_protocol=candidate['spatial_protocol'],
        batch_size=candidate['batch_size'], postprocess_config=candidate['postprocess_config'],
    )
    if cuda:
        torch.cuda.synchronize(args.device)
    if report['loaded_weight_bytes'] != total:
        raise ValueError('Runtime loaded bytes differ from frozen candidate inventory')
    metadata = {r['video_id']: r for r in load_jsonl(args.index)}
    per_video = [{'video_id': r['video_id'], 'selected_frames': len(r['predictions']),
                  'frame_count': metadata[r['video_id']]['frame_count'],
                  'selection_rate': len(r['predictions']) / metadata[r['video_id']]['frame_count']}
                 for r in load_jsonl(args.output)]
    report.update({
        'release_id': manifest['release_id'], 'candidate_id': args.candidate,
        'candidate': candidate, 'inventory': inventory,
        'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'index_sha256': hashlib.sha256(Path(args.index).read_bytes()).hexdigest(),
        'elapsed_seconds': time.perf_counter() - start,
        'peak_vram_bytes': torch.cuda.max_memory_allocated(args.device) if cuda else None,
        'per_video': per_video,
        'empty_prediction_rate': sum(r['selected_frames'] == 0 for r in per_video) / len(per_video),
        'official_f_video': None, 'competition_score': None,
        'status': 'engineering_verified_candidate_not_officially_scored',
    })
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
