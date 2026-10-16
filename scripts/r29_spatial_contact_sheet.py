#!/usr/bin/env python3
"""Visualize frozen public-cache pilot outputs; never creates human labels."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--oracle', type=Path, required=True)
    parser.add_argument('--sampled-gt', type=Path, required=True)
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.oracle.read_text())
    gt = {(r['video_id'], r['frame'], tuple(r['ratio'])): r for r in json.loads(args.sampled_gt.read_text())}
    rows = {(r['video_id'], r['frame'], r['axis']): r for r in data['per_frame']}
    selections = [('034', 240), ('034', 360), ('033', 420), ('033', 630), ('032', 90)]
    cell_w, cell_h = 400, 280
    sheet = Image.new('RGB', (cell_w * 3, cell_h * 2), 'white')
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 14)
    small = ImageFont.truetype(font_path, 12)
    for index, (source, frame) in enumerate(selections):
        row = rows[source, frame, 'x']
        reference = gt[source, frame, (1, 3)]
        path = args.image_root / source / f'{frame}.png'
        if hashlib.sha256(path.read_bytes()).hexdigest() != reference['png_sha256']:
            raise ValueError('PNG identity mismatch')
        original = Image.open(path).convert('RGB')
        image = original.resize((400, 225), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(image)
        scale = 400 / original.width
        center = [(original.width - 120) / 2, 0, (original.width + 120) / 2, original.height]
        def box(rect, color, width):
            scaled = [value * scale for value in rect]
            scaled[2] = min(scaled[2], image.width - 1)
            scaled[3] = min(scaled[3], image.height - 1)
            draw.rectangle(scaled, outline=color, width=width)
        box(center, '#1868ff', 5)
        box(row['qwen_crop'], '#ff1525', 2)
        box(row['oracle_crop'], '#00d44a', 3)
        x, y = (index % 3) * cell_w, (index // 3) * cell_h
        sheet.paste(image, (x, y + 25))
        canvas = ImageDraw.Draw(sheet)
        canvas.text((x + 5, y + 4), f'RV cache {source} f{frame}; 1:3 / x-slide', fill='black', font=font)
        canvas.text((x + 5, y + 252), f'Qwen {row["qwen"]:.3f} | center {row["center"]:.3f} | oracle {row["oracle"]:.3f}', fill='black', font=small)
    canvas = ImageDraw.Draw(sheet)
    x, y = 2 * cell_w + 12, cell_h + 15
    for offset, (color, label) in enumerate([('#ff1525', 'Qwen frozen coordinate'), ('#1868ff', 'Center (blue visible behind red if equal)'), ('#00d44a', '129-candidate GT-assisted oracle')]):
        canvas.line((x, y + offset * 35, x + 30, y + offset * 35), fill=color, width=5)
        canvas.text((x + 38, y + offset * 35 - 8), label, fill='black', font=small)
    lines = ['Oracle maximizes mean IoU over six existing', 'human boxes. Oracle is not itself a human box.', 'Only four exposed public sources; no B3 control.', 'Cache names/SHA checked; raw-media binding', 'remains unverified. No official-test images.']
    for offset, line in enumerate(lines):
        canvas.text((x, y + 110 + offset * 22), line, fill='black', font=small)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.quantize(colors=96).save(args.output, optimize=True)
    print(json.dumps({'output': str(args.output), 'bytes': args.output.stat().st_size,
                      'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      'scope': 'Diagnostic overlay of existing public cache; no new human labels or media QC'}))


if __name__ == '__main__':
    main()
