"""Evaluate upstream G'MIC on the original portrait, without product changes."""

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from clearcanvas.photo_watermarks import detect_photo_watermark_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--patches', type=int, nargs='+', default=[7, 11, 17])
    parser.add_argument('--lookup', type=int, default=100)
    parser.add_argument('--method', choices=['patch', 'pde', 'flow', 'matchpatch'], default='patch')
    parser.add_argument('--iterations', type=int, default=20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    executable = Path('artifacts/gmic-probe/runtime/gmic-4.0.5-cli-win64/gmic.exe').resolve()
    source = Path('artifacts/non-ai-followup-2026-10-07/06-photo-tiled-changed.png')
    box = (1250, 500, 2100, 1120)
    with Image.open(source).convert('RGBA') as full:
        mask = detect_photo_watermark_mask(full)
        original = full.crop(box).convert('RGB')
    with Image.fromarray(mask) as full_mask:
        cropped_mask = full_mask.crop(box)
    source_path, mask_path = args.output / 'input.png', args.output / 'mask.png'
    original.save(source_path)
    selected = np.array(cropped_mask) != 0
    # Flow multiplies its diffusion tensors by mask values; use a binary 0/1
    # mask, not the 0/255 display mask used by the application.
    with Image.fromarray(selected.astype(np.uint8)) as binary_mask:
        binary_mask.save(mask_path)
    columns = len(args.patches) + 1
    sheet = Image.new('RGB', (850 * columns, 650), 'white')
    with Image.open('artifacts/face-refined/verified/06-photo-tiled-changed-result.png') as old:
        sheet.paste(old.crop(box), (0, 30))
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 5), 'REFINED BASELINE', fill='black', font_size=20)
    for index, patch in enumerate(args.patches, 1):
        result_path = args.output / f'patch-{patch}.png'
        start = time.monotonic()
        operation = '-inpaint[0]' if args.method == 'patch' else '-inpaint_pde[0]'
        parameters = (f'[1],{patch},{args.lookup},0.5,1,9,0,0.05,10,0'
                      if args.method == 'patch' else f'[1],-9,{patch},{args.iterations}')
        if args.method == 'flow':
            operation = '-inpaint_flow[0]'
            parameters = f'[1],10,{args.iterations},5,1,{patch}'
        elif args.method == 'matchpatch':
            operation = '-inpaint_matchpatch[0]'
            parameters = f'[1],0,{patch},{args.iterations},5,0,0'
        command = [str(executable), '-v', '-1', str(source_path.resolve()),
                   str(mask_path.resolve()), operation, parameters,
                   '-output[0]', str(result_path.resolve())]
        completed = subprocess.run(command, capture_output=True, timeout=240,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        if completed.returncode:
            raise RuntimeError(completed.stderr.decode('utf-8', errors='replace'))
        with Image.open(result_path).convert('RGB') as result:
            actual = np.array(result)
        before = np.array(original)
        actual[~selected] = before[~selected]
        with Image.fromarray(actual) as result:
            result.save(result_path)
            sheet.paste(result, (850 * index, 30))
        draw.text((850 * index + 10, 5), f'G-MIC {args.method.upper()} {patch}',
                  fill='black', font_size=20)
        print(f'patch={patch} seconds={time.monotonic() - start:.2f}', flush=True)
        sheet.save(args.output / 'comparison.jpg')
    original.close()
    cropped_mask.close()
    sheet.close()


if __name__ == '__main__':
    main()
