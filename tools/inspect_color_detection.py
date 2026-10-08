"""Diagnostic masks for colored synthetic inputs; no changes to acceptance sources."""

import runpy
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from clearcanvas.photo_watermarks import detect_photo_watermark_mask


def main():
    pair = runpy.run_path('tests/test_photo_watermarks.py')['photo_pair']
    output = Path('artifacts/color-fix-0.3.7/diagnosis')
    output.mkdir(parents=True, exist_ok=True)
    for hue, opacity, corner in ((0, 255, False), (105, 160, True)):
        color = tuple(int(c) for c in cv2.cvtColor(
            np.uint8([[[hue, 230, 240]]]), cv2.COLOR_HSV2RGB
        )[0, 0])
        clean, marked = pair(color, opacity, corner)
        with clean, marked:
            before, truth = np.array(marked), np.array(clean)
            ink = np.max(np.abs(before.astype(int) - truth), axis=2) > 20
            mask = detect_photo_watermark_mask(marked)
            missed = ink & (mask == 0)
            nearby = cv2.dilate(ink.astype(np.uint8), np.ones((11, 11), np.uint8))
            extra = (mask != 0) & (nearby == 0)
            for label, selected in (('missed', missed), ('extra', extra)):
                _, _, stats, _ = cv2.connectedComponentsWithStats(selected.astype(np.uint8))
                print(hue, label, stats[1:].tolist())
            overlay = before.copy()
            overlay[mask != 0] = (0, 255, 0, 255)
            overlay[missed] = (255, 0, 0, 255)
            overlay[extra] = (255, 255, 0, 255)
            Image.fromarray(overlay).save(output / f'{hue}-mask.png')
            marked.save(output / f'{hue}-input.png')


if __name__ == '__main__':
    main()
