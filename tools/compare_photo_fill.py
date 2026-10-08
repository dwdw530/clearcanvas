"""Compare interpolation on the unchanged user samples, never a clean reference."""

import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from skimage.restoration import inpaint_biharmonic

from clearcanvas.photo_repair import interpolate_wide_photo_strokes
from clearcanvas.photo_watermarks import detect_photo_watermark_mask


def biharmonic(rgb, mask):
    values = inpaint_biharmonic(rgb, mask != 0, channel_axis=-1)
    return np.clip(np.rint(values * 255), 0, 255).astype(np.uint8)


def main():
    output = Path('artifacts/color-fix-0.3.7/interpolation')
    output.mkdir(parents=True, exist_ok=True)
    path = Path('artifacts/non-ai-followup-2026-10-07/06-photo-tiled-changed.png')
    with Image.open(path).convert('RGB') as image:
        rgb = np.array(image)
        mask = detect_photo_watermark_mask(image)
        for name, fill in (
            ('harmonic', interpolate_wide_photo_strokes),
            ('biharmonic', biharmonic),
            ('ns', lambda r, m: cv2.inpaint(r, m, 5, cv2.INPAINT_NS)),
            ('telea', lambda r, m: cv2.inpaint(r, m, 5, cv2.INPAINT_TELEA)),
        ):
            result = rgb.copy()
            start = time.monotonic()
            joined = cv2.dilate(mask, np.ones((9, 9), np.uint8))
            n, labels, boxes, _ = cv2.connectedComponentsWithStats(joined)
            for label in range(1, n):
                x, y, w, h, _ = boxes[label]
                x1, x2 = max(0, x-25), min(image.width, x+w+25)
                y1, y2 = max(0, y-25), min(image.height, y+h+25)
                local = mask[y1:y2, x1:x2]
                selected = (labels[y1:y2, x1:x2] == label) & (local != 0)
                repaired = fill(rgb[y1:y2, x1:x2].copy(), local)
                result[y1:y2, x1:x2][selected] = repaired[selected]
            Image.fromarray(result).save(output / f'{name}.png')
            print(name, round(time.monotonic()-start, 2), flush=True)
        boxes = [(1330, 540, 2050, 1100), (1950, 440, 2570, 940)]
        for idx, box in enumerate(boxes):
            sheet = Image.new('RGB', (1440, 1160), 'white')
            draw = ImageDraw.Draw(sheet)
            for k, name in enumerate(('harmonic', 'biharmonic', 'ns', 'telea')):
                with Image.open(output / f'{name}.png') as im:
                    crop = im.crop(box)
                    crop.thumbnail((720, 550))
                    x, y = (k % 2)*720, (k//2)*580
                    sheet.paste(crop, (x, y+25))
                    draw.text((x+10, y+4), name, fill='black', font_size=18)
            sheet.save(output / f'comparison-{idx}.jpg')


if __name__ == '__main__':
    main()
