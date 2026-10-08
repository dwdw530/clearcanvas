"""Compare existing non-neural fills on the unchanged portrait crop."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import patchmatch
from PIL import Image, ImageDraw
from skimage.restoration import inpaint_biharmonic

from clearcanvas import repair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--modes', nargs='+')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path('artifacts')
    source = root / 'non-ai-followup-2026-10-07/06-photo-tiled-changed.png'
    mask_path = root / 'patchmatch-0.3.8/verified/06-photo-tiled-changed-mask.png'
    box = (1250, 500, 2100, 1120)
    with Image.open(source).convert('RGBA') as original:
        cropped = original.crop(box)
    with Image.open(mask_path) as image:
        mask = np.array(image.crop(box))
    current = repair.restore_wide_texture

    def fill(rgb, selected, excluded, fallback, cancelled):
        if mode.startswith(('structure', 'edge')):
            bound = int(mode[4:] if mode.startswith('edge') else mode[9:])
            smooth = inpaint_biharmonic(rgb, excluded != 0, channel_axis=-1) * 255
            smooth = np.clip(smooth, fallback.astype(float) - bound,
                             fallback.astype(float) + bound)
            textured = current(rgb, selected, excluded, fallback, cancelled).astype('float32')
            detail = textured - cv2.GaussianBlur(textured, (0, 0), 4)
            candidate = smooth + detail
            if mode.startswith('edge'):
                gray = cv2.cvtColor(fallback, cv2.COLOR_RGB2GRAY).astype('float32')
                fine = gray - cv2.GaussianBlur(gray, (5, 5), 0)
                energy = np.sqrt(cv2.GaussianBlur(fine * fine, (21, 21), 0))
                low = cv2.GaussianBlur(fallback.astype('float32'), (0, 0), 4)
                gx = cv2.Sobel(low, cv2.CV_32F, 1, 0) / 8
                gy = cv2.Sobel(low, cv2.CV_32F, 0, 1) / 8
                gradient = np.sqrt(gx * gx + gy * gy).max(axis=2)
                weight = (np.clip((gradient - 0.5) / 1.5, 0, 1)
                          * np.clip((3 - energy) / 2, 0, 1))[:, :, None]
                candidate = candidate * weight + textured * (1 - weight)
            candidate = np.clip(np.rint(candidate), 0, 255).astype('uint8')
        elif mode.startswith('xphoto'):
            code = {'xphoto-shift': cv2.xphoto.INPAINT_SHIFTMAP,
                    'xphoto-fast': cv2.xphoto.INPAINT_FSR_FAST,
                    'xphoto-best': cv2.xphoto.INPAINT_FSR_BEST}[mode]
            space = cv2.COLOR_RGB2LAB if mode == 'xphoto-shift' else cv2.COLOR_RGB2BGR
            inverse = cv2.COLOR_LAB2RGB if mode == 'xphoto-shift' else cv2.COLOR_BGR2RGB
            working = cv2.cvtColor(rgb, space)
            candidate = np.zeros_like(rgb)
            cv2.xphoto.inpaint(working, (excluded == 0).astype('uint8') * 255, candidate, code)
            candidate = cv2.cvtColor(candidate, inverse)
        elif mode.startswith('color'):
            lab = cv2.cvtColor(fallback, cv2.COLOR_RGB2LAB).astype('float32')
            guide = np.stack((lab[:, :, 0], lab[:, :, 1] * 2, lab[:, :, 2]), axis=-1)
            patchmatch.set_random_seed(42)
            candidate = patchmatch.inpaint_regularity(
                rgb, selected, guide,
                global_mask=((excluded != 0) & (selected == 0)).astype('uint8'),
                patch_size=3, guide_weight=float(mode[5]),
            )
        elif mode.startswith('scaled'):
            scale = int(mode[6])
            size = (max(16, rgb.shape[1]//scale), max(16, rgb.shape[0]//scale))
            clean = rgb.copy()
            clean[excluded != 0] = fallback[excluded != 0]
            small = cv2.resize(clean, size, interpolation=cv2.INTER_AREA)
            small_mask = cv2.resize(selected, size, interpolation=cv2.INTER_AREA)
            small_mask = (small_mask > 0).astype('uint8') * 255
            small_excluded = (cv2.resize(excluded, size, interpolation=cv2.INTER_AREA) > 0)
            other = (small_excluded & (small_mask == 0)).astype('uint8')
            patchmatch.set_random_seed(42)
            candidate = cv2.resize(patchmatch.inpaint(small, small_mask, global_mask=other,
                                                    patch_size=3), rgb.shape[1::-1],
                                   interpolation=cv2.INTER_CUBIC)
        elif mode.startswith('biharmonic'):
            candidate = np.clip(np.rint(inpaint_biharmonic(
                rgb, excluded != 0, channel_axis=-1) * 255), 0, 255).astype('uint8')
        elif mode.startswith('ns'):
            candidate = cv2.inpaint(rgb, excluded, 5, cv2.INPAINT_NS)
        else:
            patchmatch.set_random_seed(42)
            kwargs = dict(global_mask=((excluded != 0) & (selected == 0)).astype('uint8'),
                          patch_size=int(mode[2]))
            if 'guide' in mode:
                yy, xx = np.mgrid[:rgb.shape[0], :rgb.shape[1]].astype('float32')
                guide = np.stack((yy, xx, np.zeros_like(xx)), axis=-1)
                candidate = patchmatch.inpaint_regularity(rgb, selected, guide,
                                                         guide_weight=0.25, **kwargs)
            else:
                candidate = patchmatch.inpaint(rgb, selected, **kwargs)
        if mode.endswith('clone'):
            pad = 16
            candidate = cv2.copyMakeBorder(candidate, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
            target = cv2.copyMakeBorder(fallback, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
            clone_mask = cv2.copyMakeBorder(selected, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
            x, y, w, h = cv2.boundingRect(clone_mask)
            candidate = cv2.seamlessClone(candidate, target, clone_mask.copy(),
                                         (x+w//2, y+h//2), cv2.NORMAL_CLONE)[pad:-pad, pad:-pad]
        result = fallback.copy()
        result[selected != 0] = candidate[selected != 0]
        return result

    modes = args.modes or ['baseline', 'tightbaseline', 'tightpm3clone', 'tightpm5clone',
                           'tightscaled2clone', 'tightscaled3clone']
    sheet = Image.new('RGB', (850*3, 650*((len(modes)+2)//3)), 'white')
    for i, mode in enumerate(modes):
        current_mask = mask
        if mode.startswith('tight'):
            mode = mode[5:]
            pixels = np.array(cropped)[:, :, :3].astype('int16')
            ink = (np.max(np.abs(pixels - (255, 0, 100)), axis=2) <= 24).astype('uint8')*255
            current_mask = cv2.dilate(ink, np.ones((3, 3), 'uint8')) & mask
        repair.restore_wide_texture = current if mode == 'baseline' else fill
        with repair._repair_image(cropped, current_mask, photographic=True) as result:
            result.save(args.output / (modes[i] + '.png'))
            x, y = i % 3 * 850, i // 3 * 650
            sheet.paste(result, (x, y+30))
            ImageDraw.Draw(sheet).text((x+5, y+5), modes[i], fill='black', font_size=20)
        print(modes[i], flush=True)
    sheet.save(args.output / 'comparison.jpg')


if __name__ == '__main__':
    main()
