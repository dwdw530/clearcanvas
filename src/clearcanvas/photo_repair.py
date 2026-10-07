"""Bounded local texture transfer for photographic watermark strokes."""

import cv2
import numpy as np

from clearcanvas.documents import DocumentError


def restore_photo_texture(rgb, mask, excluded, fallback, cancelled, step_budget):
    """Copy small matching clean patches; retain interpolation when unsupported.

    `excluded` includes all watermark regions and transparent pixels, even those
    outside the component currently being repaired. Neither can be a donor.
    """
    missing = mask != 0
    if (
        not step_budget or np.count_nonzero(missing) > 50_000
        or mask.size > 500_000 or min(mask.shape) < 7
    ):
        return fallback, 0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    ring = (cv2.dilate(mask, np.ones((21, 21), np.uint8)) != 0) & (excluded == 0)
    detail = gray - cv2.GaussianBlur(gray, (5, 5), 0)
    if np.count_nonzero(ring) < 49 or np.std(detail[ring]) < 1.0:
        return fallback, 0

    radius, size, search_radius = 3, 7, 60
    valid = cv2.dilate(excluded, np.ones((size, size), np.uint8)) == 0
    valid[:radius] = valid[-radius:] = False
    valid[:, :radius] = valid[:, -radius:] = False
    if not np.any(valid):
        return fallback, 0
    other_marks = (excluded != 0) & ~missing
    height, width = mask.shape
    source = rgb.astype(np.float32)
    pixels = rgb.copy()
    deferred = np.zeros(mask.shape, dtype=bool)
    steps = 0
    for _ in range(min(step_budget, 2000)):
        if cancelled():
            raise DocumentError("已取消。", "cancelled")
        if not np.any(missing):
            break
        counts = cv2.boxFilter(
            (missing | other_marks).astype(np.float32), -1, (size, size), normalize=False
        )
        # Fill thin tips with the strongest known surroundings first. Already
        # filled pixels guide matching, but never become source patches.
        priority = np.where(missing & ~deferred, counts, np.inf)
        priority[:radius] = priority[-radius:] = np.inf
        priority[:, :radius] = priority[:, -radius:] = np.inf
        y, x = np.unravel_index(priority.argmin(), priority.shape)
        if not np.isfinite(priority[y, x]) or priority[y, x] >= size * size - 3:
            break
        target_slice = np.s_[y-radius:y+radius+1, x-radius:x+radius+1]
        steps += 1
        known = (~(missing | other_marks)[target_slice]).astype(np.float32)
        target = pixels[target_slice].astype(np.float32)
        x1, x2 = max(radius, x - search_radius), min(width - radius, x + search_radius + 1)
        y1, y2 = max(radius, y - search_radius), min(height - radius, y + search_radius + 1)
        weights = np.repeat(known[:, :, None], 3, axis=2)
        scores = cv2.matchTemplate(
            source[y1-radius:y2+radius, x1-radius:x2+radius],
            target, cv2.TM_SQDIFF, mask=weights,
        )
        scores[~valid[y1:y2, x1:x2]] = np.inf
        scores[~np.isfinite(scores)] = np.inf
        sy, sx = np.unravel_index(scores.argmin(), scores.shape)
        if scores[sy, sx] / (known.sum() * 3) > 400:
            # A difficult edge should not prevent unrelated texture elsewhere
            # in the region from being repaired. Keep this patch for fallback.
            deferred[target_slice] = True
            continue
        sy, sx = sy + y1, sx + x1
        donor = rgb[sy-radius:sy+radius+1, sx-radius:sx+radius+1]
        selected = missing[target_slice]
        pixels[target_slice][selected] = donor[selected]
        selected[:] = False
    pixels[missing] = fallback[missing]
    return pixels, steps
