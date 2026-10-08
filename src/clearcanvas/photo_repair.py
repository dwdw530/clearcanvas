"""Bounded local texture transfer for photographic watermark strokes."""

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg

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


def interpolate_wide_photo_strokes(rgb, mask, cancelled=lambda: False):
    """Solve a bounded harmonic fill, avoiding overshoot and patch seams.

    Every marked pixel is excluded from boundary colors. Unknowns at the canvas
    edge use only real neighbors, so padding cannot inject black into the fill.
    """
    def check(_=None):
        if cancelled():
            raise DocumentError("已取消。", "cancelled")

    check()
    missing = mask != 0
    if not np.any(missing) or np.all(missing):
        return rgb.copy()
    yy, xx = np.where(missing)
    n = len(yy)
    ids = np.full(mask.shape, -1, np.int32)
    ids[yy, xx] = np.arange(n)
    rows, columns, values = [], [], []
    rhs = np.zeros((n, 3), np.float64)
    diagonal = np.zeros(n)
    height, width = mask.shape
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        check()
        valid = (yy + dy >= 0) & (yy + dy < height) & (xx + dx >= 0) & (xx + dx < width)
        i = np.flatnonzero(valid)
        y, x = yy[valid] + dy, xx[valid] + dx
        neighbor = ids[y, x]
        diagonal[i] += 1
        unknown = neighbor >= 0
        rows.append(i[unknown])
        columns.append(neighbor[unknown])
        values.append(-np.ones(unknown.sum()))
        rhs[i[~unknown]] += rgb[y[~unknown], x[~unknown]]
    rows.append(np.arange(n))
    columns.append(np.arange(n))
    values.append(diagonal)
    matrix = coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))), shape=(n, n)
    ).tocsr()
    pixels = cv2.inpaint(rgb, mask, 5, cv2.INPAINT_NS)
    for channel in range(3):
        check()
        solved, status = cg(
            matrix, rhs[:, channel], x0=pixels[yy, xx, channel].astype(np.float64),
            rtol=1e-5, atol=0.01, maxiter=400, callback=check,
        )
        if status == 0 and np.all(np.isfinite(solved)):
            pixels[yy, xx, channel] = np.clip(np.rint(solved), 0, 255).astype(np.uint8)
    check()
    pixels[~missing] = rgb[~missing]
    return pixels
