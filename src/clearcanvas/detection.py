"""Conservative, offline detection of pale diagonal text on light documents."""

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from clearcanvas.documents import DocumentError
from clearcanvas.photo_watermarks import detect_photo_watermark_mask


@dataclass
class WatermarkDetection:
    mask: np.ndarray
    photographic: bool = False


def _overprinted_graphics(rgb, gray, opaque, check):
    """Find pale graphics interrupted by several rows of dark document text."""
    short = min(gray.shape)
    spread = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2)
    candidate = (
        (gray >= 185) & (gray <= 250) & (spread >= 8) & (spread <= 70) & opaque
    ).astype(np.uint8)
    dark = (rgb.min(axis=2) < 145).astype(np.uint8)
    candidate[cv2.dilate(dark, np.ones((3, 3), np.uint8)) != 0] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    cores = cv2.erode(candidate, np.ones((2, 2), np.uint8))
    support = np.bincount(labels[cores != 0], minlength=count)
    valid = np.zeros(count, dtype=np.uint8)
    for label in range(1, count):
        _, _, w, h, area = stats[label]
        if area >= 12 and support[label] >= 4 and min(w, h) >= 3:
            valid[label] = 1
    strokes = valid[labels]
    size = max(5, int(short * 0.035)) | 1
    bridged = cv2.morphologyEx(strokes, cv2.MORPH_CLOSE, np.ones((size, size), np.uint8))
    groups, group_labels, group_stats, _ = cv2.connectedComponentsWithStats(bridged, 8)
    accepted = np.zeros(groups, dtype=np.uint8)
    for group in range(1, groups):
        check()
        x, y, w, h, area = group_stats[group]
        if min(w, h) < short * 0.18 or max(w, h) < short * 0.30:
            continue
        local = group_labels[y:y + h, x:x + w] == group
        crossed = local & (dark[y:y + h, x:x + w] != 0)
        rows = np.count_nonzero(crossed, axis=1) >= max(3, w * 0.025)
        starts = np.flatnonzero(rows & ~np.r_[False, rows[:-1]])
        if len(starts) < 3 or np.ptp(starts) < short * 0.15:
            continue
        # Filled panels and page backgrounds are not overprinted graphics.
        coverage = np.count_nonzero(strokes[y:y + h, x:x + w] & local) / max(1, w * h)
        if 0.015 <= coverage <= 0.40 and area < gray.size * 0.65:
            accepted[group] = 1
    return ((accepted[group_labels] != 0) & (strokes != 0)).astype(np.uint8) * 255


def _tiled_graphics(rgb, gray, opaque, check):
    """Require repeated colored shapes in both axes and text crossing their strokes."""
    short = min(gray.shape)
    spread = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2)
    dark = (rgb.min(axis=2) < 145).astype(np.uint8)
    protected = cv2.dilate(dark, np.ones((3, 3), np.uint8))
    candidate = (
        (gray >= 185) & (gray <= 250) & (spread >= 8) & (spread <= 90)
        & opaque & (protected == 0)
    ).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    templates = sorted(
        (i for i in range(1, count)
         if 40 <= stats[i, 4] and short * 0.025 <= stats[i, 2] <= short * 0.25
         and 6 <= stats[i, 3] <= short * 0.25),
        key=lambda i: -stats[i, 4],
    )[:24]
    result = np.zeros(gray.shape, dtype=np.uint8)
    tested = set()
    accepted_colors = []
    rgb_int = rgb.astype(np.int16)
    for label in templates:
        check()
        x, y, w, h, _ = stats[label]
        color = np.median(rgb[y:y + h, x:x + w][labels[y:y + h, x:x + w] == label], axis=0)
        if any(np.max(np.abs(color - previous)) <= 4 for previous in accepted_colors):
            continue
        key = (*np.round(color / 16).astype(int), int(w // 4), int(h // 4))
        if key in tested:
            continue
        tested.add(key)
        template = np.pad((labels[y:y + h, x:x + w] == label).astype(np.uint8), 2)
        scores = cv2.matchTemplate(candidate, template, cv2.TM_CCOEFF_NORMED)
        matches = []
        for _ in range(64):
            check()
            _, score, _, (mx, my) = cv2.minMaxLoc(scores)
            if score < 0.80:
                break
            matches.append((mx, my))
            scores[max(0, my - h):my + h + 1, max(0, mx - w):mx + w + 1] = -1
        if len(matches) < 4:
            continue
        points = np.array(matches)
        if np.ptp(points[:, 0]) < short * 0.35 or np.ptp(points[:, 1]) < short * 0.35:
            continue
        same_color = (
            (np.max(np.abs(rgb_int - color), axis=2) <= 22)
            & (candidate != 0)
        ).astype(np.uint8)
        bridge_size = max(5, int(short * 0.045)) | 1
        bridged = cv2.morphologyEx(
            same_color, cv2.MORPH_CLOSE, np.ones((bridge_size, bridge_size), np.uint8)
        )
        rows = np.count_nonzero((bridged != 0) & (dark != 0), axis=1) >= 6
        starts = np.flatnonzero(rows & ~np.r_[False, rows[:-1]])
        if len(starts) >= 3 and np.ptp(starts) >= short * 0.15:
            result |= same_color * 255
            accepted_colors.append(color)
    return cv2.dilate(result, np.ones((3, 3), np.uint8))


def detect_watermark(image: Image.Image, cancelled=lambda: False) -> WatermarkDetection:
    def check():
        if cancelled():
            raise DocumentError("已取消。", "cancelled")

    check()
    with image.convert("RGBA") as small:
        small.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        rgba = np.array(small)
    rgb = rgba[:, :, :3]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    neutral = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2) <= 24
    opaque = rgba[:, :, 3] >= 250
    if np.mean((gray >= 238) & neutral & opaque) < 0.60:
        return WatermarkDetection(detect_photo_watermark_mask(image, cancelled), photographic=True)

    # Screenshots can contain application chrome; keep it outside the paper stencil.
    _, _, paper_stats, _ = cv2.connectedComponentsWithStats(
        ((gray >= 250) & neutral & opaque).astype(np.uint8), 8
    )
    paper = np.ones(gray.shape, dtype=np.uint8)
    if len(paper_stats) > 1:
        x, y, w, h, area = max(paper_stats[1:], key=lambda item: item[4])
        if area >= gray.size * 0.35:
            paper[:] = 0
            paper[y:y + h, x:x + w] = 1
    opaque = opaque & (paper != 0)

    candidate = ((gray >= 155) & (gray <= 250) & neutral & opaque).astype(np.uint8)
    # Keep dark body text and its antialiased edges out of the repair mask.
    protected = cv2.dilate((rgb.min(axis=2) < 145).astype(np.uint8), np.ones((3, 3), np.uint8))
    candidate[protected != 0] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    # JPEG ringing around body text is not a watermark stroke.
    cores = cv2.erode(candidate, np.ones((3, 3), np.uint8))
    core_counts = np.bincount(labels[cores != 0], minlength=count)
    short = min(gray.shape)
    valid = np.zeros(count, dtype=np.uint8)
    heights = []
    for label in range(1, count):
        _, _, w, h, area = stats[label]
        if (
            6 <= area <= short * short * 0.08
            and 3 <= min(w, h)
            and max(w, h) < short * 0.65
            and core_counts[label] >= 3
        ):
            valid[label] = 1
            heights.append(min(w, h))
    check()
    glyphs = valid[labels]
    join = max(5, min(65, int(np.percentile(heights, 75) * 1.2))) if heights else 5
    joined = cv2.dilate(glyphs, np.ones((join, join), np.uint8))
    groups, group_labels, group_stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    accepted = np.zeros(groups, dtype=np.uint8)
    for group in range(1, groups):
        check()
        x, y, w, h, _ = group_stats[group]
        local = (group_labels[y : y + h, x : x + w] == group) & (glyphs[y : y + h, x : x + w] != 0)
        ids = np.unique(labels[y : y + h, x : x + w][local])
        if len(ids) < 3:
            continue
        ys, xs = np.where(local)
        points = np.column_stack((xs, ys)).astype(np.float32)
        _, (rw, rh), angle = cv2.minAreaRect(points)
        if rw < rh:
            angle += 90
        angle = abs((angle + 90) % 180 - 90)
        length, thickness = max(rw, rh), min(rw, rh)
        if (
            12 <= angle <= 78
            and length >= short * 0.20
            and 5 <= thickness <= short * 0.48
            and length / thickness >= 2.2
            and 0.06 <= len(xs) / max(1, length * thickness) <= 0.65
        ):
            accepted[group] = 1
    mask = ((accepted[group_labels] != 0) & (glyphs != 0)).astype(np.uint8) * 255
    graphics = _overprinted_graphics(rgb, gray, opaque, check)
    mask = cv2.bitwise_or(mask, graphics)
    mask = cv2.bitwise_or(mask, _tiled_graphics(rgb, gray, opaque, check))
    if not np.any(mask):
        raise DocumentError("未可靠识别到水印，请框选需要处理的区域。", "no_watermark")
    # Upscale the stencil, then recheck full-resolution pixels to protect text.
    mask = cv2.resize(mask, image.size, interpolation=cv2.INTER_NEAREST)
    radius = max(1, int(max(image.size) / max(gray.shape)))
    mask = cv2.dilate(mask, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
    mask[cv2.resize(paper, image.size, interpolation=cv2.INTER_NEAREST) == 0] = 0
    check()
    with image.convert("RGBA") as full:
        pixels = np.array(full)
    full_gray = cv2.cvtColor(pixels[:, :, :3], cv2.COLOR_RGB2GRAY)
    dark = cv2.dilate(
        (pixels[:, :, :3].min(axis=2) < 145).astype(np.uint8), np.ones((3, 3), np.uint8)
    )
    neutral = pixels[:, :, :3].max(axis=2).astype(np.int16) - pixels[:, :, :3].min(axis=2) <= 90
    mask[(dark != 0) | (full_gray < 150) | ~neutral | (pixels[:, :, 3] < 250)] = 0
    if not np.any(mask) or np.count_nonzero(mask) > min(mask.size * 0.15, 2_000_000):
        raise DocumentError("自动识别范围不可靠，请框选分次处理。", "no_watermark")
    return WatermarkDetection(mask)


def detect_watermark_mask(image: Image.Image, cancelled=lambda: False) -> np.ndarray:
    return detect_watermark(image, cancelled).mask
