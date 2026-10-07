"""Pixel masks for high-contrast lettering over photographs, without model weights."""

import cv2
import numpy as np
from PIL import Image

from clearcanvas.documents import DocumentError


def _same_lettering(first, second, labels, stats):
    """Similar line positions and sizes alone are not evidence of repetition."""
    stencils = []
    clipped = []
    for ids in (first, second):
        boxes = stats[ids]
        x1, y1 = boxes[:, :2].min(axis=0)
        x2, y2 = (boxes[:, :2] + boxes[:, 2:4]).max(axis=0)
        stencils.append(np.isin(labels[y1:y2, x1:x2], ids).astype(np.uint8))
        clipped.append(x1 == 0 or y1 == 0 or x2 == labels.shape[1] or y2 == labels.shape[0])
    a, b = stencils
    # A word clipped by the canvas can be much smaller than its complete copy.
    # Keep shape correlation mandatory; only the size gate changes at an edge.
    lower, upper = (0.35, 1 / 0.35) if any(clipped) else (0.65, 1.5)
    if not all(lower <= u / v <= upper for u, v in zip(a.shape, b.shape, strict=True)):
        return False
    # Small line-fit differences can trim a stroke. Compare with a short margin
    # rather than requiring identical bounding boxes or an exact bitmap match.
    if a.size > b.size:
        a, b = b, a
    margin = max(2, int(min(a.shape[0], b.shape[0]) * 0.15))
    target = np.pad(b, ((margin, margin), (margin, margin)))
    if any(u > v for u, v in zip(a.shape, target.shape, strict=True)):
        return False
    score = cv2.matchTemplate(target, a, cv2.TM_CCOEFF_NORMED).max()
    return score >= 0.65


def _refine_neutral_corner_line(rgb, gray, opaque, contrast, labels, stats, ids, height, angle):
    """Complete a verified pale text line using its own color, not a global cutoff."""
    if abs(angle) > 12:
        return None
    tall = ids[stats[ids, 3] >= height * 0.7]
    if len(tall) < 4:
        return None
    core = np.isin(labels, tall)
    histogram = np.bincount(gray[core], minlength=256)
    mode = int(np.convolve(histogram, np.ones(9), mode="same").argmax())
    same = np.abs(gray.astype(np.int16) - mode) <= 18
    if mode < 150 or np.mean(same[core]) < 0.55:
        return None
    color = np.median(rgb[core & same], axis=0)
    # Colored/translucent lettering varies with its background. A single color
    # model is unsafe there; keep the existing multi-contrast detector instead.
    if np.ptp(color) > 24:
        return None
    stable = [i for i in tall if np.mean(same[labels == i]) >= 0.4]
    if len(stable) < 4:
        return None
    boxes = stats[tall]
    text_height = int(np.median(boxes[:, 3]))
    left, top = boxes[:, :2].min(axis=0)
    right, bottom = (boxes[:, :2] + boxes[:, 2:4]).max(axis=0)
    # Allow one disconnected character before the strong strokes. Do not scan
    # the entire bottom strip for similarly colored scenery or captions.
    x1 = max(0, int(left - text_height * 1.6))
    x2 = min(gray.shape[1], int(right + text_height * 0.6))
    y1 = max(0, int(top - text_height * 0.06))
    y2 = min(gray.shape[0], int(bottom + text_height * 0.15))
    candidate = (
        (np.max(np.abs(rgb[y1:y2, x1:x2].astype(np.int16) - color), axis=2) <= 24)
        & (contrast[y1:y2, x1:x2] > 12)
        & opaque[y1:y2, x1:x2]
    )
    count, local_labels, local_stats, _ = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8), 8
    )
    chosen = np.zeros(count, np.uint8)
    for i in range(1, count):
        _, _, w, h, area = local_stats[i]
        if (
            area >= 3 and max(w, h) <= text_height * 1.8
            and area / max(w, h) < text_height * 0.4
        ):
            chosen[i] = 255
    result = np.zeros(gray.shape, np.uint8)
    result[y1:y2, x1:x2] = chosen[local_labels]
    # Antialiasing and transparency can darken parts of a stroke on tree shadows.
    # Keep those connected strokes only when the color model supplies support.
    touched = np.bincount(labels[result != 0], minlength=len(stats))
    for i in ids:
        _, _, w, h, area = stats[i]
        if (
            h >= text_height * 0.45 and touched[i] >= max(3, area * 0.08)
            and area / max(w, h) < text_height * 0.4
        ):
            result[labels == i] = 255
    if np.mean(result[core & same] != 0) < 0.8:
        return None
    return result


def _repeat_stencil(candidate, stencil, check):
    ys, xs = np.where(stencil)
    template = stencil[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].astype(np.float32)
    h, w = template.shape
    if min(h, w) < 5 or h * w > candidate.size * 0.25:
        return stencil
    padding = ((h - 1, h - 1), (w - 1, w - 1))
    target = np.pad(candidate.astype(np.float32), padding)
    visible = np.pad(np.ones(candidate.shape, np.float32), padding)

    def window_sum(values):
        integral = cv2.integral(values, sdepth=cv2.CV_64F)
        return integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]

    # Normalized binary correlation over the visible part of each window also
    # recognizes copies clipped by the image edge, without treating padding as ink.
    check()
    n = window_sum(visible)
    observed = window_sum(target)
    expected = cv2.matchTemplate(visible, template, cv2.TM_CCORR)
    overlap = cv2.matchTemplate(target, template, cv2.TM_CCORR)
    denominator = np.sqrt(np.maximum(expected * (n - expected) * observed * (n - observed), 0))
    scores = np.divide(
        n * overlap - expected * observed,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 1,
    )
    scores[(expected < template.sum() * 0.35) | (n < template.size * 0.30)] = 0
    output = np.pad(stencil, padding)
    for _ in range(48):
        check()
        _, score, _, (x, y) = cv2.minMaxLoc(scores)
        if score < 0.82:
            break
        output[y : y + h, x : x + w] |= template.astype(np.uint8)
        scores[max(0, y - h // 2) : y + h // 2 + 1, max(0, x - w // 2) : x + w // 2 + 1] = 0
    return output[h - 1 : h - 1 + candidate.shape[0], w - 1 : w - 1 + candidate.shape[1]]


def detect_photo_watermark_mask(image: Image.Image, cancelled=lambda: False) -> np.ndarray:
    def check():
        if cancelled():
            raise DocumentError("已取消。", "cancelled")

    check()
    with image.convert("RGBA") as small:
        small.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        pixels = np.array(small)
    rgb = pixels[:, :, :3]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    short = min(gray.shape)
    spread = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2)
    opaque = pixels[:, :, 3] >= 250
    result = np.zeros(gray.shape, dtype=np.uint8)
    refined_any = False
    light_lines = []
    for light, blended in ((True, False), (True, True), (False, False), (False, True)):
        check()
        kernel = np.ones((max(9, int(short * 0.025)) | 1,) * 2, np.uint8)
        contrast = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT if light else cv2.MORPH_BLACKHAT, kernel)
        if light and blended:
            candidate = (spread <= 110) & opaque & (contrast > 45) & (gray >= 135)
        elif light:
            candidate = (spread <= 12) & opaque & (gray >= 248)
        elif blended:
            candidate = (spread <= 50) & opaque & (contrast > 20) & (gray <= 90)
        else:
            candidate = (spread <= 25) & opaque & (gray <= 55)
        candidate = candidate.astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
        valid = np.zeros(count, dtype=np.uint8)
        for label in range(1, count):
            _, _, w, h, area = stats[label]
            if (
                max(5, short * 0.008) <= h <= short * 0.16
                and 2 <= w <= short * 0.16
                and 0.08 <= w / h <= 3
                and 6 <= area
                and 0.10 <= area / (w * h) <= 0.90
            ):
                valid[label] = 1
        if valid.sum() < 7:
            continue
        # Group similarly sized strokes at several scales; avoid joining scenery
        # into a single giant component when one font is smaller than another.
        groups = []
        for height in (short * 0.018, short * 0.035, short * 0.065, short * 0.10):
            check()
            eligible = valid.copy()
            eligible[(stats[:, 3] < height * 0.50) | (stats[:, 3] > height * 1.9)] = 0
            strokes = eligible[labels]
            size = max(3, int(height * 1.2)) | 1
            joined = cv2.dilate(strokes, np.ones((size, size), np.uint8))
            n, joined_labels, boxes, _ = cv2.connectedComponentsWithStats(joined, 8)
            for group in range(1, n):
                x, y, w, h, _ = boxes[group]
                region = joined_labels[y : y + h, x : x + w] == group
                ids = np.unique(
                    labels[y : y + h, x : x + w][region & (strokes[y : y + h, x : x + w] != 0)]
                )
                if len(ids) < 5:
                    continue
                heights = stats[ids, 3]
                median = float(np.median(heights))
                if np.mean((heights >= median * 0.55) & (heights <= median * 1.7)) < 0.85:
                    continue
                centers = (stats[ids, :2] + stats[ids, 2:4] / 2).astype(np.float32)
                vx, vy, cx, cy = cv2.fitLine(centers, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
                distances = np.abs((centers[:, 0] - cx) * vy - (centers[:, 1] - cy) * vx)
                ids = ids[distances <= median * 0.65]
                if len(ids) < 5:
                    continue
                yy, xx = np.where(np.isin(labels[y : y + h, x : x + w], ids))
                _, (rw, rh), angle = cv2.minAreaRect(np.column_stack((xx, yy)).astype(np.float32))
                if rw < rh:
                    angle += 90
                angle = (angle + 90) % 180 - 90
                length, thickness = max(rw, rh), min(rw, rh)
                if abs(angle) > 45 or length < median * 4 or thickness > median * 1.8:
                    continue
                if length / max(thickness, 1) < 4 or length < short * 0.13:
                    continue
                # The joined box includes dilation and strokes discarded by the
                # line fit. Classify position using only the surviving lettering.
                groups.append(
                    (ids, x + int(xx.min()), y + int(yy.min()),
                     int(np.ptp(xx)) + 1, int(np.ptp(yy)) + 1, median, angle)
                )
        accepted = set()
        for ids, _x, y, _w, _h, height, angle in groups:
            # A single top line is often a poster heading, not a watermark.
            near_edge = y > gray.shape[0] * 0.82
            # Disjoint words on the same heading are not repeated watermarks.
            repeats = {
                round(gy / max(height * 2, 1))
                for other, _, gy, _, _, gh, ga in groups
                if 0.65 <= gh / height <= 1.5
                and abs(ga - angle) < 8
                and abs(gy - y) > height * 2
                and not np.intersect1d(ids, other).size
                and _same_lettering(ids, other, labels, stats)
            }
            centers = stats[:, :2] + stats[:, 2:4] / 2
            direction = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
            along = centers @ direction
            # Scattered texture can fit a baseline without forming a continuous
            # caption. Repeated watermarks may legitimately span separate words.
            continuous = np.max(np.diff(np.sort(along[ids]))) <= height * 3.5
            if not ((near_edge and len(ids) >= 7 and continuous) or len(repeats) >= 1):
                continue
            if not light and any(
                max(0, min(_x + _w, lx + lw) - max(_x, lx))
                * max(0, min(y + _h, ly + lh) - max(y, ly))
                > min(_w * _h, lw * lh) * 0.5
                for lx, ly, lw, lh in light_lines
            ):
                # Dark gaps bounded by pale lettering resemble a second text
                # line. They are background, not another watermark to erase.
                continue
            if light:
                light_lines.append((_x, y, _w, _h))
            across = centers @ np.array([-direction[1], direction[0]])
            baseline = np.median(across[ids])
            line = (
                (valid != 0)
                & (stats[:, 3] >= height * 0.45)
                & (stats[:, 3] <= height * 1.8)
                & (np.abs(across - baseline) <= height * 0.6)
            )
            expanded = ids
            for _ in range(3):
                expanded = np.flatnonzero(
                    line
                    & (along >= along[expanded].min() - height * 3)
                    & (along <= along[expanded].max() + height * 3)
                )
                if not expanded.size:
                    expanded = ids
                    break
            if near_edge and light and blended:
                refined = _refine_neutral_corner_line(
                    rgb, gray, opaque, contrast, labels, stats, expanded, height, angle
                )
                check()
                if refined is not None:
                    result |= refined
                    refined_any = True
                    continue
            accepted.update(int(i) for i in expanded)
            if near_edge:
                # Disconnected radicals and short crossbars belong to the same
                # lettering, but are too small to establish a text line alone.
                core = np.isin(labels, expanded).astype(np.uint8)
                yy, xx = np.where(core)
                padding = max(2, int(height * 0.40))
                core_color = np.median(rgb[core != 0], axis=0)
                nearby = cv2.dilate(core, np.ones((padding * 2 + 1,) * 2, np.uint8))
                fragments = np.unique(labels[(nearby != 0) & (candidate != 0)])
                for fragment in fragments:
                    fx, fy, fw, fh, area = stats[fragment]
                    if (
                        area >= 3
                        and fw <= height * 1.8
                        and fh <= height * 1.8
                        and fx >= xx.min() - padding
                        and fx + fw <= xx.max() + padding + 1
                        and fy >= yy.min() - padding
                        and fy + fh <= yy.max() + padding + 1
                        and np.max(np.abs(
                            np.median(rgb[labels == fragment], axis=0) - core_color
                        )) <= 30
                    ):
                        accepted.add(int(fragment))
        selected = np.zeros(count, dtype=np.uint8)
        selected[list(accepted)] = 255
        result |= selected[labels]
        if light and not blended and accepted and groups:
            ids = max(groups, key=lambda group: len(group[0]))[0]
            if len(ids) >= 7 and all(int(label) in accepted for label in ids):
                repeated = _repeat_stencil(candidate, np.isin(labels, ids).astype(np.uint8), check)
                result |= repeated * 255
    check()
    if not np.any(result):
        raise DocumentError("未可靠识别到照片水印，请框选需要处理的区域。", "no_watermark")
    result = cv2.resize(result, image.size, interpolation=cv2.INTER_NEAREST)
    radius = max(2, int(np.ceil(max(image.size) / max(gray.shape) * 1.5)))
    if refined_any:
        # Color completion includes lower-contrast strokes whose antialiased
        # fringe extends beyond the downsampled threshold boundary.
        radius += 1
    result = cv2.dilate(result, np.ones((radius * 2 + 1,) * 2, np.uint8))
    with image.convert("RGBA") as full:
        result[np.asarray(full)[:, :, 3] < 250] = 0
    check()
    if not np.any(result) or np.count_nonzero(result) > min(result.size * 0.15, 2_000_000):
        raise DocumentError("照片水印识别范围不可靠，请框选分次处理。", "no_watermark")
    return result
