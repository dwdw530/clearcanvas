"""Adapt upstream PatchMatch and OpenCV seamless cloning to masked photo repair."""

import cv2
import numpy as np
import patchmatch

from clearcanvas.documents import DocumentError
from clearcanvas.structure_inpaint import restore_photo_structure


def restore_wide_texture(rgb, mask, excluded, fallback, cancelled=lambda: False):
    """Restore local texture using existing non-neural algorithms.

    A bounded crop keeps each native call finite. Other watermarks and transparent
    areas are excluded from the source corpus, even when they are outside this hole.
    """
    def check():
        if cancelled():
            raise DocumentError("已取消。", "cancelled")

    check()
    selected = mask != 0
    if not np.any(selected) or mask.size > 1_500_000 or np.all(excluded):
        return fallback
    if not patchmatch.patchmatch_available:
        raise DocumentError("纹理修复组件无法加载，请重新安装完整版本。", "repair_unavailable")

    # Flat regions do not benefit from copying texture and must retain their
    # smooth tone. This also avoids injecting unrelated noise into a flat sky.
    safe = cv2.dilate(excluded, np.ones((5, 5), np.uint8)) == 0
    ring = (cv2.dilate(mask, np.ones((21, 21), np.uint8)) != 0) & safe
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    detail = gray - cv2.GaussianBlur(gray, (5, 5), 0)
    if np.count_nonzero(ring) < 49:
        return fallback
    if np.std(detail[ring]) < 1.0:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0) / 8
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1) / 8
        # A clean edge is structure, not texture. Continue it without copying
        # patches; a uniform gradient still keeps the harmonic interpolation.
        if np.count_nonzero(ring & (np.hypot(gx, gy) > 3)) >= 8:
            structure = restore_photo_structure(rgb, excluded, cancelled)
            structure[~selected] = fallback[~selected]
            return structure
        return fallback

    other = ((excluded != 0) & ~selected).astype(np.uint8)
    patchmatch.set_random_seed(42)
    try:
        candidate = patchmatch.inpaint(rgb, mask, global_mask=other, patch_size=3)
    except (RuntimeError, ValueError) as error:
        raise DocumentError("纹理修复失败，请缩小处理区域后重试。") from error
    check()
    candidate[other != 0] = fallback[other != 0]

    # Poisson blending removes tonal seams. The destination must already be
    # clean: the clone solver's boundary band must never contain watermark ink.
    pad = 16
    source = cv2.copyMakeBorder(candidate, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    target = cv2.copyMakeBorder(fallback, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    clone_mask = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    x, y, width, height = cv2.boundingRect(clone_mask)
    repaired = cv2.seamlessClone(
        source, target, clone_mask.copy(), (x + width // 2, y + height // 2), cv2.NORMAL_CLONE
    )[pad:-pad, pad:-pad].copy()
    # A word may cross both textured hair and smooth skin. Do not transfer the
    # same amount of texture over the whole word: extend the observed local
    # high-frequency energy into the hole and keep smooth portions interpolated.
    clean_gray = cv2.cvtColor(fallback, cv2.COLOR_RGB2GRAY).astype(np.float32)
    fine = clean_gray - cv2.GaussianBlur(clean_gray, (5, 5), 0)
    energy = np.sqrt(cv2.GaussianBlur(fine * fine, (21, 21), 0))
    energy = cv2.inpaint(energy, excluded, 3, cv2.INPAINT_NS)
    weight = np.clip((energy - 1.0) / 0.8, 0, 1)[:, :, None]
    structure = fallback
    if np.count_nonzero(selected & (weight[:, :, 0] < 0.95)) >= 49:
        structure = restore_photo_structure(rgb, excluded, cancelled)
    repaired = np.clip(np.rint(
        repaired.astype(np.float32) * weight + structure * (1 - weight)
    ), 0, 255).astype(np.uint8)
    # seamlessClone mutates its mask and changes pixels beyond an irregular hole.
    # Keep the original selection immutable and explicitly composite the result.
    repaired[~selected] = fallback[~selected]
    check()
    return repaired
