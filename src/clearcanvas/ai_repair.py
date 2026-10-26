"""Offline request planning, strict response validation and mask-only RGB composition."""

from __future__ import annotations

import base64
import binascii
import json
import math
from dataclasses import dataclass, field
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from clearcanvas.ai_service import MAX_RESPONSE_BYTES
from clearcanvas.detection import detect_watermark
from clearcanvas.documents import MAX_IMAGE_PIXELS, DocumentError, Preview, validate_path
from clearcanvas.repair import _make_result, source_stamp

CANVAS = (1536, 1024)
MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_CORRECTION_PIXELS = 12


def check_cancel(cancelled):
    if cancelled():
        raise DocumentError("已取消。", "cancelled")


def png_bytes(image):
    # New pixels, not Image.info: metadata, EXIF and original filenames never travel.
    with Image.frombytes(image.mode, image.size, image.tobytes()) as clean, BytesIO() as output:
        clean.save(output, format="PNG")
        return output.getvalue()


@dataclass
class AiPlan:
    original: Preview | None
    stamp: tuple[int, int] | None
    pixels: np.ndarray = field(repr=False)
    mask: np.ndarray = field(repr=False)
    crop: tuple[int, int, int, int]
    content: tuple[int, int, int, int]
    canvas: Image.Image = field(repr=False)
    remote_mask: Image.Image = field(repr=False)
    automatic: bool = False
    regions: list = field(default_factory=list)

    @property
    def downsampled(self):
        x1, y1, x2, y2 = self.crop
        left, top, right, bottom = self.content
        return right - left < x2 - x1 or bottom - top < y2 - y1

    def close(self):
        self.canvas.close()
        self.remote_mask.close()
        self.pixels = np.empty((0, 0, 4), dtype=np.uint8)
        self.mask = np.empty((0, 0), dtype=np.uint8)


def plan_image(image, mask, *, preview=None, stamp=None, automatic=False, regions=()):
    width, height = image.size
    if width * height > MAX_IMAGE_PIXELS or mask.shape != (height, width):
        raise DocumentError("图片或蒙版尺寸超出安全限制。", "limit")
    mask = (mask != 0).astype(np.uint8) * 255
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise DocumentError("未可靠识别到水印，请先框选；未上传图片。", "no_watermark")
    if len(xs) > min(width * height * 0.3, 2_000_000):
        raise DocumentError("选区过大，请缩小到水印附近分次处理。", "limit")
    padding = max(96, math.ceil(max(xs.max() - xs.min(), ys.max() - ys.min()) * 0.4))
    crop = (
        max(0, int(xs.min()) - padding),
        max(0, int(ys.min()) - padding),
        min(width, int(xs.max()) + padding + 1),
        min(height, int(ys.max()) + padding + 1),
    )
    x1, y1, x2, y2 = crop
    scale = min(CANVAS[0] / (x2 - x1), CANVAS[1] / (y2 - y1))
    size = (
        max(1, min(CANVAS[0], round((x2 - x1) * scale))),
        max(1, min(CANVAS[1], round((y2 - y1) * scale))),
    )
    left, top = (CANVAS[0] - size[0]) // 2, (CANVAS[1] - size[1]) // 2
    content = (left, top, left + size[0], top + size[1])
    with image.convert("RGBA") as rgba:
        pixels = np.array(rgba)
    canvas = Image.new("RGB", CANVAS, (240, 240, 240))
    with Image.fromarray(pixels[y1:y2, x1:x2]) as local:
        with local.resize(size, Image.Resampling.LANCZOS) as resized:
            canvas.paste(resized, (left, top), resized.getchannel("A"))
    # API alpha semantics: transparent repair pixels, opaque protected/padded pixels.
    remote = Image.new("RGBA", CANVAS, (255, 255, 255, 255))
    with Image.fromarray(mask[y1:y2, x1:x2]) as local_mask:
        with local_mask.resize(size, Image.Resampling.NEAREST) as scaled:
            alpha = np.array(remote.getchannel("A"))
            alpha[top : top + size[1], left : left + size[0]] = 255 - np.array(scaled)
            with Image.fromarray(alpha) as channel:
                remote.putalpha(channel)
    return AiPlan(
        preview, stamp, pixels, mask, crop, content, canvas, remote, automatic, list(regions)
    )


def prepare_plan(preview, stamp, regions, automatic, cancelled=lambda: False):
    check_cancel(cancelled)
    if preview.info.kind not in {"PNG", "JPEG", "JPG"}:
        raise DocumentError("当前版本 PDF 仅支持本地处理，不会上传。", "type")
    path = validate_path(preview.info.path)
    if source_stamp(path) != stamp:
        raise DocumentError("原文件已变化，请重新打开。", "source")
    with Image.open(path) as source:
        if (
            source.format not in {"PNG", "JPEG"}
            or getattr(source, "n_frames", 1) != 1
            or source.width * source.height > MAX_IMAGE_PIXELS
        ):
            raise DocumentError("仅支持安全尺寸内的静态 PNG/JPEG。", "type")
        with ImageOps.exif_transpose(source) as oriented, oriented.convert("RGBA") as image:
            if automatic:
                mask = detect_watermark(image, cancelled).mask
            else:
                mask = np.zeros((image.height, image.width), np.uint8)
                for left, top, right, bottom in regions:
                    if (
                        not all(
                            math.isfinite(v) and 0 <= v <= 1 for v in (left, top, right, bottom)
                        )
                        or left >= right
                        or top >= bottom
                    ):
                        raise DocumentError("选区无效。")
                    x1, y1 = (
                        math.floor(left * image.width + 1e-9),
                        math.floor(top * image.height + 1e-9),
                    )
                    x2, y2 = (
                        math.ceil(right * image.width - 1e-9),
                        math.ceil(bottom * image.height - 1e-9),
                    )
                    mask[y1:y2, x1:x2] = 255
            check_cancel(cancelled)
            plan = plan_image(
                image, mask, preview=preview, stamp=stamp, automatic=automatic, regions=regions
            )
    if source_stamp(path) != stamp:
        plan.close()
        raise DocumentError("准备期间原文件发生变化，请重新打开。", "source")
    return plan


def test_plan(cancelled=lambda: False):
    check_cancel(cancelled)
    with Image.new("RGBA", CANVAS, "#ecefe9") as image:
        draw = ImageDraw.Draw(image)
        for y in range(80, 1000, 100):
            draw.rectangle((100, y, 1400, y + 40), fill=(y % 255, 130, 100, 255))
        draw.text((710, 480), "TEST", fill="black")
        mask = np.zeros((1024, 1536), np.uint8)
        mask[470:520, 700:800] = 255
        return plan_image(image, mask)


def encode_plan(plan, send_mask, cancelled=lambda: False):
    check_cancel(cancelled)
    if plan.original and source_stamp(plan.original.info.path) != plan.stamp:
        raise DocumentError("原文件已变化，未提交请求。", "source")
    image = png_bytes(plan.canvas)
    mask = png_bytes(plan.remote_mask) if send_mask else None
    check_cancel(cancelled)
    return image, mask


def decode_response(raw, expected=CANVAS):
    if len(raw) > MAX_RESPONSE_BYTES:
        raise DocumentError("接口响应超过安全预算。", "limit")
    try:
        body = json.loads(raw)
        if not isinstance(body, dict):
            raise ValueError
        if "error" in body:
            error = body["error"]
            code = error.get("code") if isinstance(error, dict) else None
            if code in {"content_policy_violation", "moderation_blocked"}:
                raise DocumentError("服务按内容策略拒绝本次编辑；不会自动改写指令重试。", "policy")
            raise ValueError
        data = body.get("data")
        if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
            raise ValueError
        item = data[0]
        text = item.get("b64_json")
        if "url" in item or not isinstance(text, str) or not text:
            raise ValueError
        if len(text) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
            raise DocumentError("返回图片文件超过 32 MiB。", "limit")
        image_bytes = base64.b64decode(text, validate=True)
        if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
            raise DocumentError("返回图片文件超出安全预算。", "limit")
        with Image.open(BytesIO(image_bytes)) as source:
            if (
                source.format not in {"PNG", "JPEG"}
                or getattr(source, "n_frames", 1) != 1
                or source.width * source.height > MAX_IMAGE_PIXELS
            ):
                raise DocumentError("仅接受安全尺寸内的静态 PNG/JPEG。", "response")
            if source.size != expected:
                raise DocumentError("返回画布尺寸不匹配，拒绝拉伸对齐。", "alignment")
        # n_frames/getexif may load PNG data; verify needs a freshly opened stream.
        with Image.open(BytesIO(image_bytes)) as source:
            source.verify()
        with Image.open(BytesIO(image_bytes)) as source:
            if source.getexif().get(274, 1) != 1:
                raise DocumentError("返回画布方向不匹配。", "alignment")
            return source.convert("RGB")
    except DocumentError:
        raise
    except (
        ValueError,
        TypeError,
        RecursionError,
        KeyError,
        binascii.Error,
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ):
        raise DocumentError(
            "未取得唯一、可解码的 Base64 图片；不访问外链或猜测响应字段。", "response"
        ) from None


def _alignment_check(plan, returned):
    """Estimate only from protected content, never from padding or repair pixels."""
    if returned.size != plan.canvas.size:
        return "rejected", None, 0.0
    original = np.array(plan.canvas)
    result = np.array(returned)
    protected = np.array(plan.remote_mask.getchannel("A")) == 255
    content = np.zeros(protected.shape, np.uint8)
    x1, y1, x2, y2 = plan.content
    content[y1:y2, x1:x2] = 255
    protected &= content != 0
    # Transparent input is not usable correspondence evidence.
    cx1, cy1, cx2, cy2 = plan.crop
    alpha = plan.pixels[cy1:cy2, cx1:cx2, 3]
    alpha = cv2.resize(alpha, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    content[y1:y2, x1:x2] = alpha
    protected &= content >= 250
    valid = cv2.erode(protected.astype(np.uint8) * 255, np.ones((17, 17), np.uint8))
    if np.count_nonzero(valid) < 10000:
        return "uncertain", None, 0.0
    orb = cv2.ORB_create(nfeatures=1600)
    a, da = orb.detectAndCompute(cv2.cvtColor(original, cv2.COLOR_RGB2GRAY), valid)
    b, db = orb.detectAndCompute(cv2.cvtColor(result, cv2.COLOR_RGB2GRAY), valid)
    if da is None or db is None or len(a) < 20 or len(b) < 20:
        return "uncertain", None, 0.0
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    matches = [
        pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance
    ]
    if len(matches) < 20:
        return "uncertain", None, 0.0
    src = np.float32([a[m.queryIdx].pt for m in matches])
    dst = np.float32([b[m.trainIdx].pt for m in matches])
    matrix, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=2
    )
    if matrix is None or inliers is None or inliers.sum() < 16 or inliers.mean() < 0.6:
        return "uncertain", None, 0.0
    points = src[inliers.ravel() != 0]
    spread = np.prod(np.ptp(points, axis=0)) / ((x2 - x1) * (y2 - y1))
    if spread < 0.2:
        return "uncertain", None, 0.0
    corners = np.float32([[x1, y1], [x2, y1], [x1, y2], [x2, y2]])
    mapped = corners @ matrix[:, :2].T + matrix[:, 2]
    displacement = float(np.max(np.linalg.norm(mapped - corners, axis=1)))
    if displacement <= 3:
        return "aligned", matrix, displacement
    scale = math.hypot(matrix[0, 0], matrix[1, 0])
    rotation = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
    quadrants = (points[:, 0] >= (x1 + x2) / 2).astype(np.uint8)
    quadrants += 2 * (points[:, 1] >= (y1 + y2) / 2).astype(np.uint8)
    strong = (
        inliers.sum() >= 40
        and inliers.mean() >= 0.75
        and spread >= 0.35
        and np.count_nonzero(np.bincount(quadrants, minlength=4) >= 4) >= 3
    )
    if (
        strong
        and displacement <= MAX_CORRECTION_PIXELS
        and abs(scale - 1) <= 0.005
        and abs(rotation) <= 0.25
    ):
        return "correctable", matrix, displacement
    return "rejected", matrix, displacement


def alignment_status(plan, returned):
    """Conservative candidate thresholds, not facial fidelity certification."""
    return _alignment_check(plan, returned)[0]


def _correction_covers_mask(plan, matrix):
    x1, y1, x2, y2 = plan.content
    cx1, cy1, cx2, cy2 = plan.crop
    needed = np.zeros((CANVAS[1], CANVAS[0]), np.uint8)
    # Area sampling includes small original mask strokes lost by nearest-neighbor upload masks.
    needed[y1:y2, x1:x2] = cv2.resize(
        plan.mask[cy1:cy2, cx1:cx2], (x2 - x1, y2 - y1), interpolation=cv2.INTER_AREA
    )
    # Include interpolation support; never invent missing edge content by border replication.
    needed = cv2.dilate(needed, np.ones((9, 9), np.uint8))
    available = np.zeros(needed.shape, np.uint8)
    # Letterbox padding is not evidence that displaced source content survived.
    available[y1:y2, x1:x2] = 255
    coverage = cv2.warpAffine(
        available,
        matrix,
        CANVAS,
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return bool(np.all(coverage[needed != 0] == 255))


@dataclass
class AiValidated:
    image: Image.Image = field(repr=False)
    alignment: str
    raw_image: Image.Image | None = field(default=None, repr=False)
    displacement: float = 0.0

    @property
    def returned_image(self):
        return self.raw_image if self.raw_image is not None else self.image

    def close(self):
        self.image.close()
        if self.raw_image is not None and self.raw_image is not self.image:
            self.raw_image.close()


def validate_response(plan, raw, cancelled=lambda: False):
    check_cancel(cancelled)
    image = decode_response(raw)
    corrected = None
    try:
        try:
            state, matrix, displacement = _alignment_check(plan, image)
        except cv2.error:
            state, matrix, displacement = "uncertain", None, 0.0
        if state == "correctable":
            state = "rejected"
            try:
                if _correction_covers_mask(plan, matrix):
                    corrected = Image.fromarray(
                        cv2.warpAffine(
                            np.array(image),
                            matrix,
                            CANVAS,
                            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                            borderMode=cv2.BORDER_CONSTANT,
                        )
                    )
                    if alignment_status(plan, corrected) == "aligned":
                        state = "corrected"
            except cv2.error:
                pass  # Keep the paid response for review; failed correction cannot be composed.
        check_cancel(cancelled)
        if state == "corrected":
            return AiValidated(corrected, state, image, displacement)
        if corrected is not None:
            corrected.close()
        return AiValidated(image, state, displacement=displacement)
    except Exception:
        image.close()
        if corrected is not None:
            corrected.close()
        raise


def compose_result(plan, validated, config, confirmed=False, cancelled=lambda: False):
    check_cancel(cancelled)
    if validated.alignment != "aligned" and not (
        validated.alignment in {"uncertain", "corrected"} and confirmed
    ):
        raise DocumentError("需核对对齐，确认前不能合成或导出。", "alignment")
    if plan.original is None:
        raise DocumentError("接口测试图不能作为文档结果导出。")
    if source_stamp(plan.original.info.path) != plan.stamp:
        raise DocumentError("处理期间原文件发生变化，请重新打开。", "source")
    x1, y1, x2, y2 = plan.crop
    with validated.image.crop(plan.content) as content:
        with content.resize((x2 - x1, y2 - y1), Image.Resampling.LANCZOS) as restored:
            pixels = plan.pixels.copy()
            selected = plan.mask[y1:y2, x1:x2] != 0
            pixels[y1:y2, x1:x2, :3][selected] = np.array(restored)[selected]
    # Never feather outside M, never modify alpha (including inside M).
    check_cancel(cancelled)
    return _make_result(
        plan.original,
        Image.fromarray(pixels),
        plan.stamp,
        plan.regions,
        automatic=plan.automatic,
        backend="cloud_ai",
        ai_metadata={
            "model": config.model,
            "domain": config.domain,
            "canvas": CANVAS,
            "source_size": (pixels.shape[1], pixels.shape[0]),
            "downsampled": plan.downsampled,
            "alignment": validated.alignment,
            "correction_pixels": (
                validated.displacement if validated.alignment == "corrected" else 0
            ),
        },
        message="云端 AI 预览完成；请检查五官与修复区，接口可用不等于画质合格。",
    )
