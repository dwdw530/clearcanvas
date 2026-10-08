from __future__ import annotations

import math
import os
import re
import tempfile
import zlib
from contextlib import closing
from dataclasses import dataclass, field, replace
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pypdfium2 as pdfium
from PIL import Image, ImageOps
from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    EncodedStreamObject,
    NameObject,
    NumberObject,
    RectangleObject,
)
from skimage.restoration import inpaint_biharmonic

from clearcanvas.detection import detect_watermark
from clearcanvas.documents import (
    MAX_IMAGE_PIXELS,
    MAX_PREVIEW_EDGE,
    DocumentError,
    Preview,
    render_document,
)
from clearcanvas.pdf_watermarks import remove_pdf_watermarks
from clearcanvas.photo_repair import interpolate_wide_photo_strokes, restore_photo_texture
from clearcanvas.texture_inpaint import restore_wide_texture


@dataclass
class RepairResult:
    original: Preview
    preview: Preview
    image: Image.Image
    source_stamp: tuple[int, int]
    regions: list[tuple[float, float, float, float]]
    automatic: bool = False
    structured: bool = False
    message: str = "修复完成"
    skipped_pages: list[int] = field(default_factory=list)


@dataclass
class BatchResult:
    outputs: list[Path]
    errors: list[str]
    cancelled: bool
    notes: list[str] = field(default_factory=list)


def parse_pages(text: str, count: int) -> list[int]:
    pages = set()
    for part in text.replace("，", ",").split(","):
        match = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+)\s*)?", part)
        if not match:
            raise DocumentError("页码格式不正确，例如 1,3-5。")
        start, end = int(match[1]), int(match[2] or match[1])
        if not 1 <= start <= end <= count:
            raise DocumentError("页码超出范围或区间顺序错误。")
        pages.update(range(start - 1, end))
    return sorted(pages)


def source_stamp(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _check_pdf(reader: PdfReader) -> None:
    if reader.is_encrypted:
        raise DocumentError("当前修复暂不支持加密 PDF。")
    if reader.trailer["/Root"].get("/AcroForm") or reader.trailer["/Root"].get("/Perms"):
        raise DocumentError("带表单或数字签名的 PDF 暂不支持修复。")


def _load_image(preview: Preview) -> Image.Image:
    path = preview.info.path
    if preview.info.kind == "PDF":
        with path.open("rb") as stream:
            _check_pdf(PdfReader(stream))
        with pdfium.PdfDocument(path) as document:
            with closing(document[preview.page_index]) as page:
                width, height = page.get_size()
                scale = min(200 / 72, math.sqrt(MAX_IMAGE_PIXELS / (width * height)))
                with closing(page.render(scale=scale, rev_byteorder=True)) as bitmap:
                    with bitmap.to_pil() as source:
                        image = source.convert("RGBA")
    else:
        with Image.open(path) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise DocumentError("图片超过 4000 万像素。")
            with ImageOps.exif_transpose(source) as oriented:
                image = oriented.convert("RGBA")
    return image


def _make_result(preview, image, stamp, regions, **options):
    if source_stamp(preview.info.path) != stamp:
        image.close()
        raise DocumentError("处理期间原文件发生变化，请重新打开。")
    with image.copy() as small:
        small.thumbnail((MAX_PREVIEW_EDGE, MAX_PREVIEW_EDGE), Image.Resampling.LANCZOS)
        rendered = replace(preview, width=small.width, height=small.height, rgba=small.tobytes())
    return RepairResult(preview, rendered, image, stamp, list(regions), **options)


def _repair_image(image, mask, cancelled=lambda: False, *, photographic=False):
    if cancelled():
        raise DocumentError("已取消。", "cancelled")
    pixels = np.array(image)
    height, width = mask.shape
    area = np.count_nonzero(mask)
    if not area:
        raise DocumentError("没有可处理的水印像素。", "no_watermark")
    if area > min(width * height * 0.3, 2_000_000):
        raise DocumentError("选区过大，请缩小到水印附近分次处理。")
    # Inpaint separate local regions, avoiding a page-sized solve for sparse marks.
    joined = cv2.dilate(mask, np.ones((9, 9), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    texture_budget = 6000
    for label in range(1, count):
        if cancelled():
            raise DocumentError("已取消。", "cancelled")
        x, y, w, h, _ = stats[label]
        padding = 70 if photographic else 24
        x1, x2 = max(0, x - padding), min(width, x + w + padding)
        y1, y2 = max(0, y - padding), min(height, y + h + padding)
        local_mask = mask[y1:y2, x1:x2].copy()
        local_mask[labels[y1:y2, x1:x2] != label] = 0
        rgb = np.ascontiguousarray(pixels[y1:y2, x1:x2, :3])
        if photographic:
            excluded = mask[y1:y2, x1:x2].copy()
            excluded[pixels[y1:y2, x1:x2, 3] < 250] = 255
            # Keep texture transfer for ordinary narrow lettering. Broad holes
            # (over 36 pixels across) need a stable tone before any texture claim.
            wide = cv2.distanceTransform(local_mask, cv2.DIST_L2, 5).max() > 18
            if wide:
                repaired = interpolate_wide_photo_strokes(rgb, excluded, cancelled)
                repaired = restore_wide_texture(rgb, local_mask, excluded, repaired, cancelled)
            else:
                repaired = inpaint_biharmonic(
                    rgb, excluded != 0, split_into_regions=True, channel_axis=-1
                )
                repaired = np.clip(np.rint(repaired * 255), 0, 255).astype(np.uint8)
                repaired, steps = restore_photo_texture(
                    rgb, local_mask, excluded, repaired, cancelled, texture_budget
                )
                texture_budget -= steps
        else:
            repaired = cv2.inpaint(rgb, local_mask, 3, cv2.INPAINT_TELEA)
        selected = local_mask != 0
        pixels[y1:y2, x1:x2, :3][selected] = repaired[selected]
        alpha = cv2.inpaint(
            np.ascontiguousarray(pixels[y1:y2, x1:x2, 3]), local_mask, 3, cv2.INPAINT_TELEA
        )
        pixels[y1:y2, x1:x2, 3][selected] = alpha[selected]
    if cancelled():
        raise DocumentError("已取消。", "cancelled")
    return Image.fromarray(pixels)


def _repair_document_background(image, mask, cancelled=lambda: False):
    if cancelled():
        raise DocumentError("已取消。", "cancelled")
    pixels = np.array(image)
    # Estimate the paper separately so nearby text cannot bleed into wide strokes.
    # Detection has already restricted this path to pale marks on light documents.
    size = max(9, min(201, int(min(image.size) * 0.12))) | 1
    background = cv2.morphologyEx(
        pixels[:, :, :3], cv2.MORPH_CLOSE, np.ones((size, size), np.uint8)
    )
    if cancelled():
        raise DocumentError("已取消。", "cancelled")
    selected = mask != 0
    pixels[:, :, :3][selected] = background[selected]
    return Image.fromarray(pixels)


def repair_document(
    preview: Preview,
    regions: list[tuple[float, float, float, float]],
    stamp: tuple[int, int],
) -> RepairResult:
    if source_stamp(preview.info.path) != stamp:
        raise DocumentError("原文件已变化，请重新打开后处理。")
    if not regions:
        raise DocumentError("尚未选择水印区域。")
    image = _load_image(preview)
    try:
        width, height = image.size
        mask = np.zeros((height, width), dtype=np.uint8)
        for left, top, right, bottom in regions:
            if not all(math.isfinite(n) and 0 <= n <= 1 for n in (left, top, right, bottom)):
                raise DocumentError("选区超出图片范围。")
            if left >= right or top >= bottom:
                raise DocumentError("选区无效。")
            x1, y1 = math.floor(left * width + 1e-9), math.floor(top * height + 1e-9)
            x2, y2 = math.ceil(right * width - 1e-9), math.ceil(bottom * height - 1e-9)
            mask[y1:y2, x1:x2] = 255
        return _make_result(preview, _repair_image(image, mask), stamp, regions)
    finally:
        image.close()


def auto_repair_document(preview: Preview, stamp, cancelled=lambda: False) -> RepairResult:
    if cancelled():
        raise DocumentError("已取消。", "cancelled")
    if source_stamp(preview.info.path) != stamp:
        raise DocumentError("原文件已变化，请重新打开后处理。")
    if preview.info.kind == "PDF":
        with preview.info.path.open("rb") as stream:
            reader = PdfReader(stream)
            _check_pdf(reader)
            writer = PdfWriter()
            page = writer.add_page(reader.pages[preview.page_index])
            count = remove_pdf_watermarks(page, cancelled)
            if count:
                with BytesIO() as buffer:
                    writer.write(buffer)
                    with pdfium.PdfDocument(buffer.getvalue()) as document:
                        with closing(document[0]) as rendered_page:
                            w, h = rendered_page.get_size()
                            scale = min(2.5, MAX_PREVIEW_EDGE / max(w, h))
                            with closing(
                                rendered_page.render(scale=scale, rev_byteorder=True)
                            ) as bitmap:
                                with bitmap.to_pil() as rendered:
                                    image = rendered.convert("RGBA")
                if cancelled():
                    image.close()
                    raise DocumentError("已取消。", "cancelled")
                return _make_result(
                    preview,
                    image,
                    stamp,
                    [],
                    automatic=True,
                    structured=True,
                    message=f"已移除 {count} 个独立水印，正文与文字层保留",
                )
    with _load_image(preview) as image:
        detection = detect_watermark(image, cancelled)
        if detection.photographic:
            result = _repair_image(image, detection.mask, cancelled, photographic=True)
        else:
            result = _repair_document_background(image, detection.mask, cancelled)
    return _make_result(
        preview,
        result,
        stamp,
        [],
        automatic=True,
        message="自动修复预览完成"
        + ("；当前页导出将转为图像" if preview.info.kind == "PDF" else ""),
    )


def export_result(
    result: RepairResult,
    destination: Path,
    overwrite: bool = False,
    pages: list[int] | None = None,
    cancelled=lambda: False,
    progress=lambda done, total: None,
) -> Path:
    if cancelled():
        raise DocumentError("已取消。", "cancelled")
    destination = destination.resolve()
    original = result.original.info.path
    if destination == original or (
        destination.exists() and os.path.samefile(destination, original)
    ):
        raise DocumentError("不能覆盖原文件，请使用另一个文件名。")
    allowed = {".pdf"} if result.original.info.kind == "PDF" else {".png", ".jpg", ".jpeg"}
    if destination.suffix.lower() not in allowed:
        raise DocumentError("导出格式与文档类型不匹配。")
    if destination.exists() and not overwrite:
        raise DocumentError("目标文件已存在。")
    handle, name = tempfile.mkstemp(prefix=".clearcanvas-", dir=destination.parent)
    os.close(handle)
    temporary = Path(name)
    try:
        if result.original.info.kind == "PDF":
            _export_pdf(result, temporary, pages, cancelled, progress)
        elif destination.suffix.lower() == ".png":
            result.image.save(temporary, format="PNG")
        else:
            with Image.new("RGB", result.image.size, "white") as rgb:
                rgb.paste(result.image, mask=result.image.getchannel("A"))
                rgb.save(temporary, format="JPEG", quality=95, subsampling=0)
        if cancelled():
            raise DocumentError("已取消。", "cancelled")
        if overwrite:
            os.replace(temporary, destination)
        elif os.name == "nt":
            os.rename(temporary, destination)
        else:
            os.link(temporary, destination)
        return destination
    finally:
        temporary.unlink(missing_ok=True)


def _export_pdf(result, destination, pages, cancelled, progress) -> None:
    original = result.original
    if source_stamp(original.info.path) != result.source_stamp:
        raise DocumentError("原 PDF 已变化，已取消导出。")
    with original.info.path.open("rb") as stream:
        reader = PdfReader(stream)
        _check_pdf(reader)
        writer = PdfWriter()
        writer.clone_document_from_reader(reader)
        indices = sorted(set(pages if pages is not None else [original.page_index]))
        if not indices or any(not 0 <= index < len(writer.pages) for index in indices):
            raise DocumentError("导出页码无效。")
        if result.automatic:
            _export_auto_pages(writer, result, indices, cancelled, progress)
            if source_stamp(original.info.path) != result.source_stamp:
                raise DocumentError("原 PDF 已变化，已取消导出。")
            writer.write(destination)
            progress(len(indices), len(indices))
            return
        with pdfium.PdfDocument(original.info.path) as document:
            for index in indices:
                with closing(document[index]) as page:
                    w, h = page.get_size()
                    if (
                        abs(w - original.source_width) > 0.01
                        or abs(h - original.source_height) > 0.01
                    ):
                        raise DocumentError(f"第 {index + 1} 页尺寸或方向不同，请单独框选处理。")
        for done, index in enumerate(indices):
            if cancelled():
                raise DocumentError("已取消。", "cancelled")
            progress(done, len(indices))
            current = (
                result
                if index == original.page_index
                else repair_document(
                    replace(original, page_index=index),
                    result.regions,
                    result.source_stamp,
                )
            )
            try:
                _replace_page_image(writer, index, current)
            finally:
                if current is not result:
                    current.image.close()
        if source_stamp(original.info.path) != result.source_stamp:
            raise DocumentError("原 PDF 已变化，已取消导出。")
        writer.compress_identical_objects(remove_duplicates=False, remove_unreferenced=True)
        writer.write(destination)
        progress(len(indices), len(indices))


def _export_auto_pages(writer, result, indices, cancelled, progress):
    result.skipped_pages.clear()
    changed = 0
    for done, index in enumerate(indices):
        if cancelled():
            raise DocumentError("已取消。", "cancelled")
        progress(done, len(indices))
        if remove_pdf_watermarks(writer.pages[index], cancelled):
            changed += 1
            continue
        current = None
        try:
            if index == result.original.page_index and not result.structured:
                current = result
            else:
                preview = render_document(result.original.info.path, index)
                current = auto_repair_document(preview, result.source_stamp, cancelled)
            _replace_page_image(writer, index, current)
            changed += 1
        except DocumentError as error:
            if error.code != "no_watermark":
                raise
            result.skipped_pages.append(index + 1)
        finally:
            if current is not None and current is not result:
                current.image.close()
    if not changed:
        raise DocumentError("所选页面均未可靠识别到水印，未生成导出文件。", "no_watermark")


def _replace_page_image(writer, index, result):
    page = writer.pages[index]
    original = result.original
    with result.image.convert("RGB") as image:
        raster = EncodedStreamObject()
        raster.update(
            {
                NameObject("/Type"): NameObject("/XObject"),
                NameObject("/Subtype"): NameObject("/Image"),
                NameObject("/Width"): NumberObject(image.width),
                NameObject("/Height"): NumberObject(image.height),
                NameObject("/ColorSpace"): NameObject("/DeviceRGB"),
                NameObject("/BitsPerComponent"): NumberObject(8),
                NameObject("/Filter"): NameObject("/FlateDecode"),
            }
        )
        raster._data = zlib.compress(image.tobytes())
    resources = DictionaryObject(
        {
            NameObject("/XObject"): DictionaryObject(
                {NameObject("/CCImage"): writer._add_object(raster)}
            )
        }
    )
    content = DecodedStreamObject()
    w, h = original.source_width, original.source_height
    content.set_data(f"q {w:.6f} 0 0 {h:.6f} 0 0 cm /CCImage Do Q".encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(content)
    page[NameObject("/Resources")] = resources
    page[NameObject("/MediaBox")] = RectangleObject([0, 0, w, h])
    page[NameObject("/CropBox")] = RectangleObject([0, 0, w, h])
    page[NameObject("/Rotate")] = NumberObject(0)
    for key in ("/Annots", "/TrimBox", "/BleedBox", "/ArtBox"):
        page.pop(key, None)


def batch_export(reference, paths, directory, page_text, cancelled, progress):
    outputs, errors, notes = [], [], []
    for index, path in enumerate(paths):
        if cancelled():
            break
        progress(index, len(paths))
        result = None
        try:
            stamp = source_stamp(path)
            preview = render_document(path)
            pages = None
            if preview.info.kind == "PDF":
                pages = (
                    list(range(preview.info.page_count))
                    if page_text == "all"
                    else parse_pages(page_text, preview.info.page_count)
                )
                if pages[0] != preview.page_index:
                    preview = render_document(path, pages[0])
            if reference.automatic:
                for page_index in pages if pages is not None else [0]:
                    if cancelled():
                        raise DocumentError("已取消。", "cancelled")
                    if page_index != preview.page_index:
                        preview = render_document(path, page_index)
                    try:
                        result = auto_repair_document(preview, stamp, cancelled)
                        break
                    except DocumentError as error:
                        if error.code != "no_watermark":
                            raise
                if result is None:
                    raise DocumentError("未可靠识别到水印，已跳过。", "no_watermark")
            else:
                if (preview.info.kind == "PDF") != (
                    reference.original.info.kind == "PDF"
                ) or preview.logical_size != reference.original.logical_size:
                    raise DocumentError("类型或尺寸与选区模板不同，已跳过。")
                result = repair_document(preview, reference.regions, stamp)
            suffix = ".pdf" if preview.info.kind == "PDF" else ".png"
            destination = directory / f"{path.stem}_净界{suffix}"
            number = 1
            while destination.exists():
                destination = directory / f"{path.stem}_净界_{number}{suffix}"
                number += 1
            export_result(result, destination, pages=pages, cancelled=cancelled)
            outputs.append(destination)
            if result.skipped_pages:
                summary = ", ".join(map(str, result.skipped_pages[:20]))
                if len(result.skipped_pages) > 20:
                    summary += " ..."
                notes.append(f"{path.name}: 第 {summary} 页未可靠识别，保留原页。")
        except Exception as error:
            if isinstance(error, DocumentError) and error.code == "cancelled":
                break
            errors.append(f"{path.name}: {error}")
        finally:
            if result is not None:
                result.image.close()
    progress(len(paths), len(paths))
    return BatchResult(outputs, errors, cancelled(), notes)
