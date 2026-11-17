"""Read-only, bounded RGBA differences on the processing grid, projected to preview."""

from contextlib import closing
from dataclasses import dataclass

import numpy as np
import pypdfium2 as pdfium
from PIL import Image

from clearcanvas.documents import DocumentError
from clearcanvas.repair import RepairResult, _load_image, source_stamp


@dataclass(frozen=True)
class ChangeHighlight:
    width: int
    height: int
    mask: bytes


def compute_change_highlight(result: RepairResult, cancelled) -> ChangeHighlight:
    def check():
        if cancelled():
            raise DocumentError("已取消。", "cancelled")

    def check_source():
        check()
        try:
            unchanged = source_stamp(result.original.info.path) == result.source_stamp
        except OSError:
            unchanged = False
        if not unchanged:
            raise DocumentError("原文件已变化或不可访问，请重新打开后生成高亮。", "source_changed")

    check_source()
    if result.image.mode != "RGBA":
        raise DocumentError("结果像素格式不一致，无法生成实际修改高亮。")
    original = result.original
    if result.structured:
        if (original.width, original.height) == result.image.size:
            source = np.frombuffer(original.rgba, np.uint8).reshape(
                original.height, original.width, 4
            )
            mask = _project_changes(source, result, check)
        else:
            # PDFium rounds render dimensions up. Pick a scale within both exact-size intervals.
            with pdfium.PdfDocument(original.info.path) as document:
                check()
                with closing(document[original.page_index]) as page:
                    width, height = page.get_size()
                    target_width, target_height = result.image.size
                    lower = max((target_width - 1) / width, (target_height - 1) / height)
                    upper = min(target_width / width, target_height / height)
                    if lower >= upper:
                        raise DocumentError("PDF 渲染网格不一致，无法生成实际修改高亮。")
                    check()
                    with closing(page.render(
                        scale=(lower + upper) / 2, rev_byteorder=True
                    )) as bitmap:
                        check()
                        with bitmap.to_pil() as rendered, rendered.convert("RGBA") as source:
                            mask = _project_changes(source, result, check)
    else:
        with _load_image(original) as source:
            check_source()
            mask = _project_changes(source, result, check)
    check_source()
    return ChangeHighlight(result.preview.width, result.preview.height, mask)


def _project_changes(source, result, check):
    width, height = result.image.size
    source_size = (
        source.size if isinstance(source, Image.Image) else (source.shape[1], source.shape[0])
    )
    if source_size != (width, height):
        raise DocumentError("原件与结果像素尺寸不一致，请重新处理后生成高亮。")
    preview_width, preview_height = result.preview.width, result.preview.height
    mask = np.zeros((preview_height, preview_width), np.uint8)
    # At most 256×256 pixels per temporary array; never copy the full result image.
    for top in range(0, height, 256):
        for left in range(0, width, 256):
            check()
            bottom, right = min(top + 256, height), min(left + 256, width)
            box = (left, top, right, bottom)
            with result.image.crop(box) as after:
                if isinstance(source, Image.Image):
                    with source.crop(box) as before:
                        changed = np.any(np.asarray(before) != np.asarray(after), axis=2)
                else:
                    changed = np.any(source[top:bottom, left:right] != np.asarray(after), axis=2)
            y, x = np.nonzero(changed)
            mask[(y + top) * preview_height // height, (x + left) * preview_width // width] = 255
    check()
    return mask.tobytes()
