"""Read-only document rendering. PDFium calls must stay on one worker."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_raw
from PIL import Image, ImageOps, UnidentifiedImageError

SUPPORTED_SUFFIXES = frozenset({".pdf", ".png", ".jpg", ".jpeg"})
MAX_FILE_BYTES = 1024**3
MAX_IMAGE_PIXELS = 40_000_000
MAX_PREVIEW_EDGE = 3200


class DocumentError(Exception):
    def __init__(self, message: str, code: str = "read") -> None:
        super().__init__(message)
        self.code = code


class PreviewCancelled(Exception):
    pass


@dataclass(frozen=True)
class DocumentInfo:
    path: Path
    kind: str
    page_count: int
    byte_size: int


@dataclass(frozen=True)
class Preview:
    info: DocumentInfo
    page_index: int
    width: int
    height: int
    rgba: bytes
    source_width: float
    source_height: float
    unit: str

    @property
    def logical_size(self) -> tuple[float, float]:
        factor = 96 / 72 if self.unit == "pt" else 1
        return self.source_width * factor, self.source_height * factor


def validate_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise DocumentError("仅支持 PDF、JPG 和 PNG 文件。", "type")
    try:
        path = path.resolve(strict=True)
        if not path.is_file():
            raise DocumentError("请选择文件，不能导入文件夹。", "type")
        size = path.stat().st_size
    except (OSError, RuntimeError) as exc:
        raise DocumentError("文件不存在或无法访问。", "access") from exc
    if size == 0:
        raise DocumentError("文件是空的，无法预览。", "empty")
    if size > MAX_FILE_BYTES:
        raise DocumentError("单个文件不能超过 1 GiB。", "limit")
    return path


def render_document(
    value: str | Path,
    page_index: int = 0,
    password: str | None = None,
    cancelled: Callable[[], bool] = lambda: False,
) -> Preview:
    def check() -> None:
        if cancelled():
            raise PreviewCancelled

    check()
    path = validate_path(value)
    if page_index < 0:
        raise DocumentError("页码超出文档范围。", "page")
    try:
        if path.suffix.lower() == ".pdf":
            return _render_pdf(path, page_index, password, check)
        if page_index != 0:
            raise DocumentError("图片只有一页。", "page")
        return _render_image(path, check)
    except (DocumentError, PreviewCancelled):
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise DocumentError("图片尺寸过大，无法安全预览。", "limit") from exc
    except (UnidentifiedImageError, OSError, ValueError, pdfium.PdfiumError) as exc:
        raise DocumentError("无法读取文件，请检查文件是否完整或仍可访问。") from exc


def _render_image(path: Path, check: Callable[[], None]) -> Preview:
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(path) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise DocumentError("文件实际格式不是 JPG 或 PNG。", "type")
            if getattr(image, "n_frames", 1) > 1:
                raise DocumentError("暂不支持动画 PNG。", "type")
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise DocumentError("图片超过 4000 万像素，无法安全预览。", "limit")
            source_width, source_height = image.size
            if image.getexif().get(274) in {5, 6, 7, 8}:
                source_width, source_height = source_height, source_width
            check()
            image.draft("RGB", (MAX_PREVIEW_EDGE, MAX_PREVIEW_EDGE))
            oriented = ImageOps.exif_transpose(image)
            try:
                oriented.thumbnail((MAX_PREVIEW_EDGE, MAX_PREVIEW_EDGE), Image.Resampling.LANCZOS)
                check()
                with oriented.convert("RGBA") as rgba:
                    return Preview(
                        DocumentInfo(path, image.format, 1, path.stat().st_size),
                        0,
                        rgba.width,
                        rgba.height,
                        rgba.tobytes(),
                        source_width,
                        source_height,
                        "px",
                    )
            finally:
                oriented.close()


def _render_pdf(
    path: Path, page_index: int, password: str | None, check: Callable[[], None]
) -> Preview:
    try:
        document = pdfium.PdfDocument(path, password=password)
    except pdfium.PdfiumError as exc:
        if exc.err_code == pdfium_raw.FPDF_ERR_PASSWORD:
            raise DocumentError("此 PDF 需要密码，或输入的密码不正确。", "password") from exc
        raise
    with document:
        count = len(document)
        if not 0 <= page_index < count:
            raise DocumentError("页码超出文档范围。", "page")
        check()
        document.init_forms()
        with closing(document[page_index]) as page:
            width, height = page.get_size()
            if not all(math.isfinite(n) and 0 < n <= 200_000 for n in (width, height)):
                raise DocumentError("PDF 页面尺寸异常，无法安全预览。", "limit")
            scale = min(2.5, MAX_PREVIEW_EDGE / max(width, height))
            with closing(page.render(scale=scale, rev_byteorder=True)) as bitmap:
                check()
                with bitmap.to_pil() as image, image.convert("RGBA") as rgba:
                    return Preview(
                        DocumentInfo(path, "PDF", count, path.stat().st_size),
                        page_index,
                        rgba.width,
                        rgba.height,
                        rgba.tobytes(),
                        width,
                        height,
                        "pt",
                    )
