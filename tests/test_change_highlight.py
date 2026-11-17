from dataclasses import replace

import numpy as np
import pytest
from PIL import Image, ImageOps
from test_automatic import marked_pdf

from clearcanvas.change_highlight import compute_change_highlight
from clearcanvas.documents import DocumentError, render_document
from clearcanvas.repair import RepairResult, _load_image, auto_repair_document, source_stamp


@pytest.fixture
def pixel_result(tmp_path):
    path = tmp_path / "差异 原件.png"
    pixels = np.random.default_rng(17).integers(0, 256, (389, 513, 4), np.uint8)
    with Image.fromarray(pixels) as image:
        image.save(path)
    original = render_document(path)
    result = RepairResult(original, original, Image.fromarray(pixels), source_stamp(path), [])
    yield result
    result.image.close()


@pytest.mark.parametrize("size", [(513, 389), (37, 29), (800, 600)])
def test_exact_rgba_changes_project_without_filling_selection(pixel_result, size):
    result = pixel_result
    result.regions = [(0, 0, 1, 1)]
    changes = [(0, 0), (512, 388), (255, 256), (300, 19)]
    for x, y in changes:
        rgba = list(result.image.getpixel((x, y)))
        rgba[3 if x == 300 else 0] ^= 255
        result.image.putpixel((x, y), tuple(rgba))
    result.preview = replace(result.original, width=size[0], height=size[1])
    highlight = compute_change_highlight(result, lambda: False)
    expected = np.zeros((size[1], size[0]), np.uint8)
    for x, y in changes:
        expected[y * size[1] // 389, x * size[0] // 513] = 255
    assert highlight.mask == expected.tobytes()
    assert (highlight.width, highlight.height) == size


def test_zero_changes_do_not_treat_preview_resampling_as_changes(pixel_result):
    result = pixel_result
    result.preview = replace(result.original, width=31, height=27, rgba=b"not-a-diff-input")
    highlight = compute_change_highlight(result, lambda: False)
    assert highlight.mask == bytes(31 * 27)


def test_exif_jpeg_uses_oriented_full_resolution_grid(tmp_path):
    path = tmp_path / "方向 原件.jpg"
    pixels = np.random.default_rng(71).integers(0, 256, (25, 41, 3), np.uint8)
    with Image.fromarray(pixels) as image:
        exif = image.getexif()
        exif[274] = 6
        image.save(path, exif=exif)
    preview = render_document(path)
    with Image.open(path) as image, ImageOps.exif_transpose(image) as oriented:
        result = RepairResult(preview, preview, oriented.convert("RGBA"), source_stamp(path), [])
    try:
        assert result.image.size == (25, 41)
        assert not any(compute_change_highlight(result, lambda: False).mask)
        result.image.putpixel((24, 40), (0, 0, 0, 0))
        expected = bytearray(25 * 41)
        expected[-1] = 255
        assert compute_change_highlight(result, lambda: False).mask == bytes(expected)
    finally:
        result.image.close()


def test_source_and_grid_mismatch_rejected_without_altering_result(pixel_result):
    result = pixel_result
    original_bytes = result.image.tobytes()
    with Image.new("RGBA", (10, 10)) as wrong:
        mismatch = replace(result, image=wrong)
        with pytest.raises(DocumentError, match="尺寸不一致"):
            compute_change_highlight(mismatch, lambda: False)
    result.original.info.path.write_bytes(b"external change")
    with pytest.raises(DocumentError) as error:
        compute_change_highlight(result, lambda: False)
    assert error.value.code == "source_changed"
    assert result.image.tobytes() == original_bytes


def test_source_change_during_loading_rejected(pixel_result, monkeypatch):
    from clearcanvas import change_highlight

    def load(preview):
        image = _load_image(preview)
        preview.info.path.write_bytes(b"changed while decoding")
        return image

    monkeypatch.setattr(change_highlight, "_load_image", load)
    with pytest.raises(DocumentError) as error:
        compute_change_highlight(pixel_result, lambda: False)
    assert error.value.code == "source_changed"


@pytest.mark.parametrize("stop", [1, 5])
def test_cancellation_before_loading_and_between_tiles_preserves_result(pixel_result, stop):
    calls = 0

    def cancelled():
        nonlocal calls
        calls += 1
        return calls >= stop

    original = pixel_result.image.tobytes()
    with pytest.raises(DocumentError) as error:
        compute_change_highlight(pixel_result, cancelled)
    assert error.value.code == "cancelled"
    assert pixel_result.image.tobytes() == original


def test_structured_pdf_same_grid_and_rerendered_grid_match(tmp_path):
    path = marked_pdf(tmp_path / "独立对象.pdf")
    result = auto_repair_document(render_document(path), source_stamp(path))
    try:
        first = compute_change_highlight(result, lambda: False)
        before = np.frombuffer(result.original.rgba, np.uint8).reshape(
            result.original.height, result.original.width, 4
        )
        expected = np.any(before != np.asarray(result.image), axis=2).astype(np.uint8) * 255
        assert first.mask == expected.tobytes()
        assert np.count_nonzero(expected) > 0
        # A different original preview cannot be scaled into the truth grid.
        resized = replace(result.original, width=1, height=1, rgba=b"\xff" * 4)
        assert compute_change_highlight(replace(result, original=resized), lambda: False) == first
    finally:
        result.image.close()


def test_raster_pdf_compares_processing_pixels_before_preview_projection(sample_pdf):
    from clearcanvas.repair import repair_document

    preview = render_document(sample_pdf)
    result = repair_document(preview, [(0.06, 0.078, 0.41, 0.128)], source_stamp(sample_pdf))
    try:
        with _load_image(preview) as original:
            changed = np.any(np.asarray(original) != np.asarray(result.image), axis=2)
        y, x = np.nonzero(changed)
        expected = np.zeros((result.preview.height, result.preview.width), np.uint8)
        expected[
            y * result.preview.height // result.image.height,
            x * result.preview.width // result.image.width,
        ] = 255
        assert np.count_nonzero(expected) > 0
        assert compute_change_highlight(result, lambda: False).mask == expected.tobytes()
    finally:
        result.image.close()
