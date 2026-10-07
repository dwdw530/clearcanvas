import hashlib

import pytest
from PIL import Image
from pypdf import PdfReader, PdfWriter

from clearcanvas.documents import (
    MAX_PREVIEW_EDGE,
    DocumentError,
    PreviewCancelled,
    render_document,
    validate_path,
)


def test_image_preview_is_read_only(sample_image):
    before = hashlib.sha256(sample_image.read_bytes()).hexdigest()
    preview = render_document(sample_image)
    assert (preview.width, preview.height) == (960, 640)
    assert preview.info.kind == "PNG"
    assert preview.logical_size == (960, 640)
    assert len(preview.rgba) == 960 * 640 * 4
    assert hashlib.sha256(sample_image.read_bytes()).hexdigest() == before


def test_pdf_preview_pages_and_original_preserved(sample_pdf):
    before = sample_pdf.read_bytes()
    first = render_document(sample_pdf)
    second = render_document(sample_pdf, 1)
    assert first.info.page_count == 3
    assert (first.source_width, first.source_height) == (595, 842)
    assert second.source_width > second.source_height
    assert first.rgba != second.rgba
    pixels = Image.frombytes("RGBA", (first.width, first.height), first.rgba)
    assert pixels.convert("L").getextrema()[0] < 150
    assert sample_pdf.read_bytes() == before


def test_pdf_rotation_and_crop(sample_pdf, tmp_path):
    reader = PdfReader(sample_pdf)
    page = reader.pages[0]
    page.cropbox.upper_right = (400, 600)
    page.rotate(90)
    writer = PdfWriter()
    writer.add_page(page)
    path = tmp_path / "旋转.pdf"
    writer.write(path)
    preview = render_document(path)
    assert (preview.source_width, preview.source_height) == (600, 400)


def test_encrypted_pdf(encrypted_pdf):
    for password in (None, "wrong"):
        with pytest.raises(DocumentError) as error:
            render_document(encrypted_pdf, password=password)
        assert error.value.code == "password"
    preview = render_document(encrypted_pdf, password="clearcanvas-test")
    assert preview.info.page_count == 3


def test_exif_orientation(tmp_path):
    image = Image.new("RGB", (600, 400), "red")
    exif = image.getexif()
    exif[274] = 6
    path = tmp_path / "手机照片.JPG"
    image.save(path, exif=exif)
    preview = render_document(path)
    assert (preview.width, preview.height) == (400, 600)
    assert preview.logical_size == (400, 600)


def test_transparency(tmp_path):
    path = tmp_path / "透明.png"
    Image.new("RGBA", (32, 32), (30, 80, 70, 100)).save(path)
    preview = render_document(path)
    assert preview.rgba[3] == 100


def test_large_image_downsample_keeps_logical_dimensions(tmp_path):
    path = tmp_path / "全景.jpg"
    Image.new("RGB", (6800, 800), "white").save(path)
    preview = render_document(path)
    assert preview.logical_size == (6800, 800)
    assert max(preview.width, preview.height) <= MAX_PREVIEW_EDGE


@pytest.mark.parametrize("page_index", [-1, 3, 100])
def test_out_of_range_pdf(sample_pdf, page_index):
    with pytest.raises(DocumentError) as error:
        render_document(sample_pdf, page_index)
    assert error.value.code == "page"


def test_out_of_range_image(sample_image):
    with pytest.raises(DocumentError, match="只有一页"):
        render_document(sample_image, 1)


@pytest.mark.parametrize("suffix", [".png", ".pdf", ".jpg"])
def test_corrupt_files(tmp_path, suffix):
    path = tmp_path / f"broken{suffix}"
    path.write_bytes(b"This is not an image or PDF")
    with pytest.raises(DocumentError):
        render_document(path)


def test_type_and_access_validation(tmp_path):
    with pytest.raises(DocumentError) as missing:
        validate_path(tmp_path / "missing.png")
    assert missing.value.code == "access"
    path = tmp_path / "wrong.gif"
    Image.new("RGB", (20, 20)).save(path)
    with pytest.raises(DocumentError) as kind:
        validate_path(path)
    assert kind.value.code == "type"
    disguised = tmp_path / "fake.png"
    disguised.write_bytes(path.read_bytes())
    with pytest.raises(DocumentError) as disguised_error:
        render_document(disguised)
    assert disguised_error.value.code == "type"


def test_empty_file(tmp_path):
    path = tmp_path / "empty.pdf"
    path.touch()
    with pytest.raises(DocumentError) as error:
        validate_path(path)
    assert error.value.code == "empty"


def test_pixel_limit(sample_image, monkeypatch):
    monkeypatch.setattr("clearcanvas.documents.MAX_IMAGE_PIXELS", 100)
    with pytest.raises(DocumentError) as error:
        render_document(sample_image)
    assert error.value.code == "limit"


def test_file_size_limit(sample_image, monkeypatch):
    monkeypatch.setattr("clearcanvas.documents.MAX_FILE_BYTES", 10)
    with pytest.raises(DocumentError) as error:
        validate_path(sample_image)
    assert error.value.code == "limit"


def test_cancel_before_open(sample_image):
    with pytest.raises(PreviewCancelled):
        render_document(sample_image, cancelled=lambda: True)
