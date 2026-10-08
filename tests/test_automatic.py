import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    ArrayObject,
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
    NumberObject,
    RectangleObject,
)
from PySide6.QtWidgets import QMessageBox

from clearcanvas.detection import detect_watermark_mask
from clearcanvas.documents import DocumentError, render_document
from clearcanvas.pdf_watermarks import remove_pdf_watermarks
from clearcanvas.repair import auto_repair_document, batch_export, export_result, source_stamp


@pytest.fixture
def pale_image(tmp_path):
    image = Image.new("RGBA", (1000, 800), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 65), "Report - body text must remain unchanged", fill="black", font_size=22)
    draw.text((70, 110), "Gray caption - not a watermark", fill=(185, 185, 185), font_size=18)
    layer = Image.new("RGBA", image.size)
    ImageDraw.Draw(layer).text((220, 340), "SAMPLE WATERMARK", fill=(205, 205, 205), font_size=48)
    rotated = layer.rotate(30, resample=Image.Resampling.BICUBIC)
    image.alpha_composite(rotated)
    path = tmp_path / "automatic.png"
    image.save(path)
    return path


def marked_pdf(path, form=False, named=False, outer=False):
    writer = PdfWriter()
    font = writer._add_object(
        DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
    )
    properties = DictionaryObject({NameObject("/Subtype"): NameObject("/Watermark")})
    resources = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
            NameObject("/Properties"): DictionaryObject({NameObject("/WM"): properties}),
        }
    )
    tag = b"/WM" if named else b"<</Subtype /Watermark>>"
    contents = (
        b"q 0 g BT /F1 18 Tf 40 700 Td (BODY TEXT) Tj ET Q "
        b"/Artifact " + tag + b" BDC q .8 g BT /F1 40 Tf "
        b".866 .5 -.5 .866 150 250 Tm (WATERMARK) Tj ET Q EMC "
        b"q 0 g BT /F1 12 Tf 40 650 Td (AFTER MARK) Tj ET Q"
    )
    if outer:
        contents = contents.replace(b"/Artifact ", b"q /Artifact ")
        contents = contents.replace(b"BDC q ", b"BDC ").replace(b"Q EMC", b"EMC Q")
    stream = DecodedStreamObject()
    stream.set_data(contents)
    if form:
        stream.update(
            {
                NameObject("/Type"): NameObject("/XObject"),
                NameObject("/Subtype"): NameObject("/Form"),
                NameObject("/BBox"): RectangleObject([0, 0, 600, 800]),
                NameObject("/Resources"): resources,
            }
        )
        reference = writer._add_object(stream)
        resources = DictionaryObject(
            {
                NameObject("/XObject"): DictionaryObject(
                    {
                        NameObject("/Shared"): reference,
                    }
                )
            }
        )
        stream = DecodedStreamObject()
        stream.set_data(b"q /Shared Do Q")
    reference = writer._add_object(stream)
    for width, height in ((600, 800), (800, 600), (600, 800)):
        page = writer.add_blank_page(width, height)
        page[NameObject("/Resources")] = resources
        page[NameObject("/Contents")] = reference
    writer.add_outline_item("Kept bookmark", 0)
    writer.add_metadata({"/Title": "Original metadata"})
    writer.write(path)
    return path


@pytest.mark.parametrize("kind", ["PNG", "JPEG", "PDF"])
def test_auto_image_and_scan(pale_image, tmp_path, kind):
    source = tmp_path / ("input." + {"PNG": "png", "JPEG": "jpg", "PDF": "pdf"}[kind])
    with Image.open(pale_image).convert("RGB") as image:
        image.save(source, format=kind)
    before_hash = hashlib.sha256(source.read_bytes()).digest()
    result = auto_repair_document(render_document(source), source_stamp(source))
    assert result.automatic and not result.structured
    assert not result.regions
    output = export_result(result, tmp_path / ("output.pdf" if kind == "PDF" else "output.png"))
    assert output.exists()
    assert result.preview.rgba != result.original.rgba
    assert hashlib.sha256(source.read_bytes()).digest() == before_hash
    pixels = np.array(result.image)
    if kind == "PNG":
        original = np.array(Image.open(source).convert("RGBA"))
        assert np.array_equal(original[:160], pixels[:160])
        mark = (original[:, :, 0] > 170) & (original[:, :, 0] < 230)
        mark[:160] = False
        assert np.mean(pixels[:, :, 0][mark]) > 248
        assert np.mean(pixels[:, :, 0][mark] > 245) > 0.99


def test_mask_protects_dark_text_and_outside(pale_image):
    with Image.open(pale_image) as image:
        original = np.array(image)
        # Add body text through the watermark, not only elsewhere on the page.
        ImageDraw.Draw(image).text((260, 350), "BODY TEXT", fill="black", font_size=24)
        image.save(pale_image)
        original = np.array(image)
        mask = detect_watermark_mask(image)
    result = auto_repair_document(render_document(pale_image), source_stamp(pale_image))
    pixels = np.array(result.image)
    assert np.array_equal(original[mask == 0], pixels[mask == 0])
    assert np.array_equal(original[original[:, :, 0] < 145], pixels[original[:, :, 0] < 145])


@pytest.mark.parametrize("color", ["white", "#547a83"])
def test_no_watermark_and_gray_caption_rejected(tmp_path, color):
    image = Image.new("RGB", (1000, 800), color)
    ImageDraw.Draw(image).text((100, 350), "Ordinary gray heading", fill="#bbbbbb", font_size=48)
    path = tmp_path / "clean.png"
    image.save(path)
    with pytest.raises(DocumentError) as error:
        auto_repair_document(render_document(path), source_stamp(path))
    assert error.value.code == "no_watermark"


@pytest.mark.parametrize("form,named", [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize("outer", [False, True])
def test_structured_pdf_preserves_body_shared_resources_and_other_pages(
    tmp_path, form, named, outer,
):
    path = marked_pdf(tmp_path / "marked.pdf", form, named, outer)
    original_bytes = path.read_bytes()
    before = render_document(path, 1)
    result = auto_repair_document(render_document(path), source_stamp(path))
    assert result.automatic and result.structured
    output = export_result(result, tmp_path / "result.pdf")
    reader = PdfReader(output)
    assert "WATERMARK" not in reader.pages[0].extract_text()
    assert "BODY TEXT" in reader.pages[0].extract_text()
    assert "AFTER MARK" in reader.pages[0].extract_text()
    assert "WATERMARK" in reader.pages[1].extract_text()
    assert render_document(output, 1).rgba == before.rgba
    assert reader.metadata.title == "Original metadata"
    assert reader.outline[0].title == "Kept bookmark"
    assert path.read_bytes() == original_bytes
    output_all = export_result(result, tmp_path / "all.pdf", pages=[0, 1, 2])
    assert all("WATERMARK" not in p.extract_text() for p in PdfReader(output_all).pages)
    assert tuple(PdfReader(output_all).pages[1].mediabox) == (0, 0, 800, 600)


def test_annotations_links_and_unmarked_artifacts(tmp_path):
    path = marked_pdf(tmp_path / "marked.pdf")
    writer = PdfWriter(clone_from=path)
    page = writer.pages[0]
    # Watermark annotations are distinct from links and ordinary artifacts.
    page[NameObject("/Annots")] = ArrayObject(
        [
            writer._add_object(
                DictionaryObject(
                    {
                        NameObject("/Type"): NameObject("/Annot"),
                        NameObject("/Subtype"): NameObject(kind),
                        NameObject("/Rect"): RectangleObject([0, 0, 10, 10]),
                        NameObject("/F"): NumberObject(4),
                    }
                )
            )
            for kind in ("/Watermark", "/Link")
        ]
    )
    assert remove_pdf_watermarks(page) == 2
    assert len(page["/Annots"]) == 1
    assert page["/Annots"][0].get_object()["/Subtype"] == "/Link"
    content = DecodedStreamObject()
    content.set_data(b"/Artifact BMC q BT /F1 20 Tf (Header) Tj ET Q EMC")
    page.replace_contents(content)
    assert remove_pdf_watermarks(page) == 0
    assert "Header" in page.extract_text()


def test_unisolated_mark_does_not_change_graphics_state(tmp_path):
    path = marked_pdf(tmp_path / "marked.pdf")
    writer = PdfWriter(clone_from=path)
    page = writer.pages[0]
    content = DecodedStreamObject()
    content.set_data(b"/Artifact <</Subtype /Watermark>> BDC .5 g EMC BT /F1 12 Tf (Keep) Tj ET")
    page.replace_contents(content)
    assert remove_pdf_watermarks(page) == 0
    assert b".5 g" in page.get_contents().get_data()


def test_auto_cancel_and_source_change(pale_image, tmp_path):
    stamp = source_stamp(pale_image)
    preview = render_document(pale_image)
    with pytest.raises(DocumentError, match="取消"):
        auto_repair_document(preview, stamp, lambda: True)
    pale_image.write_bytes(pale_image.read_bytes() + b"changed")
    with pytest.raises(DocumentError, match="变化"):
        auto_repair_document(preview, stamp)


def test_auto_export_skips_clean_pages_and_no_empty_success(tmp_path):
    source = marked_pdf(tmp_path / "marked.pdf")
    writer = PdfWriter(clone_from=source)
    writer.pages[1].pop("/Contents", None)
    writer.write(source)
    result = auto_repair_document(render_document(source), source_stamp(source))
    before = render_document(source, 1)
    target = export_result(result, tmp_path / "out.pdf", pages=[0, 1])
    assert result.skipped_pages == [2]
    assert render_document(target, 1).rgba == before.rgba
    with pytest.raises(DocumentError, match="未生成"):
        export_result(result, tmp_path / "clean-only.pdf", pages=[1])
    assert not (tmp_path / "clean-only.pdf").exists()
    assert not list(tmp_path.glob(".clearcanvas-*"))


def test_auto_batch_mixed_types_and_blank_first_page(pale_image, tmp_path):
    source = marked_pdf(tmp_path / "marked.pdf")
    writer = PdfWriter(clone_from=source)
    writer.pages[0].pop("/Contents", None)
    writer.write(source)
    reference = auto_repair_document(render_document(pale_image), source_stamp(pale_image))
    out = tmp_path / "batch"
    out.mkdir()
    result = batch_export(
        reference, [source, pale_image], out, "all", lambda: False, lambda *a: None
    )
    assert not result.errors
    assert len(result.notes) == 1
    assert len(result.outputs) == 2
    pdf = PdfReader(next(p for p in result.outputs if p.suffix == ".pdf"))
    assert "WATERMARK" not in pdf.pages[1].extract_text()
    assert "BODY TEXT" in pdf.pages[1].extract_text()


def test_one_click_ui_without_selection(qtbot, window, pale_image, tmp_path, monkeypatch):
    assert not window.editor.auto_button.isEnabled()
    window.add_files([pale_image])
    qtbot.waitUntil(lambda: window._preview is not None)
    assert not window.canvas.regions
    assert window.editor.auto_button.isEnabled()
    assert not window.editor.repair_button.isEnabled()
    window.editor.auto_button.click()
    assert not window.editor.auto_button.isEnabled()
    qtbot.waitUntil(lambda: window.editor.result is not None, timeout=15000)
    assert window.editor.result.automatic
    assert window.badge.text() == "结果"
    artifacts = Path(__file__).resolve().parents[1] / "artifacts" / "qa"
    artifacts.mkdir(parents=True, exist_ok=True)
    window.resize(1000, 680)
    assert window.grab().save(str(artifacts / "auto-image-result.png"))
    output = tmp_path / "ui.png"
    monkeypatch.setattr(
        "clearcanvas.editor.QFileDialog.getSaveFileName", lambda *a: (str(output), "PNG (*.png)")
    )
    window.editor.export_button.click()
    qtbot.waitUntil(lambda: output.exists() and not window.editor.busy, timeout=15000)
    window.editor.tabs.setCurrentIndex(0)
    assert window.badge.text() == "原件"
    window.editor.clear.trigger()
    assert window.editor.result is None
    assert not window.editor.export_button.isEnabled()


def test_ui_no_detection_preserves_original(qtbot, window, sample_image, monkeypatch):
    messages = []
    monkeypatch.setattr("clearcanvas.editor.QMessageBox.warning", lambda *a: messages.append(a[2]))
    window.add_files([sample_image])
    qtbot.waitUntil(lambda: window._preview is not None)
    original = window._preview.rgba
    window.editor.auto_button.click()
    qtbot.waitUntil(lambda: not window.editor.busy, timeout=10000)
    assert window.editor.result is None
    assert window._preview.rgba == original
    assert len(messages) == 1
    assert "未可靠识别" in messages[0]


def test_ui_structured_pdf(qtbot, window, tmp_path, monkeypatch):
    source = marked_pdf(tmp_path / "marked.pdf")
    window.add_files([source])
    qtbot.waitUntil(lambda: window._preview is not None)
    window.editor.auto_button.click()
    qtbot.waitUntil(lambda: window.editor.result is not None, timeout=15000)
    assert window.editor.result.structured
    monkeypatch.setattr("clearcanvas.editor.QMessageBox.warning", lambda *a: QMessageBox.Yes)
    output = tmp_path / "all.pdf"
    monkeypatch.setattr(
        "clearcanvas.editor.QFileDialog.getSaveFileName", lambda *a: (str(output), "PDF (*.pdf)")
    )
    window.editor.scope.setCurrentIndex(1)
    window.editor.export_button.click()
    qtbot.waitUntil(lambda: output.exists() and not window.editor.busy, timeout=15000)
    assert all("BODY TEXT" in p.extract_text() for p in PdfReader(output).pages)


@pytest.mark.parametrize("angle", [-45, -20, 20, 45, 70])
def test_diagonal_detection_angles(tmp_path, angle):
    image = Image.new("RGBA", (1000, 1000), "white")
    layer = Image.new("RGBA", image.size)
    ImageDraw.Draw(layer).text((200, 450), "SAMPLE WATERMARK", fill="#cccccc", font_size=48)
    image.alpha_composite(layer.rotate(angle, resample=Image.Resampling.BICUBIC))
    mask = detect_watermark_mask(image)
    pixels = np.array(image)
    mark = (pixels[:, :, 0] > 170) & (pixels[:, :, 0] < 230)
    assert np.mean(mask[mark] != 0) > 0.99


@pytest.mark.parametrize("size", [(500, 400), (3000, 2400)])
def test_detection_rescaling(pale_image, size):
    with Image.open(pale_image) as source:
        with source.resize(size, Image.Resampling.LANCZOS) as image:
            mask = detect_watermark_mask(image)
            assert mask.shape == (size[1], size[0])
            assert np.count_nonzero(mask) > 100
            assert not np.any(mask[: int(size[1] * 0.18)])


def test_pdf_crop_rotation_links_preserved(tmp_path):
    path = marked_pdf(tmp_path / "marked.pdf")
    writer = PdfWriter(clone_from=path)
    page = writer.pages[0]
    page.rotate(90)
    page.cropbox = RectangleObject([20, 20, 580, 780])
    page[NameObject("/Annots")] = ArrayObject(
        [
            writer._add_object(
                DictionaryObject(
                    {
                        NameObject("/Type"): NameObject("/Annot"),
                        NameObject("/Subtype"): NameObject("/Link"),
                        NameObject("/Rect"): RectangleObject([40, 650, 100, 670]),
                    }
                )
            )
        ]
    )
    writer.write(path)
    result = auto_repair_document(render_document(path), source_stamp(path))
    output = export_result(result, tmp_path / "out.pdf")
    page = PdfReader(output).pages[0]
    assert page.rotation == 90
    assert tuple(page.cropbox) == (20, 20, 580, 780)
    assert len(page["/Annots"]) == 1
    assert "BODY TEXT" in page.extract_text()


def test_auto_export_cancel_during_pages(tmp_path):
    path = marked_pdf(tmp_path / "marked.pdf")
    result = auto_repair_document(render_document(path), source_stamp(path))
    stop = False

    def progress(done, total):
        nonlocal stop
        stop = done >= 1

    output = tmp_path / "cancelled.pdf"
    with pytest.raises(DocumentError, match="取消"):
        export_result(result, output, pages=[0, 1, 2], cancelled=lambda: stop, progress=progress)
    assert not output.exists()
    assert not list(tmp_path.glob(".clearcanvas-*"))


def test_encrypted_and_form_pdf_auto_rejected(encrypted_pdf, tmp_path):
    preview = render_document(encrypted_pdf, password="clearcanvas-test")
    with pytest.raises(DocumentError, match="加密"):
        auto_repair_document(preview, source_stamp(encrypted_pdf))
    path = marked_pdf(tmp_path / "form.pdf")
    writer = PdfWriter(clone_from=path)
    writer.root_object[NameObject("/AcroForm")] = DictionaryObject(
        {
            NameObject("/Fields"): ArrayObject(),
        }
    )
    writer.write(path)
    with pytest.raises(DocumentError, match="表单"):
        auto_repair_document(render_document(path), source_stamp(path))


def test_chinese_watermark(tmp_path):
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if not font_path.exists():
        pytest.skip("Windows Chinese font fixture")
    image = Image.new("RGBA", (1200, 1000), "white")
    layer = Image.new("RGBA", image.size)
    font = ImageFont.truetype(str(font_path), 64)
    ImageDraw.Draw(layer).text((220, 430), "内部资料仅供参考", fill="#cccccc", font=font)
    image.alpha_composite(layer.rotate(30, resample=Image.Resampling.BICUBIC))
    pixels = np.array(image)
    mark = (pixels[:, :, 0] > 170) & (pixels[:, :, 0] < 230)
    mask = detect_watermark_mask(image)
    assert np.mean(mask[mark] != 0) > 0.99
