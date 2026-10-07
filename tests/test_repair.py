import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw
from pypdf import PdfReader
from PySide6.QtCore import QPointF, Qt

from clearcanvas.documents import DocumentError, render_document
from clearcanvas.repair import export_result, repair_document, source_stamp


@pytest.fixture
def watermarked_image(tmp_path):
    path = tmp_path / "真实处理样本.png"
    image = Image.new("RGB", (600, 400), (185, 210, 224))
    draw = ImageDraw.Draw(image)
    draw.text((215, 187), "WATERMARK", fill=(35, 40, 45), font_size=24)
    image.save(path)
    return path


REGIONS = [(200 / 600, 170 / 400, 385 / 600, 230 / 400)]


def test_watermark_removed_and_outside_pixels_unchanged(watermarked_image, tmp_path):
    before = watermarked_image.read_bytes()
    preview = render_document(watermarked_image)
    result = repair_document(preview, REGIONS, source_stamp(watermarked_image))
    original = np.array(Image.open(watermarked_image).convert("RGBA"))
    repaired = np.array(result.image)
    assert np.abs(repaired[170:230, 200:385, :3].astype(float) - [185, 210, 224]).mean() < 3
    mask = np.ones((400, 600), dtype=bool)
    mask[170:230, 200:385] = False
    assert np.array_equal(original[mask], repaired[mask])
    output = export_result(result, tmp_path / "处理结果.png")
    assert np.array_equal(np.array(Image.open(output)), repaired)
    assert watermarked_image.read_bytes() == before


def test_pdf_repair_and_export_preserve_other_pages(sample_pdf, tmp_path):
    before_hash = hashlib.sha256(sample_pdf.read_bytes()).hexdigest()
    original = render_document(sample_pdf)
    # The fixture's title is the visible mark being removed, not a blank area.
    result = repair_document(original, [(0.06, 0.078, 0.41, 0.128)], source_stamp(sample_pdf))
    output = export_result(result, tmp_path / "PDF去水印.pdf")
    reader = PdfReader(output)
    assert len(reader.pages) == 3
    assert tuple(reader.pages[0].mediabox) == (0, 0, 595, 842)
    assert not reader.pages[0].extract_text().strip()
    assert reader.pages[1].extract_text() == PdfReader(sample_pdf).pages[1].extract_text()
    assert render_document(output, 1).rgba == render_document(sample_pdf, 1).rgba
    processed = render_document(output)
    assert processed.rgba != original.rgba
    rgba = np.frombuffer(processed.rgba, np.uint8).reshape(processed.height, processed.width, 4)
    roi = rgba[
        int(processed.height * 0.082) : int(processed.height * 0.125),
        int(processed.width * 0.075) : int(processed.width * 0.38),
        :3,
    ]
    assert (roi > 235).mean() > 0.99
    assert hashlib.sha256(sample_pdf.read_bytes()).hexdigest() == before_hash


def test_export_does_not_overwrite_original_or_existing(watermarked_image, tmp_path):
    result = repair_document(
        render_document(watermarked_image), REGIONS, source_stamp(watermarked_image)
    )
    with pytest.raises(DocumentError, match="原文件"):
        export_result(result, watermarked_image, True)
    output = tmp_path / "存在.png"
    output.write_bytes(b"keep")
    with pytest.raises(DocumentError, match="已存在"):
        export_result(result, output)
    assert output.read_bytes() == b"keep"


def test_changed_source_rejected(watermarked_image):
    stamp = source_stamp(watermarked_image)
    preview = render_document(watermarked_image)
    watermarked_image.write_bytes(watermarked_image.read_bytes() + b"change")
    with pytest.raises(DocumentError, match="变化"):
        repair_document(preview, REGIONS, stamp)


@pytest.mark.parametrize("regions", [[], [(0, 0, 1, 1)], [(-1, 0, 0.1, 0.1)]])
def test_bad_selection(watermarked_image, regions):
    with pytest.raises(DocumentError):
        repair_document(
            render_document(watermarked_image), regions, source_stamp(watermarked_image)
        )


def test_ui_selection_repair_compare_export(
    qtbot, window, watermarked_image, tmp_path, monkeypatch
):
    window.add_files([watermarked_image])
    qtbot.waitUntil(lambda: window._preview is not None)
    artifacts = Path(__file__).resolve().parents[1] / "artifacts" / "qa"
    artifacts.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(watermarked_image, artifacts / "watermarked-sample.png")
    assert window.grab().save(str(artifacts / "repair-before.png"))
    view = window.canvas
    first = view.mapFromScene(QPointF(200, 170))
    last = view.mapFromScene(QPointF(385, 230))
    qtbot.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=first)
    qtbot.mouseMove(view.viewport(), last)
    qtbot.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=last)
    assert window.editor.repair_button.isEnabled()
    window.editor.repair_button.click()
    qtbot.waitUntil(lambda: window.editor.result is not None, timeout=10000)
    assert window.badge.text() == "结果"
    assert window.grab().save(str(artifacts / "repair-after.png"))
    window.editor.tabs.setCurrentIndex(0)
    assert window.badge.text() == "原件"
    window.editor.tabs.setCurrentIndex(1)
    output = tmp_path / "界面导出.png"
    monkeypatch.setattr(
        "clearcanvas.editor.QFileDialog.getSaveFileName", lambda *args: (str(output), "PNG (*.png)")
    )
    window.editor.export_button.click()
    qtbot.waitUntil(lambda: output.exists() and not window.editor.busy, timeout=10000)
    assert not window.editor.dirty
    assert Image.open(output).size == (600, 400)
    shutil.copyfile(output, artifacts / "watermark-removed.png")


def test_ui_pdf_repair_export(qtbot, window, sample_pdf, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(
        "clearcanvas.editor.QMessageBox.warning", lambda *args: QMessageBox.StandardButton.Yes
    )
    window.add_files([sample_pdf])
    qtbot.waitUntil(lambda: window._preview is not None)
    window.canvas.regions = [(0.06, 0.078, 0.41, 0.128)]
    window.canvas.selection_changed.emit()
    window.editor.repair_button.click()
    qtbot.waitUntil(lambda: window.editor.result is not None, timeout=10000)
    output = tmp_path / "界面去水印.pdf"
    monkeypatch.setattr(
        "clearcanvas.editor.QFileDialog.getSaveFileName", lambda *args: (str(output), "PDF (*.pdf)")
    )
    window.editor.export_button.click()
    qtbot.waitUntil(lambda: output.exists() and not window.editor.busy, timeout=10000)
    assert len(PdfReader(output).pages) == 3


def test_pdf_selected_pages_export(sample_pdf, tmp_path):
    from pypdf import PdfWriter

    reader = PdfReader(sample_pdf)
    writer = PdfWriter()
    for _ in range(3):
        writer.add_page(reader.pages[0])
    source = tmp_path / "同尺寸多页.pdf"
    writer.write(source)
    result = repair_document(
        render_document(source), [(0.06, 0.078, 0.41, 0.128)], source_stamp(source)
    )
    output = export_result(result, tmp_path / "指定页.pdf", pages=[0, 2])
    pages = PdfReader(output).pages
    assert not pages[0].extract_text().strip()
    assert "ClearCanvas" in pages[1].extract_text()
    assert not pages[2].extract_text().strip()


def test_pdf_mixed_sizes_export_rejected(sample_pdf, tmp_path):
    result = repair_document(
        render_document(sample_pdf), [(0.06, 0.078, 0.41, 0.128)], source_stamp(sample_pdf)
    )
    target = tmp_path / "不能错位.pdf"
    with pytest.raises(DocumentError, match="尺寸或方向不同"):
        export_result(result, target, pages=[0, 1])
    assert not target.exists()
    assert not list(tmp_path.glob(".clearcanvas-*"))


def test_batch_partial_failure_and_unique_output(watermarked_image, tmp_path):
    from clearcanvas.repair import batch_export

    reference = repair_document(
        render_document(watermarked_image), REGIONS, source_stamp(watermarked_image)
    )
    broken = tmp_path / "损坏.png"
    broken.write_bytes(b"bad image")
    output_dir = tmp_path / "批量"
    output_dir.mkdir()
    result = batch_export(
        reference,
        [broken, watermarked_image, watermarked_image],
        output_dir,
        "1",
        lambda: False,
        lambda *args: None,
    )
    assert len(result.outputs) == 2
    assert result.outputs[0] != result.outputs[1]
    assert len(result.errors) == 1
    assert all(path.exists() for path in result.outputs)


@pytest.mark.parametrize("value", ["", "0", "3-1", "1,,2", "1-10", "1;a"])
def test_invalid_page_ranges(value):
    from clearcanvas.repair import parse_pages

    with pytest.raises(DocumentError):
        parse_pages(value, 5)


def test_page_ranges_and_cancel(watermarked_image, tmp_path):
    from clearcanvas.repair import parse_pages

    assert parse_pages("1, 3-5,3", 5) == [0, 2, 3, 4]
    result = repair_document(
        render_document(watermarked_image), REGIONS, source_stamp(watermarked_image)
    )
    output = tmp_path / "取消.png"
    with pytest.raises(DocumentError, match="取消"):
        export_result(result, output, cancelled=lambda: True)
    assert not output.exists()


def test_png_alpha_watermark_removed(tmp_path):
    path = tmp_path / "透明水印.png"
    image = Image.new("RGBA", (600, 400), (50, 80, 60, 0))
    ImageDraw.Draw(image).text((215, 187), "WATERMARK", fill=(50, 80, 60, 255), font_size=24)
    image.save(path)
    result = repair_document(render_document(path), REGIONS, source_stamp(path))
    assert np.array(result.image)[170:230, 200:385, 3].max() < 5


def test_visible_small_image_fit_stays_responsive(qtbot, window, watermarked_image):
    window.hide()
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, False)
    window.show()
    window.add_files([watermarked_image])
    qtbot.waitUntil(lambda: window._preview is not None, timeout=5000)
    events = []
    window.canvas.zoom_changed.connect(events.append)
    qtbot.wait(150)
    assert len(events) < 10
    assert not window.canvas.horizontalScrollBar().isVisible()
    assert not window.canvas.verticalScrollBar().isVisible()
    window.editor.select.trigger()
    assert not window.canvas.selection_mode


@pytest.mark.parametrize("kind", ["JPEG", "PDF"])
def test_jpeg_and_scanned_pdf_watermark_removal(watermarked_image, tmp_path, kind):
    source = tmp_path / ("扫描.pdf" if kind == "PDF" else "照片.jpg")
    with Image.open(watermarked_image).convert("RGB") as image:
        image.save(source, format=kind)
    preview = render_document(source)
    result = repair_document(preview, REGIONS, source_stamp(source))
    destination = tmp_path / ("修复.pdf" if kind == "PDF" else "修复.jpg")
    export_result(result, destination)
    output = render_document(destination)
    pixels = np.frombuffer(output.rgba, np.uint8).reshape(output.height, output.width, 4)
    selected = pixels[
        int(0.45 * output.height) : int(0.56 * output.height),
        int(0.35 * output.width) : int(0.64 * output.width),
        :3,
    ]
    assert (selected < 100).mean() < 0.001
