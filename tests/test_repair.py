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
    regions = list(view.regions)
    window.editor.fit_regions_button.click()
    center = view.mapToScene(view.viewport().rect().center())
    zoom = view.zoom
    for _ in range(4):
        qtbot.mousePress(window.editor.hold_original, Qt.MouseButton.LeftButton)
        assert window.editor.temporary_original and window.editor.tabs.currentIndex() == 1
        assert all(not item.isVisible() for item in view._overlays)
        qtbot.mouseRelease(window.editor.hold_original, Qt.MouseButton.LeftButton)
        assert not window.editor.temporary_original
        assert view.zoom == zoom
        current_center = view.mapToScene(view.viewport().rect().center())
        assert (current_center - center).manhattanLength() < 2 / zoom
    window.editor.tabs.setCurrentIndex(0)
    assert window.badge.text() == "原件"
    window.editor.tabs.setCurrentIndex(1)
    assert view.regions == regions and window.editor.dirty
    result = window.editor.result
    window.editor.highlight_switch.click()
    qtbot.waitUntil(lambda: not window.editor.busy, timeout=10000)
    cache = window.editor.highlight_cache
    assert cache is not None and any(cache.mask)
    assert view._highlight.isVisible() and window.editor.dirty
    window.editor.tabs.setCurrentIndex(0)
    assert not view._highlight.isVisible()
    window.editor.tabs.setCurrentIndex(1)
    assert view._highlight.isVisible()
    window.editor.highlight_switch.click()
    window.editor.highlight_switch.click()
    assert window.editor.highlight_cache is cache and not window.editor.busy
    assert window.editor.result is result
    output = tmp_path / "界面导出.png"
    monkeypatch.setattr(
        "clearcanvas.editor.QFileDialog.getSaveFileName", lambda *args: (str(output), "PNG (*.png)")
    )
    window.editor.export_button.click()
    qtbot.waitUntil(lambda: output.exists() and not window.editor.busy, timeout=10000)
    assert not window.editor.dirty
    assert Image.open(output).size == (600, 400)
    assert window.editor.output_path == output
    with Image.open(output) as image:
        assert image.convert("RGBA").tobytes() == result.image.tobytes()
    window.editor.highlight_switch.click()
    without_highlight = tmp_path / "不带高亮.png"
    export_result(result, without_highlight)
    with Image.open(output) as enabled, Image.open(without_highlight) as disabled:
        assert enabled.tobytes() == disabled.tobytes()
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
    window.editor.pan.trigger()
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


def _draw_region(qtbot, view, first, last):
    start = view.mapFromScene(QPointF(*first))
    end = view.mapFromScene(QPointF(*last))
    qtbot.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=start)
    qtbot.mouseMove(view.viewport(), end)
    qtbot.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=end)


def test_space_gesture_focus_and_selection_history(qtbot, window, watermarked_image):
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtWidgets import QLineEdit

    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, False)
    window.show()
    window.activateWindow()
    window.add_files([watermarked_image])
    qtbot.waitUntil(lambda: window._preview is not None)
    view = window.canvas
    view.set_zoom(2)
    view.centerOn(300, 200)
    view.setFocus()
    qtbot.wait(30)
    start, end = QPointF(245, 180), QPointF(285, 205)
    qtbot.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=view.mapFromScene(start))
    qtbot.keyPress(view, Qt.Key.Key_Space)
    qtbot.mouseMove(view.viewport(), view.mapFromScene(end))
    qtbot.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=view.mapFromScene(end))
    qtbot.keyRelease(view, Qt.Key.Key_Space)
    assert len(view.regions) == 1
    first = list(view.regions)
    center = view.mapToScene(view.viewport().rect().center())
    qtbot.keyPress(view, Qt.Key.Key_Space)
    for kind in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
        QCoreApplication.sendEvent(
            view, QKeyEvent(kind, Qt.Key.Key_Space, Qt.NoModifier, " ", True)
        )
    position = view.viewport().rect().center()
    qtbot.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=position)
    qtbot.keyRelease(view, Qt.Key.Key_Space)
    assert view._gesture == "pan"
    offset = view.mapFromScene(QPointF(20, 10)) - view.mapFromScene(QPointF(0, 0))
    qtbot.mouseMove(view.viewport(), position + offset)
    qtbot.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=position)
    assert view.regions == first and view.selection_mode and window.editor.select.isChecked()
    assert view.viewport().cursor().shape() == Qt.CursorShape.CrossCursor
    assert (view.mapToScene(view.viewport().rect().center()) - center).manhattanLength() > 1
    view.centerOn(300, 200)
    _draw_region(qtbot, view, (310, 210), (340, 230))
    original = list(view.regions)
    assert len(original) == 2
    for _ in range(2):
        qtbot.keyClick(view, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert not view.regions and window.editor.redo.isEnabled()
    for _ in range(2):
        qtbot.keyClick(view, Qt.Key.Key_Y, Qt.KeyboardModifier.ControlModifier)
    assert view.regions == original
    window.editor.finish(repair_document(
        window._preview, original, source_stamp(watermarked_image)
    ))
    window.editor.undo.trigger()
    assert window.editor.result is None and not window.editor.dirty
    window.editor.redo.trigger()
    assert view.regions == original and window.editor.result is None
    window.editor.undo.trigger()
    _draw_region(qtbot, view, (330, 180), (350, 195))
    assert not view.redo_regions and not window.editor.redo.isEnabled()
    qtbot.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=view.mapFromScene(start))
    qtbot.keyPress(view, Qt.Key.Key_Space)
    QCoreApplication.sendEvent(view, QEvent(QEvent.Type.FocusOut))
    qtbot.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=view.mapFromScene(end))
    assert len(view.regions) == 2 and view._draft is None and not view._space_down
    entry = QLineEdit(window)
    entry.show()
    entry.setFocus()
    entry.setText("native")
    entry.selectAll()
    qtbot.keyClicks(entry, "with space")
    qtbot.keyClick(entry, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    qtbot.keyClick(entry, Qt.Key.Key_Y, Qt.KeyboardModifier.ControlModifier)
    assert entry.text() == "with space" and len(view.regions) == 2
    window.editor.clear.trigger()
    assert not view.regions and not view.redo_regions


@pytest.mark.parametrize("kind", ["PNG", "PDF"])
def test_fit_all_regions_clamps_zoom_without_changing_coordinates(
    qtbot, window, sample_image, sample_pdf, kind
):
    window.add_files([sample_pdf if kind == "PDF" else sample_image])
    qtbot.waitUntil(lambda: window._preview is not None)
    view = window.canvas
    view.regions = [(0, 0, 0.2, 0.2), (0.75, 0.8, 1, 1)]
    original = list(view.regions)
    window.editor.refresh()
    window.fit_regions_action.trigger()
    width, height = window._preview.logical_size
    viewport = view.viewport().rect().adjusted(20, 20, -20, -20)
    for x, y in ((0, 0), (width, height)):
        assert viewport.contains(view.mapFromScene(QPointF(x, y)))
    assert view.regions == original and not view._fit
    view.regions = [(0.499, 0.499, 0.501, 0.501)]
    window.editor.fit_regions_button.click()
    assert view.zoom == 8.0
    window.resize(1100, 720)
    qtbot.wait(30)
    assert view.zoom == 8.0


def test_output_target_tracks_real_partial_success_and_missing_file(
    qtbot, window, tmp_path, monkeypatch, watermarked_image
):
    from PySide6.QtWidgets import QMessageBox

    from clearcanvas.repair import batch_export

    monkeypatch.setattr(QMessageBox, "information", lambda *a: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a: QMessageBox.Ok)
    output = tmp_path / "中文 空格 # % [结果].png"
    with Image.new("RGBA", (12, 8), "green") as image:
        image.save(output)
    opened = []
    monkeypatch.setattr(
        "clearcanvas.editor.QDesktopServices.openUrl",
        lambda url: opened.append(url.toLocalFile()) or False,
    )
    window.editor.finish(output)
    window.editor.open_file_button.click()
    assert Path(opened[-1]) == output and window.editor.output_path == output
    output.unlink()
    before = list(opened)
    window.editor.open_file_button.click()
    assert opened == before
    window.editor.open_folder_button.click()
    assert Path(opened[-1]) == tmp_path
    batch_directory = tmp_path / "批量 输出"
    batch_directory.mkdir()
    wrong_size = tmp_path / "尺寸不同.png"
    with Image.new("RGB", (16, 16), "white") as image:
        image.save(wrong_size)
    reference = repair_document(
        render_document(watermarked_image), REGIONS, source_stamp(watermarked_image)
    )
    try:
        batch = batch_export(
            reference, [wrong_size, watermarked_image, watermarked_image], batch_directory, "1",
            lambda: bool(list(batch_directory.glob("*.png"))), lambda *a: None,
        )
        assert batch.cancelled and len(batch.outputs) == 1 and len(batch.errors) == 1
        window.editor.finish(batch)
        assert window.editor.output_path == batch_directory
        assert not window.editor.open_file_button.isVisible()
        window.editor.open_folder_button.click()
        assert Path(opened[-1]) == batch_directory
        failed = batch_export(
            reference, [wrong_size], batch_directory, "1", lambda: False, lambda *a: None
        )
        assert not failed.outputs
        window.editor.finish(failed)
        assert window.editor.output_path == batch_directory and window.editor.output_count == 1
    finally:
        reference.image.close()


@pytest.mark.parametrize("outcome", ["cancel", "source_changed", "expired"])
def test_highlight_failure_cancel_and_stale_reply_preserve_dirty_result(
    qtbot, window, watermarked_image, monkeypatch, outcome
):
    from dataclasses import replace
    from threading import Event

    from PySide6.QtWidgets import QMessageBox

    from clearcanvas.change_highlight import ChangeHighlight, compute_change_highlight

    monkeypatch.setattr(QMessageBox, "warning", lambda *a: QMessageBox.Ok)
    window.add_files([watermarked_image])
    qtbot.waitUntil(lambda: window._preview is not None)
    result = repair_document(window._preview, REGIONS, source_stamp(watermarked_image))
    window.editor.finish(result)
    started, resume = Event(), Event()

    def compute(value, cancelled):
        started.set()
        assert resume.wait(5)
        return compute_change_highlight(value, cancelled)

    monkeypatch.setattr("clearcanvas.editor.compute_change_highlight", compute)
    try:
        window.editor.highlight_switch.click()
        qtbot.waitUntil(started.is_set)
        assert not window.files.isEnabled() and not window.canvas.isEnabled()
        marker = window.editor.highlight_task
        fake = ChangeHighlight(1, 1, b"\xff")
        window._loaded(window._token, fake)
        assert window.editor.busy and window.editor.dirty and window.editor.highlight_task is marker
        if outcome == "cancel":
            window.editor.cancel()
        elif outcome == "source_changed":
            watermarked_image.write_bytes(b"changed externally")
        else:
            window._preview = replace(window._preview, page_index=1)
        resume.set()
        qtbot.waitUntil(lambda: not window.editor.busy, timeout=10000)
        assert window.editor.result is result and window.editor.dirty
        assert result.image.size == (600, 400)
        assert window.editor.highlight_cache is None
        assert not window.editor.highlight_switch.isChecked()
        window._loaded(window._token, fake)
        assert window.editor.result is result and window.editor.dirty
    finally:
        resume.set()
        window.editor.dirty = False


def test_hold_original_keyboard_and_lost_release_restore_result(qtbot, window, watermarked_image):
    from PySide6.QtCore import QCoreApplication, QEvent

    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, False)
    window.show()
    window.activateWindow()
    window.add_files([watermarked_image])
    qtbot.waitUntil(lambda: window._preview is not None)
    window.editor.finish(repair_document(window._preview, REGIONS, source_stamp(watermarked_image)))
    button = window.editor.hold_original
    button.setFocus()
    qtbot.keyPress(button, Qt.Key.Key_Space)
    assert window.editor.temporary_original and window.editor.tabs.currentIndex() == 1
    qtbot.keyRelease(button, Qt.Key.Key_Space)
    assert not window.editor.temporary_original
    qtbot.mousePress(button, Qt.MouseButton.LeftButton)
    QCoreApplication.sendEvent(button, QEvent(QEvent.Type.UngrabMouse))
    assert not window.editor.temporary_original and window.editor.dirty
    qtbot.mouseRelease(button, Qt.MouseButton.LeftButton)
    qtbot.mousePress(button, Qt.MouseButton.LeftButton)
    QCoreApplication.sendEvent(window, QEvent(QEvent.Type.WindowDeactivate))
    assert not window.editor.temporary_original and window.editor.tabs.currentIndex() == 1
    qtbot.mouseRelease(button, Qt.MouseButton.LeftButton)
    window.editor.dirty = False


@pytest.mark.parametrize("mode", ["structured", "manual", "automatic"])
def test_pdf_processing_info_uses_current_result_not_export_scope(
    qtbot, window, tmp_path, sample_pdf, monkeypatch, mode
):
    from PySide6.QtWidgets import QMessageBox
    from test_automatic import marked_pdf

    monkeypatch.setattr(QMessageBox, "warning", lambda *a: QMessageBox.Yes)
    path = marked_pdf(tmp_path / "marked.pdf") if mode == "structured" else sample_pdf
    if mode == "automatic":
        with Image.new("RGBA", (1000, 800), "white") as image:
            layer = Image.new("RGBA", image.size)
            ImageDraw.Draw(layer).text(
                (220, 340), "SAMPLE WATERMARK", fill=(205, 205, 205), font_size=48
            )
            with layer.rotate(30, resample=Image.Resampling.BICUBIC) as rotated:
                image.alpha_composite(rotated)
            layer.close()
            with image.convert("RGB") as rgb:
                path = tmp_path / "扫描 水印.pdf"
                rgb.save(path, "PDF", save_all=True, append_images=[rgb])
    window.add_files([path])
    qtbot.waitUntil(lambda: window._preview is not None)
    editor = window.editor
    if mode != "manual":
        editor.auto_button.click()
    else:
        editor.window.canvas.regions = [(0.06, 0.078, 0.41, 0.128)]
        editor.start()
    qtbot.waitUntil(lambda: editor.result is not None, timeout=15000)
    assert editor.result.structured == (mode == "structured")
    text = editor.processing_info.text()
    assert ("保留文字层" in text) == editor.result.structured
    assert ("文字不能复制" in text) != editor.result.structured
    editor.scope.setCurrentIndex(1)
    assert text in editor.processing_info.text()
    assert "其他页" in editor.processing_info.text()
    editor.ai.mode.setCurrentIndex(1)
    assert not editor.ai.cloud and text in editor.processing_info.text()
    editor.dirty = False
    window.page_spin.setValue(2)
    qtbot.waitUntil(lambda: window._preview.page_index == 1)
    assert editor.result is None and not window.canvas.regions and not window.canvas.redo_regions
    assert "文字层" not in editor.processing_info.text()


@pytest.mark.parametrize("zoom", [0.25, 1.0, 4.0, 8.0])
def test_click_jitter_preserves_result_until_intentional_selection(
    qtbot, window, watermarked_image, zoom
):
    from PySide6.QtCore import QPoint

    window.add_files([watermarked_image])
    qtbot.waitUntil(lambda: window._preview is not None)
    view, editor = window.canvas, window.editor
    view.regions = list(REGIONS)
    editor.finish(repair_document(window._preview, REGIONS, source_stamp(watermarked_image)))
    result = editor.result
    view.set_zoom(zoom)
    view.centerOn(300, 200)
    editor.select.trigger()
    assert editor.result is result and editor.tabs.currentIndex() == 1
    position = view.mapFromScene(QPointF(300, 200))
    qtbot.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=position)
    qtbot.mouseMove(view.viewport(), position + QPoint(2, 2))
    qtbot.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=position + QPoint(2, 2))
    assert editor.result is result and editor.dirty and editor.export_button.isEnabled()
    assert view.regions == REGIONS and view._draft is None
    editor.pan.trigger()
    assert editor.result is result and not view.selection_mode
    editor.select.trigger()
    _draw_region(qtbot, view, (280, 190), (340, 240))
    assert editor.result is None and not editor.dirty and not editor.export_button.isEnabled()
    assert len(view.regions) == 2 and view.regions[0] == REGIONS[0]
    assert editor.select.isChecked()
