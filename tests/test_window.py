import threading

from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QApplication

from clearcanvas.preview_service import PreviewService


def wait_preview(qtbot, window, name, page=0):
    qtbot.waitUntil(
        lambda: (
            window._preview is not None
            and window._preview.info.path.name == name
            and window._preview.page_index == page
            and window.stack.currentWidget() is window.canvas
        ),
        timeout=10_000,
    )


def test_empty_controls(window):
    assert window.stack.currentWidget() is window.empty
    assert not window.fit_action.isEnabled()
    assert not window.next_action.isEnabled()
    assert not window.remove_action.isEnabled()


def test_sidebar_toggles(window):
    for dock in (window.file_dock, window.details_dock):
        action = dock.toggleViewAction()
        assert action.isEnabled()
        assert dock.isVisible()
        action.trigger()
        assert not dock.isVisible()
        action.trigger()
        assert dock.isVisible()


def test_import_deduplicate_and_remove(qtbot, window, sample_image):
    window.add_files([sample_image, sample_image])
    wait_preview(qtbot, window, sample_image.name)
    assert window.files.count() == 1
    assert window.canvas.has_document
    assert window.fit_action.isEnabled()
    assert not window.page_spin.isEnabled()
    window.remove_current()
    assert window.files.count() == 0
    assert window.stack.currentWidget() is window.empty
    assert not window.canvas.has_document


def test_pdf_navigation_and_zoom(qtbot, window, sample_pdf):
    window.add_files([sample_pdf])
    wait_preview(qtbot, window, sample_pdf.name)
    window.next_action.trigger()
    wait_preview(qtbot, window, sample_pdf.name, 1)
    assert window.previous_action.isEnabled()
    window.page_spin.setValue(3)
    wait_preview(qtbot, window, sample_pdf.name, 2)
    assert not window.next_action.isEnabled()
    window.actual_action.trigger()
    assert window.canvas.zoom == 1
    window.zoom_in_action.trigger()
    assert window.canvas.zoom == 1.25
    window.fit_action.trigger()
    assert window.canvas._fit


def test_fast_file_switch_and_remove(qtbot, window, sample_pdf, sample_image):
    window.add_files([sample_pdf, sample_image])
    window.files.setCurrentRow(1)
    wait_preview(qtbot, window, sample_image.name)
    assert window.details.values["kind"].text() == "PNG"
    window.remove_current()
    wait_preview(qtbot, window, sample_pdf.name)
    assert window.files.count() == 1


def test_password_and_clear_on_remove(qtbot, window, encrypted_pdf, monkeypatch):
    monkeypatch.setattr(
        "clearcanvas.window.QInputDialog.getText",
        lambda *args: ("clearcanvas-test", True),
    )
    window.add_files([encrypted_pdf])
    wait_preview(qtbot, window, encrypted_pdf.name)
    assert window._passwords
    window.remove_current()
    assert not window._passwords


def test_password_cancel(qtbot, window, encrypted_pdf, monkeypatch):
    monkeypatch.setattr("clearcanvas.window.QInputDialog.getText", lambda *args: ("", False))
    window.add_files([encrypted_pdf])
    qtbot.waitUntil(lambda: window.status.text() == "已取消打开加密 PDF")
    assert not window._passwords
    assert not window.canvas.has_document


def test_error_after_valid_preview(qtbot, window, sample_image, tmp_path):
    window.add_files([sample_image])
    wait_preview(qtbot, window, sample_image.name)
    broken = tmp_path / "损坏.pdf"
    broken.write_bytes(b"invalid pdf")
    window.add_files([broken])
    qtbot.waitUntil(lambda: window.stack.currentWidget() is window.error, timeout=10_000)
    assert not window.canvas.has_document
    assert not window.fit_action.isEnabled()


def test_local_file_drop(qtbot, window, sample_image):
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(sample_image))])
    drag = QDragEnterEvent(
        QPoint(400, 300),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(window, drag)
    assert drag.isAccepted()
    drop = QDropEvent(
        QPointF(400, 300),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(window, drop)
    wait_preview(qtbot, window, sample_image.name)


def test_latest_request_wins(qtbot, sample_image, sample_pdf, monkeypatch):
    import clearcanvas.preview_service as module

    original = module.render_document
    entered = threading.Event()
    release = threading.Event()

    def slow_render(path, page, password, cancelled):
        if path == sample_pdf:
            entered.set()
            assert release.wait(5)
        return original(path, page, password, cancelled)

    monkeypatch.setattr(module, "render_document", slow_render)
    service = PreviewService()
    results = []
    service.loaded.connect(lambda token, preview: results.append(preview))
    try:
        service.request(sample_pdf, 0, None)
        qtbot.waitUntil(entered.is_set)
        service.request(sample_image, 0, None)
        release.set()
        qtbot.waitUntil(lambda: len(results) == 1, timeout=10_000)
        assert results[0].info.path == sample_image
    finally:
        release.set()
        service.close()


def test_compact_window_geometry(qtbot, window, sample_pdf):
    window.resize(1000, 680)
    window.add_files([sample_pdf])
    wait_preview(qtbot, window, sample_pdf.name)
    qtbot.wait(100)
    assert window.canvas.width() > 400
    assert window.zoom_combo.width() >= 80
    assert window.details.width() >= 200
    rect = window.canvas.mapFromScene(window.canvas._document_rect).boundingRect()
    assert rect.width() <= window.canvas.viewport().width()
    assert rect.height() <= window.canvas.viewport().height()
