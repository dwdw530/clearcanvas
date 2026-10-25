from io import BytesIO

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QObject, QSettings, QTimer, Signal
from PySide6.QtWidgets import QDialogButtonBox, QFileDialog, QMessageBox
from test_ai_repair import response
from test_ai_settings import MemoryCredentials

from clearcanvas import ai_repair, ai_settings_dialog
from clearcanvas.ai_service import AiRequest
from clearcanvas.ai_settings import AiConfig, AiSettings
from clearcanvas.ai_settings_dialog import AiSettingsDialog
from clearcanvas.documents import DocumentError
from clearcanvas.repair import repair_document


class FakeService(QObject):
    completed = Signal(int, object, object)
    stage = Signal(int, str)

    def __init__(self):
        super().__init__()
        self.requests = []
        self.active = None
        self.error = None
        self.hang = False
        self.return_changed = False
        self.closed = False

    def submit(self, request):
        assert isinstance(request, AiRequest)
        assert self.active is None
        self.requests.append(request)
        self.active = len(self.requests)
        token = self.active

        def finish():
            if self.active != token:
                return
            if self.hang:
                return
            self.active = None
            self.stage.emit(token, "AI 处理中")
            with Image.open(BytesIO(request.image)) as image:
                if self.return_changed:
                    image.paste("blue", (0, 0, image.width, image.height))
                self.completed.emit(token, response(image), self.error)

        QTimer.singleShot(5, finish)
        return token

    def cancel(self):
        if self.active is not None:
            token, self.active = self.active, None
            QTimer.singleShot(
                0, lambda: self.completed.emit(token, None, DocumentError("已取消。", "cancelled"))
            )

    def close(self):
        self.closed = True


@pytest.fixture
def ai_window(window, tmp_path, monkeypatch):
    controller = window.editor.ai
    controller.service.close()
    service = FakeService()
    controller.service = service
    service.completed.connect(controller._network_done)
    service.stage.connect(controller._stage)
    store = AiSettings(
        QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat), MemoryCredentials()
    )
    store.save(AiConfig(), "OFFLINE-FAKE-KEY")
    controller.store = store
    monkeypatch.setattr(controller, "confirm_plan", lambda plan: True)
    monkeypatch.setattr(controller, "confirm_alignment", lambda: True)
    monkeypatch.setattr(QMessageBox, "question", lambda *a: QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a: QMessageBox.StandardButton.Cancel)
    monkeypatch.setattr(QMessageBox, "information", lambda *a: QMessageBox.StandardButton.Ok)
    yield window, controller, service
    window.editor.dirty = False


def load_image(qtbot, window, sample_image):
    window.add_files([sample_image])
    qtbot.waitUntil(lambda: window._preview is not None)
    window.canvas.regions = [(0.55, 0.65, 0.65, 0.7)]


def test_local_default_mode_and_export_never_request(
    qtbot, ai_window, sample_image, tmp_path, monkeypatch
):
    window, ai, service = ai_window
    assert not ai.cloud
    load_image(qtbot, window, sample_image)
    window.editor.start()
    qtbot.waitUntil(lambda: not window.editor.busy, timeout=10000)
    assert window.editor.result.backend == "local"
    assert not service.requests
    result = window.editor.result
    ai.mode.setCurrentIndex(1)
    assert window.editor.result is result
    assert window.editor.batch_button.isEnabled()
    output = tmp_path / "local.png"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a: (str(output), "PNG (*.png)"))
    window.editor.export()
    qtbot.waitUntil(lambda: not window.editor.busy)
    assert output.exists() and not service.requests


def test_cloud_manual_success_source_labels_export_no_second_call(
    qtbot, ai_window, sample_image, tmp_path, monkeypatch
):
    window, ai, service = ai_window
    load_image(qtbot, window, sample_image)
    ai.mode.setCurrentIndex(1)
    window.editor.start()
    qtbot.waitUntil(lambda: not window.editor.busy, timeout=10000)
    result = window.editor.result
    assert result is not None and result.backend == "cloud_ai"
    assert len(service.requests) == 1
    assert "云端 AI" in window.badge.text()
    assert not window.editor.batch_button.isEnabled()
    ai.mode.setCurrentIndex(0)
    assert window.editor.result is result
    assert not window.editor.batch_button.isEnabled()
    output = tmp_path / "cloud.png"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a: (str(output), "PNG (*.png)"))
    window.editor.export()
    qtbot.waitUntil(lambda: not window.editor.busy)
    assert output.exists()
    assert len(service.requests) == 1
    assert ai.key == "" and ai.plan is None


def test_declined_upload_no_requests_old_result_preserved(
    qtbot, ai_window, sample_image, monkeypatch
):
    window, ai, service = ai_window
    load_image(qtbot, window, sample_image)
    old = repair_document(window._preview, window.canvas.regions, window.editor.stamp)
    window.editor.finish(old)
    monkeypatch.setattr(ai, "confirm_plan", lambda p: False)
    ai.mode.setCurrentIndex(1)
    window.editor.start()
    qtbot.waitUntil(lambda: not window.editor.busy)
    assert not service.requests
    assert window.editor.result is old and window.editor.dirty


def test_cloud_auto_detection_local_and_empty_never_uploads(
    qtbot, ai_window, sample_image, monkeypatch
):
    window, ai, service = ai_window
    load_image(qtbot, window, sample_image)
    ai.mode.setCurrentIndex(1)

    def no_marks(*a):
        raise DocumentError("未识别，请框选。", "no_watermark")

    monkeypatch.setattr(ai_repair, "detect_watermark", no_marks)
    window.editor.start_auto()
    qtbot.waitUntil(lambda: not window.editor.busy)
    assert not service.requests

    class Detection:
        mask = np.zeros((640, 960), np.uint8)

    Detection.mask[400:420, 550:600] = 255
    monkeypatch.setattr(ai_repair, "detect_watermark", lambda *a: Detection())
    window.editor.start_auto()
    qtbot.waitUntil(lambda: not window.editor.busy, timeout=10000)
    assert len(service.requests) == 1
    assert window.editor.result.automatic
    assert window.editor.result.backend == "cloud_ai"


@pytest.mark.parametrize("cancel", [False, True])
def test_failure_cancel_stale_reply_preserves_result(qtbot, ai_window, sample_image, cancel):
    window, ai, service = ai_window
    load_image(qtbot, window, sample_image)
    old = repair_document(window._preview, window.canvas.regions, window.editor.stamp)
    window.editor.finish(old)
    service.hang = True
    ai.mode.setCurrentIndex(1)
    window.editor.start()
    qtbot.waitUntil(lambda: service.active is not None)
    assert not window.ai_settings_action.isEnabled()
    assert not ai.mode.isEnabled()
    assert not window.editor.export_button.isEnabled()
    token = service.active
    if cancel:
        window.editor.cancel()
    else:
        service.active = None
        service.completed.emit(token, None, DocumentError("模拟失败", "http"))
    qtbot.waitUntil(lambda: not window.editor.busy, timeout=1000)
    assert window.editor.result is old
    service.completed.emit(token, b"late", None)
    assert window.editor.result is old and len(service.requests) == 1


def test_uncertain_alignment_decline_blocks_new_result(qtbot, ai_window, sample_image, monkeypatch):
    window, ai, service = ai_window
    load_image(qtbot, window, sample_image)
    service.return_changed = True
    decisions = []
    monkeypatch.setattr(ai, "confirm_alignment", lambda: decisions.append(True) and False)
    ai.mode.setCurrentIndex(1)
    window.editor.start()
    qtbot.waitUntil(lambda: not window.editor.busy, timeout=10000)
    assert decisions
    assert window.editor.result is None
    assert not window.editor.export_button.isEnabled()
    assert len(service.requests) == 1


def test_pdf_forces_local_and_image_does_not_restore_cloud(
    qtbot, ai_window, sample_pdf, sample_image
):
    window, ai, service = ai_window
    ai.mode.setCurrentIndex(1)
    window.add_files([sample_pdf])
    qtbot.waitUntil(lambda: window._preview is not None)
    assert not ai.cloud
    assert "PDF" in window.status.text()
    window.add_files([sample_image])
    qtbot.waitUntil(lambda: window._preview is not None and window._preview.info.kind != "PDF")
    assert not ai.cloud and not service.requests


def test_test_interface_without_document_and_config_cancel(qtbot, ai_window, monkeypatch):
    window, ai, service = ai_window
    assert window._active_key is None
    dialog = AiSettingsDialog(ai.store, ai, window)
    qtbot.addWidget(dialog)
    before = ai.store.config
    dialog.model.setText("draft-model")
    monkeypatch.setattr(ai_settings_dialog, "confirm", lambda *a: True)
    dialog._test()
    qtbot.waitUntil(lambda: not dialog.testing, timeout=10000)
    assert "验证通过" in dialog.status.text()
    assert dialog.receipt is not None
    assert len(service.requests) == 1
    assert service.requests[0].config.model == "draft-model"
    assert window.editor.result is None
    dialog.reject()
    assert ai.store.config == before
    assert not ai.cloud


def test_settings_draft_key_isolation_delete_cancel_save(qtbot, ai_window, monkeypatch):
    window, ai, service = ai_window
    root = ai.store.config.root
    ai.store.credentials.values[root] = "SAVED-FAKE-KEY"
    dialog = AiSettingsDialog(ai.store, ai, window)
    qtbot.addWidget(dialog)
    assert dialog.key.echoMode() == dialog.key.EchoMode.Password
    dialog.root.setText("https://other.example")
    assert dialog.key.text() == ""
    dialog.root.setText(root)
    monkeypatch.setattr(ai_settings_dialog, "confirm", lambda *a: True)
    dialog._delete()
    assert root in ai.store.credentials.values
    dialog.reject()
    assert root in ai.store.credentials.values
    dialog = AiSettingsDialog(ai.store, ai, window)
    qtbot.addWidget(dialog)
    dialog._delete()
    dialog._save()
    assert root not in ai.store.credentials.values
    assert ai.store.key_for(root) == ""
    assert not service.requests and not ai.cloud


def test_settings_cancel_test_consent_and_receipt_invalidation(qtbot, ai_window, monkeypatch):
    window, ai, service = ai_window
    dialog = AiSettingsDialog(ai.store, ai, window)
    qtbot.addWidget(dialog)
    monkeypatch.setattr(ai_settings_dialog, "confirm", lambda *a: False)
    dialog._test()
    assert not service.requests
    monkeypatch.setattr(ai_settings_dialog, "confirm", lambda *a: True)
    dialog._test()
    qtbot.waitUntil(lambda: not dialog.testing)
    assert dialog.receipt
    dialog.key.setText("NEW-OFFLINE-KEY")
    assert dialog.receipt is None
    dialog._save()
    assert not ai.store.verified and not ai.cloud


def test_cancel_settings_test_no_document(qtbot, ai_window, monkeypatch):
    window, ai, service = ai_window
    service.hang = True
    dialog = AiSettingsDialog(ai.store, ai, window)
    qtbot.addWidget(dialog)
    monkeypatch.setattr(ai_settings_dialog, "confirm", lambda *a: True)
    dialog._test()
    qtbot.waitUntil(lambda: service.active is not None)
    dialog.reject()
    qtbot.waitUntil(lambda: not dialog.testing)
    assert not ai.active and not window.editor.busy
    assert not dialog.receipt
    dialog.reject()


def test_source_changed_during_wait_and_version_stale(qtbot, ai_window, sample_image):
    window, ai, service = ai_window
    load_image(qtbot, window, sample_image)
    service.hang = True
    ai.mode.setCurrentIndex(1)
    window.editor.start()
    qtbot.waitUntil(lambda: service.active is not None)
    ai.store.version = "changed-version"
    token, service.active = service.active, None
    service.completed.emit(token, b"invalid", None)
    qtbot.waitUntil(lambda: not window.editor.busy)
    assert window.editor.result is None


def test_dialog_compact_keyboard_labels_and_default_cancel(qtbot, ai_window):
    window, ai, _ = ai_window
    window.resize(1000, 680)
    dialog = AiSettingsDialog(ai.store, ai, window)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.wait(30)
    assert dialog.root.accessibleName() or dialog.root.parentWidget() is not None
    assert dialog.key.accessibleName() == "API Key"
    assert dialog.width() <= 1000 and dialog.height() <= 680
    assert dialog.buttons.button(QDialogButtonBox.StandardButton.Save).isVisible()
    assert dialog.buttons.button(QDialogButtonBox.StandardButton.Cancel).isVisible()
    dialog.reject()


def test_saved_test_receipt_noop_save_and_settings_preserve_result(
    qtbot, ai_window, sample_image, monkeypatch
):
    window, ai, service = ai_window
    load_image(qtbot, window, sample_image)
    old = repair_document(window._preview, window.canvas.regions, window.editor.stamp)
    window.editor.finish(old)
    ai.store.verified = True
    dialog = AiSettingsDialog(ai.store, ai, window)
    qtbot.addWidget(dialog)
    assert dialog.receipt
    dialog._save()
    assert ai.store.verified
    assert window.editor.result is old
    assert not ai.cloud and not service.requests


def test_settings_credential_failure_session_fallback_and_new_key_after_delete(
    qtbot, ai_window, monkeypatch
):
    window, ai, service = ai_window
    dialog = AiSettingsDialog(ai.store, ai, window)
    qtbot.addWidget(dialog)
    ai.store.credentials.fail = True
    monkeypatch.setattr(ai_settings_dialog, "confirm", lambda *a: True)
    dialog._delete()
    dialog.key.setText("REPLACEMENT-FAKE-KEY")
    assert not dialog.deleted
    dialog.remember.setChecked(True)
    dialog._save()
    assert ai.store.key_for(ai.store.config.root) == "REPLACEMENT-FAKE-KEY"
    assert not ai.store.remember
    assert not service.requests


def test_missing_key_no_network_and_keyboard_escape(qtbot, ai_window, sample_image, monkeypatch):
    from PySide6.QtCore import Qt

    window, ai, service = ai_window
    load_image(qtbot, window, sample_image)
    ai.store._keys[ai.store.config.root] = ""
    opened = []
    monkeypatch.setattr(ai, "open_settings", lambda: opened.append(True))
    ai.mode.setCurrentIndex(1)
    window.editor.start()
    assert opened and not service.requests and not window.editor.busy
    dialog = AiSettingsDialog(ai.store, ai, window)
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.key.setText("DRAFT-FAKE-KEY")
    dialog.root.setFocus()
    qtbot.keyClick(dialog.root, Qt.Key.Key_Tab)
    assert dialog.focusWidget() is not None
    qtbot.keyClick(dialog, Qt.Key.Key_Escape)
    assert not dialog.isVisible()
    assert ai.store.key_for(ai.store.config.root) == ""


def test_original_file_modified_during_network_rejects_compose(qtbot, ai_window, sample_image):
    window, ai, service = ai_window
    load_image(qtbot, window, sample_image)
    service.hang = True
    ai.mode.setCurrentIndex(1)
    window.editor.start()
    qtbot.waitUntil(lambda: service.active is not None)
    sample_image.write_bytes(b"modified-during-request")
    token, service.active = service.active, None
    service.completed.emit(token, response(ai.plan.canvas), None)
    qtbot.waitUntil(lambda: not window.editor.busy, timeout=10000)
    assert window.editor.result is None
    assert len(service.requests) == 1
