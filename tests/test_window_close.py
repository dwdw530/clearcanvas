from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QMessageBox, QSystemTrayIcon

from clearcanvas.window import MainWindow


def test_close_to_tray_keeps_preview_and_service(qtbot, window, sample_image, monkeypatch):
    window.add_files([sample_image])
    qtbot.waitUntil(lambda: window._preview is not None)
    preview = window._preview
    token = window._token
    monkeypatch.setattr(window, "_choose_close_action", lambda: "minimize")
    monkeypatch.setattr(QSystemTrayIcon, "isSystemTrayAvailable", lambda: True)
    monkeypatch.setattr(window.editor, "can_leave", lambda: False)
    assert not window.close()
    assert not window.isVisible()
    assert window.tray.isVisible()
    assert window._preview is preview
    assert window._token == token
    window._tray_activated(QSystemTrayIcon.ActivationReason.Trigger)
    assert window.isVisible()
    assert window._preview is preview
    monkeypatch.setattr(window.editor, "can_leave", lambda: True)


def test_no_tray_falls_back_to_taskbar(window, monkeypatch):
    monkeypatch.setattr(QSystemTrayIcon, "isSystemTrayAvailable", lambda: False)
    monkeypatch.setattr(window, "_choose_close_action", lambda: "minimize")
    window.close()
    assert window.isVisible()
    assert window.isMinimized()
    assert not window.tray.isVisible()
    window.restore_window()
    assert not window.isMinimized()


def test_cancel_close_keeps_window(window, monkeypatch):
    monkeypatch.setattr(window, "_choose_close_action", lambda: "cancel")
    assert not window.close()
    assert window.isVisible()
    assert not window._closed


def test_tray_exit_preserves_unsaved_guard(window, monkeypatch):
    monkeypatch.setattr(window.editor, "can_leave", lambda: False)
    window.hide()
    window.request_exit()
    assert window.isVisible()
    assert not window._closed
    assert not window._exit_requested
    monkeypatch.setattr(window.editor, "can_leave", lambda: True)
    window.request_exit()
    assert window._closed
    assert not window.isVisible()
    assert not window.tray.isVisible()


def test_busy_task_can_minimize_but_cannot_exit(window, monkeypatch):
    notices = []
    monkeypatch.setattr(window.editor, "busy", True)
    monkeypatch.setattr(QMessageBox, "information", lambda *args: notices.append(args))
    window.request_exit()
    assert notices
    assert not window._closed
    monkeypatch.setattr(QSystemTrayIcon, "isSystemTrayAvailable", lambda: True)
    monkeypatch.setattr(window, "_choose_close_action", lambda: "minimize")
    window.close()
    assert not window.isVisible()
    monkeypatch.setattr(window.editor, "busy", False)


def test_close_dialog_buttons_and_escape(window, monkeypatch, qtbot):
    monkeypatch.setattr(QSystemTrayIcon, "isSystemTrayAvailable", lambda: True)
    for text, expected in (
        ("最小化到托盘", "minimize"),
        ("退出程序", "exit"),
        ("取消", "cancel"),
        (None, "cancel"),
    ):
        def choose(text=text):
            box = QApplication.activeModalWidget()
            assert isinstance(box, QMessageBox)
            assert box.defaultButton().text() == "最小化到托盘"
            if text is None:
                qtbot.keyClick(box, Qt.Key.Key_Escape)
            else:
                next(button for button in box.buttons() if button.text() == text).click()

        QTimer.singleShot(0, choose)
        assert MainWindow._choose_close_action(window) == expected


def test_reentrant_close_does_not_open_another_dialog(window, monkeypatch):
    def choose():
        nested = QCloseEvent()
        window.closeEvent(nested)
        assert not nested.isAccepted()
        return "cancel"

    monkeypatch.setattr(window, "_choose_close_action", choose)
    assert not window.close()
    assert window.isVisible()
