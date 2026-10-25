"""Offline Qt/DPI evidence: synthetic images only, no credentials or network access."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QSettings, QTimer
from PySide6.QtNetwork import QNetworkAccessManager
from PySide6.QtWidgets import QApplication, QDialogButtonBox, QToolBar

from clearcanvas import ai_controller
from clearcanvas.ai_settings import AiSettings
from clearcanvas.ai_settings_dialog import AiSettingsDialog, review_dialog
from clearcanvas.theme import apply_theme
from clearcanvas.window import MainWindow


class NoCredentials:
    def read(self, root):
        raise AssertionError("Credential access forbidden")

    def write(self, root, key):
        raise AssertionError("Credential access forbidden")

    def delete(self, root):
        raise AssertionError("Credential access forbidden")


def main():
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    app = QApplication([])
    apply_theme(app)
    settings = QSettings(str(output / "isolated.ini"), QSettings.Format.IniFormat)
    ai_controller.AiSettings = lambda: AiSettings(settings, NoCredentials())

    def forbidden(*args):
        raise AssertionError("Network access forbidden")

    QNetworkAccessManager.post = forbidden
    window = MainWindow()
    window.resize(1000, 680)
    window.editor.ai.mode.setCurrentIndex(1)
    window.show()
    app.processEvents()
    window.grab().save(str(output / "workbench.png"))
    assert window.width() == 1000 and window.height() == 680
    assert window.editor.ai.panel.isVisible()
    assert all(bar.width() <= 1000 for bar in window.findChildren(QToolBar))
    dialog = AiSettingsDialog(window.editor.ai.store, window.editor.ai, window)
    dialog.show()
    app.processEvents()
    dialog.grab().save(str(output / "settings.png"))
    for kind in (QDialogButtonBox.StandardButton.Save, QDialogButtonBox.StandardButton.Cancel):
        button = dialog.buttons.button(kind)
        assert button.isVisible() and dialog.rect().contains(
            button.mapTo(dialog, button.rect().center())
        )
    assert dialog.root.accessibleName() and dialog.model.accessibleName()
    dialog.reject()
    review_size = []

    def close_review():
        active = QApplication.activeModalWidget()
        active.grab().save(str(output / "review.png"))
        buttons = active.findChild(QDialogButtonBox)
        assert buttons.button(QDialogButtonBox.StandardButton.Cancel).isDefault()
        review_size.extend([active.width(), active.height()])
        active.reject()

    QTimer.singleShot(30, close_review)
    with Image.new("RGB", (1536, 1024), "#dceee8") as image:
        assert not review_dialog(
            window,
            "需核对对齐（离线合成图）",
            "不进行任何网络请求。",
            [(image, "上传图"), (image, "完整返回图")],
        )
    result = {
        "scale": window.devicePixelRatioF(),
        "workbench": [window.width(), window.height()],
        "settings": [dialog.width(), dialog.height()],
        "review": review_size,
        "network_requests": 0,
        "real_credentials_accessed": False,
    }
    window.request_exit()
    assert window._closed and not window.editor.ai.service.thread.isRunning()
    print(json.dumps(result))
    (output / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
