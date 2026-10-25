"""Native settings and explicit, cancel-by-default upload/review dialogs."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from clearcanvas.ai_service import CANCEL_NOTICE
from clearcanvas.ai_settings import AiConfig, CredentialError, normalize_root, validate_key
from clearcanvas.documents import DocumentError


def plain_label(text=""):
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    label.setMinimumWidth(0)
    return label


def confirm(parent, title, text):
    box = QMessageBox(
        QMessageBox.Icon.Warning,
        title,
        text,
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        parent,
    )
    box.setTextFormat(Qt.TextFormat.PlainText)
    box.setDefaultButton(QMessageBox.StandardButton.Cancel)
    box.setEscapeButton(QMessageBox.StandardButton.Cancel)
    return box.exec() == QMessageBox.StandardButton.Yes


def image_label(image, title):
    panel = QWidget()
    layout = QVBoxLayout(panel)
    layout.addWidget(plain_label(title))
    rgba = image.convert("RGBA")
    qimage = QImage(rgba.tobytes(), rgba.width, rgba.height, QImage.Format.Format_RGBA8888).copy()
    rgba.close()
    label = QLabel()
    label.setAccessibleName(title)
    pixmap = QPixmap.fromImage(qimage)
    thumbnail = pixmap.scaled(
        340, 230, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
    )
    label.setPixmap(thumbnail)
    scroll = QScrollArea()
    scroll.setWidget(label)
    scroll.setMinimumSize(180, 170)
    scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(scroll, 1)
    zoom = QCheckBox("原始比例（可滚动检查）")

    def set_zoom(full):
        label.setPixmap(pixmap if full else thumbnail)
        label.adjustSize()

    zoom.toggled.connect(set_zoom)
    layout.addWidget(zoom)
    return panel


def review_dialog(parent, title, text, images):
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    layout = QVBoxLayout(dialog)
    layout.addWidget(plain_label(text))
    row = QHBoxLayout()
    for image, caption in images:
        row.addWidget(image_label(image, caption))
    layout.addLayout(row)
    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
    )
    buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确认并继续")
    buttons.button(QDialogButtonBox.StandardButton.Ok).setAutoDefault(False)
    buttons.button(QDialogButtonBox.StandardButton.Cancel).setDefault(True)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    dialog.resize(780, 490)
    return dialog.exec() == QDialog.DialogCode.Accepted


class AiSettingsDialog(QDialog):
    def __init__(self, store, controller, parent=None):
        super().__init__(parent)
        self.store, self.controller = store, controller
        self.testing = False
        self.deleted = False
        self.receipt = None
        self.setWindowTitle("AI 接口设置（实验性）")
        layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        form = QFormLayout(body)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.root = QLineEdit(store.config.root)
        self.root.setMaxLength(2048)
        form.addRow("服务地址 *", self.root)
        self.endpoint = plain_label()
        form.addRow("实际接口", self.endpoint)
        self.key = QLineEdit()
        self.key.setEchoMode(QLineEdit.EchoMode.Password)
        self.key.setMaxLength(2048)
        key_row = QHBoxLayout()
        key_row.addWidget(self.key)
        self.show_key = QCheckBox("显示")
        self.show_key.toggled.connect(
            lambda visible: self.key.setEchoMode(
                QLineEdit.EchoMode.Normal if visible else QLineEdit.EchoMode.Password
            )
        )
        key_row.addWidget(self.show_key)
        key_panel = QWidget()
        key_panel.setLayout(key_row)
        key_caption = QLabel("API Key *")
        key_caption.setBuddy(self.key)
        self.key.setAccessibleName("API Key")
        form.addRow(key_caption, key_panel)
        self.remember = QCheckBox("安全记住 Key（Windows 凭据管理器）")
        self.remember.setChecked(store.remember)
        form.addRow(self.remember)
        self.delete_key = QPushButton("删除已保存的 Key")
        self.delete_key.clicked.connect(self._delete)
        form.addRow(self.delete_key)
        self.model = QLineEdit(store.config.model)
        self.model.setMaxLength(128)
        form.addRow("模型名称 *", self.model)
        form.addRow("请求预设", plain_label("1536×1024 / high / 单图候选"))
        self.advanced = QGroupBox("高级选项")
        advanced = QFormLayout(self.advanced)
        self.mask = QCheckBox("提交蒙版（默认关闭）")
        self.mask.setChecked(store.config.send_mask)
        advanced.addRow(self.mask)
        advanced.addRow(plain_label("仅在中转文档或接口测试确认支持时启用，不会自动探测。"))
        self.timeout = QSpinBox()
        self.timeout.setRange(60, 600)
        self.timeout.setSuffix(" 秒")
        self.timeout.setValue(store.config.timeout)
        advanced.addRow("总超时（连接 15 秒）", self.timeout)
        form.addRow(self.advanced)
        self.error = plain_label(store.warning)
        self.error.setObjectName("aiError")
        form.addRow(self.error)
        self.status = plain_label("尚未验证；真实接口和去水印画质需分别验收。")
        form.addRow(self.status)
        form.addRow(plain_label("设置和本地处理不联网。取消只能停止本机等待，费用不保证撤销。"))
        scroll.setWidget(body)
        layout.addWidget(scroll)
        self.fields = body
        self.test_button = QPushButton("测试图片接口（可能计费）")
        self.test_button.setAutoDefault(False)
        self.delete_key.setAutoDefault(False)
        self.test_button.clicked.connect(self._test)
        layout.addWidget(self.test_button)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Save).setText("保存设置")
        self.buttons.button(QDialogButtonBox.StandardButton.Save).setDefault(True)
        self.buttons.accepted.connect(self._save)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.root.textChanged.connect(self._address_changed)
        for signal in (
            self.key.textChanged,
            self.model.textChanged,
            self.mask.toggled,
            self.timeout.valueChanged,
        ):
            signal.connect(self._invalidate)
        self.root.setAccessibleName("服务地址")
        self.model.setAccessibleName("模型名称")
        self.timeout.setAccessibleName("总超时")
        self.key.textChanged.connect(self._key_changed)
        self._address_changed()
        if store.verified:
            self.receipt = (store.config, self.key.text())
            self.status.setText("图片编辑接口已验证；去水印效果未验收。")
        self.resize(620, 630)
        self.key.setFocus() if not self.key.text() else self.root.setFocus()

    def _key_changed(self, text):
        if text:
            self.deleted = False

    def _invalidate(self, *_):
        self.receipt = None
        self.status.setText("尚未验证；真实效果待验证。")

    def _address_changed(self, *_):
        self._invalidate()
        self.deleted = False
        self.key.clear()  # Never carry a key from the old root to a newly typed root.
        try:
            root = normalize_root(self.root.text())
            self.endpoint.setText(root + "/images/edits")
            self.key.setText(self.store.key_for(root))
        except DocumentError as error:
            self.endpoint.setText(str(error))

    def draft(self):
        config = AiConfig(
            root=self.root.text(),
            model=self.model.text(),
            send_mask=self.mask.isChecked(),
            timeout=self.timeout.value(),
        ).validated()
        if not self.deleted:
            validate_key(self.key.text())
        return config, self.key.text()

    def _field_error(self, error):
        self.error.setText(str(error))
        {"address": self.root, "model": self.model, "key": self.key, "timeout": self.timeout}.get(
            error.code, self.root
        ).setFocus()

    def _delete(self):
        if confirm(self, "删除凭据", "保存设置时删除此服务已保存的 Key；取消对话框不会删除。"):
            self.deleted = True
            self.key.clear()
            self.remember.setChecked(False)

    def _save(self):
        if self.testing or self.controller.editor.busy:
            return
        try:
            config, key = self.draft()
            verified = self.receipt == (config, key)
            try:
                self.store.save(config, key, self.remember.isChecked(), self.deleted, verified)
            except CredentialError:
                if not confirm(
                    self,
                    "凭据保存失败",
                    "不能安全保存或删除凭据。仅本次使用？"
                    "旧凭据可能仍在系统中，请稍后在凭据管理器检查。",
                ):
                    return
                self.store.save(config, key, False, self.deleted, verified, session_only=True)
            self.controller.refresh()
            self.accept()
        except DocumentError as error:
            self._field_error(error)

    def _test(self):
        if self.testing or self.controller.editor.busy:
            return
        try:
            config, key = self.draft()
            validate_key(key)
        except DocumentError as error:
            self._field_error(error)
            return
        if not confirm(
            self,
            "测试图片接口（可能计费）",
            f"向 {config.domain} 提交程序生成的测试图，不读取当前文件。\n"
            f"模型：{config.model}；一次请求，可能计费。\n"
            "单图响应协议及默认数量尚待验证；本次测试不代表去水印画质通过。",
        ):
            return
        self.testing = True
        self.fields.setEnabled(False)
        self.test_button.setEnabled(False)
        self.buttons.button(QDialogButtonBox.StandardButton.Save).setEnabled(False)
        self.status.setText("正在验证图片编辑接口…")
        self.controller.start_test(config, key, self._tested)

    def _tested(self, error):
        self.testing = False
        self.fields.setEnabled(True)
        self.test_button.setEnabled(True)
        self.buttons.button(QDialogButtonBox.StandardButton.Save).setEnabled(True)
        if error:
            self.status.setText(str(error))
            self.receipt = None
        else:
            self.receipt = self.draft()
            self.status.setText("图片编辑接口验证通过（单张、尺寸及解码）；去水印效果未验收。")

    def reject(self):
        if self.testing:
            self.controller.cancel()
            self.status.setText("正在取消。" + CANCEL_NOTICE)
            return
        self.key.clear()
        self.receipt = None
        super().reject()
