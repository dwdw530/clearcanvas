from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPainter, QTextOption
from PySide6.QtWidgets import (
    QFormLayout,
    QLabel,
    QPlainTextEdit,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from clearcanvas.documents import Preview


def label(text: str = "", muted: bool = False) -> QLabel:
    widget = QLabel(text)
    widget.setTextFormat(Qt.TextFormat.PlainText)
    if muted:
        widget.setObjectName("muted")
    return widget


class ElidedLabel(QLabel):
    def __init__(self, text: str = "") -> None:
        super().__init__(text)
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    def minimumSizeHint(self) -> QSize:
        return QSize(0, self.fontMetrics().height())

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        text = self.fontMetrics().elidedText(
            self.text(), Qt.TextElideMode.ElideMiddle, self.width()
        )
        painter.setPen(self.palette().windowText().color())
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignVCenter, text)


def format_bytes(size: int) -> str:
    for divisor, unit in ((1024**3, "GiB"), (1024**2, "MiB"), (1024, "KiB")):
        if size >= divisor:
            return f"{size / divisor:.1f} {unit}"
    return f"{size} B"


class DocumentDetails(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 20)
        layout.setSpacing(18)
        self.name = ElidedLabel("未选择文件")
        font = self.name.font()
        font.setBold(True)
        self.name.setFont(font)
        layout.addWidget(self.name)
        form = QFormLayout()
        form.setVerticalSpacing(18)
        form.setHorizontalSpacing(14)
        self.values = {}
        for key, title in (
            ("kind", "类型"),
            ("bytes", "大小"),
            ("pages", "页数"),
            ("size", "尺寸"),
        ):
            value = label("-")
            value.setWordWrap(True)
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            form.addRow(label(title, muted=True), value)
            self.values[key] = value
        layout.addLayout(form)
        layout.addWidget(label("位置", muted=True))
        self.path = QPlainTextEdit()
        self.path.setAccessibleName("文件完整路径")
        self.path.setReadOnly(True)
        self.path.setWordWrapMode(QTextOption.WrapMode.WrapAnywhere)
        self.path.setMaximumHeight(104)
        self.path.setFrameShape(QPlainTextEdit.Shape.NoFrame)
        self.path.setStyleSheet("background: transparent; color: #61716a;")
        layout.addWidget(self.path)
        layout.addStretch()

    def clear(self) -> None:
        self.name.setText("未选择文件")
        self.name.setToolTip("")
        self.path.clear()
        for value in self.values.values():
            value.setText("-")

    def display(self, preview: Preview) -> None:
        self.name.setText(preview.info.path.name)
        self.name.setToolTip(preview.info.path.name)
        self.path.setPlainText(str(preview.info.path))
        self.values["kind"].setText(preview.info.kind)
        self.values["bytes"].setText(format_bytes(preview.info.byte_size))
        self.values["pages"].setText(str(preview.info.page_count))
        width, height = preview.source_width, preview.source_height
        unit = preview.unit
        if unit == "pt":
            width, height, unit = width * 25.4 / 72, height * 25.4 / 72, "mm"
            size = f"{width:.1f} x {height:.1f} {unit}"
        else:
            size = f"{width:.0f} x {height:.0f} {unit}"
        self.values["size"].setText(size)
