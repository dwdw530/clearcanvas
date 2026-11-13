"""Original-coordinate repair-area painting; never makes network requests."""

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPixmap, QTransform, qRgba
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from clearcanvas.ai_repair import check_repair_mask
from clearcanvas.documents import DocumentError


class AreaCanvas(QGraphicsView):
    changed = Signal()

    def __init__(self, pixels, mask, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setAccessibleName("修复范围画布")
        self.mask = mask.copy()
        self.mode = "补选"
        self.brush_size = 24
        self.history = []
        self.history_limit = max(1, min(10, 32 * 1024 * 1024 // self.mask.nbytes))
        self.last_point = None
        self.fitted = True
        height, width = mask.shape
        self.bounds = QRectF(0, 0, width, height)
        with Image.fromarray(pixels) as image:
            image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
            self.preview_size = image.size
            raw = image.tobytes()
            qimage = QImage(raw, image.width, image.height, QImage.Format.Format_RGBA8888)
            base = self.scene().addPixmap(QPixmap.fromImage(qimage))
        self.mapping = QTransform.fromScale(
            width / self.preview_size[0], height / self.preview_size[1]
        )
        base.setTransform(self.mapping)
        self.overlay = self.scene().addPixmap(QPixmap())
        self.overlay.setTransform(self.mapping)
        self.setSceneRect(self.bounds)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.set_mode("补选")
        self.redraw()

    def redraw(self):
        small = cv2.resize(self.mask, self.preview_size, interpolation=cv2.INTER_NEAREST)
        image = QImage(
            small.data, small.shape[1], small.shape[0], small.strides[0],
            QImage.Format.Format_Indexed8,
        )
        image.setColorTable([qRgba(0, 0, 0, 0)] * 255 + [qRgba(240, 90, 60, 160)])
        self.overlay.setPixmap(QPixmap.fromImage(image))

    def set_mode(self, mode):
        self.mode = mode
        self.setDragMode(
            QGraphicsView.DragMode.ScrollHandDrag if mode == "拖动"
            else QGraphicsView.DragMode.NoDrag
        )
        self.viewport().setCursor(
            Qt.CursorShape.OpenHandCursor if mode == "拖动" else Qt.CursorShape.CrossCursor
        )

    def checkpoint(self):
        self.history.append(self.mask.copy())
        if len(self.history) > self.history_limit:
            self.history.pop(0)

    def undo(self):
        if self.history:
            self.mask = self.history.pop()
            self.redraw()
            self.changed.emit()

    def clear_area(self):
        self.checkpoint()
        self.mask.fill(0)
        self.redraw()
        self.changed.emit()

    def _point(self, event):
        point = self.mapToScene(event.position().toPoint())
        return (
            max(0, min(self.mask.shape[1] - 1, round(point.x()))),
            max(0, min(self.mask.shape[0] - 1, round(point.y()))),
        )

    def _paint(self, point):
        color = 0 if self.mode == "擦除" else 255
        cv2.line(self.mask, self.last_point, point, color, self.brush_size)
        cv2.circle(self.mask, point, max(1, self.brush_size // 2), color, -1)
        self.last_point = point
        self.redraw()

    def mousePressEvent(self, event):
        if self.mode != "拖动" and event.button() == Qt.MouseButton.LeftButton:
            if self.bounds.contains(self.mapToScene(event.position().toPoint())):
                self.checkpoint()
                self.last_point = self._point(event)
                self._paint(self.last_point)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.last_point is not None:
            self._paint(self._point(event))
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.last_point is not None:
            self._paint(self._point(event))
            self.last_point = None
            self.changed.emit()
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def fit_area(self):
        self.fitted = True
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.fitInView(self.bounds, Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.fitted:
            self.fit_area()

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.fitted = False
            self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
            zoom = self.transform().m11() * factor
            if 0.02 <= zoom <= 8:
                self.scale(factor, factor)
            event.accept()
        else:
            super().wheelEvent(event)


class AiAreaDialog(QDialog):
    def __init__(self, plan, mask, parent=None, *, received=False):
        super().__init__(parent)
        self.plan, self.received = plan, received
        self.selection = None
        self.setWindowTitle(
            "调整修复范围（不联网）" if received else "第 1 步：检查修复范围（不联网）"
        )
        layout = QVBoxLayout(self)
        note = QLabel(
            "橙色是允许修改的区域，不是处理效果。用补选覆盖漏掉的水印，用擦除保护不想改的内容。\n"
            "Ctrl + 滚轮缩放；选择“拖动”查看细节。这里只调整范围，不会调用 API。"
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.canvas = AreaCanvas(plan.pixels, mask)
        row = QHBoxLayout()
        mode = QComboBox()
        mode.addItems(["补选", "擦除", "拖动"])
        mode.setAccessibleName("范围编辑工具")
        mode.currentTextChanged.connect(self.canvas.set_mode)
        row.addWidget(mode)
        row.addWidget(QLabel("画笔直径"))
        brush = QSpinBox()
        brush.setRange(2, 256)
        brush.setValue(self.canvas.brush_size)
        brush.setSuffix(" 像素")
        brush.setAccessibleName("画笔直径")
        brush.valueChanged.connect(lambda value: setattr(self.canvas, "brush_size", value))
        row.addWidget(brush)
        for text, callback in (
            ("撤销", self.canvas.undo), ("清空范围", self.canvas.clear_area),
            ("适应窗口", self.canvas.fit_area),
        ):
            button = QPushButton(text)
            button.setAutoDefault(False)
            button.clicked.connect(callback)
            row.addWidget(button)
        layout.addLayout(row)
        layout.addWidget(self.canvas, 1)
        self.info = QLabel()
        self.info.setWordWrap(True)
        layout.addWidget(self.info)
        self.canvas.changed.connect(self.update_info)
        self.update_info()
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(
            "生成本地预览（不联网）" if received else "下一步：查看上传内容"
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setAutoDefault(False)
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        available = self.screen().availableGeometry()
        self.resize(min(960, available.width() - 32), min(720, available.height() - 48))

    def update_info(self):
        area = np.count_nonzero(self.canvas.mask)
        self.info.setText(
            f"已选 {area:,} 像素（{area / self.canvas.mask.size:.1%}）；其他区域保持原样。"
            if area else "尚未选中水印。请用补选画笔标出需要处理的位置。"
        )
        if self.received:
            x1, y1, x2, y2 = self.plan.crop
            self.info.setText(
                self.info.text() + f" 已上传范围：({x1}, {y1})–({x2}, {y2})，不能超出。"
            )

    def accept(self):
        try:
            self.selection = check_repair_mask(self.plan, self.canvas.mask, received=self.received)
        except DocumentError as error:
            self.info.setText(str(error))
            return
        super().accept()
