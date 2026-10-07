from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap, QTransform, QWheelEvent
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView, QWidget

from clearcanvas.documents import Preview


class DocumentCanvas(QGraphicsView):
    zoom_changed = Signal(float)
    selection_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("documentCanvas")
        self.setAccessibleName("文档画布")
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setBackgroundBrush(QColor("#e9eded"))
        self.setAcceptDrops(False)
        self.viewport().setAcceptDrops(False)
        self._document_rect = QRectF()
        self._fit = True
        self._fitting = False
        self._zoom = 1.0
        self.regions = []
        self._overlays = []
        self._start = None
        self._draft = None
        self.selection_mode = True

    @property
    def has_document(self) -> bool:
        return not self._document_rect.isEmpty()

    @property
    def zoom(self) -> float:
        return self._zoom

    def clear(self) -> None:
        self._overlays = []
        self._draft = None
        self._start = None
        self.regions = []
        self.scene().clear()
        self._document_rect = QRectF()
        self.setSceneRect(QRectF())
        self.resetTransform()

    def display(
        self, preview: Preview, reset_zoom: bool = True, preserve_regions: bool = False
    ) -> None:
        regions = self.regions.copy() if preserve_regions else []
        self.clear()
        self.regions = regions
        image = QImage(
            preview.rgba,
            preview.width,
            preview.height,
            preview.width * 4,
            QImage.Format.Format_RGBA8888,
        ).copy()
        width, height = preview.logical_size
        self._document_rect = QRectF(0, 0, width, height)
        paper = QBrush(QColor("white"))
        if preview.info.kind == "PNG":
            tile = QPixmap(16, 16)
            tile.fill(QColor("#ffffff"))
            painter = QPainter(tile)
            painter.fillRect(0, 0, 8, 8, QColor("#e6e6e6"))
            painter.fillRect(8, 8, 8, 8, QColor("#e6e6e6"))
            painter.end()
            paper = QBrush(tile)
        self.scene().addRect(self._document_rect, Qt.PenStyle.NoPen, paper)
        item = self.scene().addPixmap(QPixmap.fromImage(image))
        item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        item.setTransform(QTransform.fromScale(width / preview.width, height / preview.height))
        self.setSceneRect(self._document_rect.adjusted(-20, -20, 20, 20))
        if reset_zoom or self._fit:
            self.fit_document()
        else:
            self.set_zoom(self._zoom)
        self._draw_regions()

    def _draw_regions(self) -> None:
        for item in self._overlays:
            self.scene().removeItem(item)
        self._overlays = []
        width, height = self._document_rect.width(), self._document_rect.height()
        pen = QPen(QColor("#c45b25"), 2)
        pen.setCosmetic(True)
        for left, top, right, bottom in self.regions:
            item = self.scene().addRect(
                QRectF(left * width, top * height, (right - left) * width, (bottom - top) * height),
                pen,
                QBrush(QColor(236, 136, 48, 45)),
            )
            item.setZValue(10)
            self._overlays.append(item)

    def show_regions(self, visible: bool) -> None:
        for item in self._overlays:
            item.setVisible(visible)

    def undo_region(self) -> None:
        if self.regions:
            self.regions.pop()
            self._draw_regions()
            self.selection_changed.emit()

    def clear_regions(self) -> None:
        self.regions = []
        self._draw_regions()
        self.selection_changed.emit()

    def mousePressEvent(self, event) -> None:
        point = self.mapToScene(event.position().toPoint())
        if (
            self.selection_mode
            and self.has_document
            and event.button() == Qt.MouseButton.LeftButton
        ):
            if self._document_rect.contains(point):
                self._start = point
                pen = QPen(QColor("#c45b25"), 2)
                pen.setCosmetic(True)
                self._draft = self.scene().addRect(QRectF(point, point), pen)
                self._draft.setZValue(20)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._start is not None:
            rect = QRectF(self._start, self.mapToScene(event.position().toPoint())).normalized()
            self._draft.setRect(rect.intersected(self._document_rect))
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._start is not None:
            rect = QRectF(self._start, self.mapToScene(event.position().toPoint())).normalized()
            rect = rect.intersected(self._document_rect)
            self.scene().removeItem(self._draft)
            self._start = self._draft = None
            if rect.width() > 1 and rect.height() > 1:
                width, height = self._document_rect.width(), self._document_rect.height()
                self.regions.append(
                    (
                        rect.left() / width,
                        rect.top() / height,
                        rect.right() / width,
                        rect.bottom() / height,
                    )
                )
                self._draw_regions()
                self.selection_changed.emit()
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def fit_document(self) -> None:
        if not self.has_document or self._fitting:
            return
        self._fit = True
        self._fitting = True
        try:
            # Auto scrollbars can oscillate viewport size while fitting a small image.
            self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            width = max(1, self.viewport().width() - 48)
            height = max(1, self.viewport().height() - 48)
            self._apply_zoom(
                min(width / self._document_rect.width(), height / self._document_rect.height())
            )
            self.centerOn(self._document_rect.center())
        finally:
            self._fitting = False

    def set_zoom(self, factor: float) -> None:
        if self.has_document:
            self._fit = False
            self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self._apply_zoom(max(0.05, min(8.0, factor)))

    def zoom_by(self, factor: float) -> None:
        self.set_zoom(self._zoom * factor)

    def _apply_zoom(self, factor: float) -> None:
        self._zoom = factor
        self.setTransform(QTransform.fromScale(factor, factor))
        self.zoom_changed.emit(factor)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._fit:
            self.fit_document()

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoom_by(1.2 if event.angleDelta().y() > 0 else 1 / 1.2)
            event.accept()
        else:
            super().wheelEvent(event)
