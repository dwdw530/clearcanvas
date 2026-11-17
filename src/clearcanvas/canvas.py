from __future__ import annotations

from PySide6.QtCore import QEvent, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap, QTransform, QWheelEvent
from PySide6.QtWidgets import QApplication, QGraphicsScene, QGraphicsView, QStyle, QWidget

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
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
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
        self._press_pos = None
        self._draft = None
        self.selection_mode = True
        self.redo_regions = []
        self._space_down = False
        self._gesture = None
        self._context = None
        self._base = None
        self._paper = None
        self._pixmaps = []
        self._regions_visible = True
        self._highlight = None
        self._sync_tool()

    @property
    def has_document(self) -> bool:
        return not self._document_rect.isEmpty()

    @property
    def zoom(self) -> float:
        return self._zoom

    def clear(self) -> None:
        self.cancel_interaction()
        self._overlays = []
        self.regions.clear()
        self.redo_regions.clear()
        self._pixmaps.clear()
        self._base = self._paper = self._highlight = None
        self._context = None
        self._regions_visible = True
        self.scene().clear()
        self._document_rect = QRectF()
        self.setSceneRect(QRectF())
        self.resetTransform()
        self._fit = True
        self._zoom = 1.0

    def display(
        self, preview: Preview, reset_zoom: bool = True, preserve_regions: bool = False
    ) -> None:
        center = self.mapToScene(self.viewport().rect()).boundingRect().center()
        old_width, old_height = self._document_rect.width(), self._document_rect.height()
        point = (
            (center.x() / old_width, center.y() / old_height)
            if self.has_document else (0.5, 0.5)
        )
        zoom, fit = self._zoom, self._fit
        context = (preview.info.path, preview.page_index)
        reuse = preserve_regions and context == self._context
        keep_view = reuse and preview.logical_size == (old_width, old_height) and not reset_zoom
        self.cancel_interaction()
        if not reuse:
            self.clear()
            self._context = context
            paper = QBrush(QColor("white"))
            if preview.info.kind == "PNG":
                tile = QPixmap(16, 16)
                tile.fill(QColor("#ffffff"))
                painter = QPainter(tile)
                painter.fillRect(0, 0, 8, 8, QColor("#e6e6e6"))
                painter.fillRect(8, 8, 8, 8, QColor("#e6e6e6"))
                painter.end()
                paper = QBrush(tile)
            self._paper = self.scene().addRect(QRectF(), Qt.PenStyle.NoPen, paper)
            self._base = self.scene().addPixmap(QPixmap())
            self._base.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self._base.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        pixmap = next((p for source, p in self._pixmaps if source is preview), None)
        if pixmap is None:
            image = QImage(
                preview.rgba, preview.width, preview.height, preview.width * 4,
                QImage.Format.Format_RGBA8888,
            )
            pixmap = QPixmap.fromImage(image)
            self._pixmaps.append((preview, pixmap))
            self._pixmaps = self._pixmaps[-2:]
        width, height = preview.logical_size
        self._document_rect = QRectF(0, 0, width, height)
        self._paper.setRect(self._document_rect)
        self._base.setPixmap(pixmap)
        self._base.setTransform(QTransform.fromScale(
            width / preview.width, height / preview.height
        ))
        if not keep_view:
            self.setSceneRect(self._document_rect.adjusted(-20, -20, 20, 20))
            if reset_zoom or fit:
                self.fit_document()
            else:
                self.set_zoom(zoom)
                self.centerOn(point[0] * width, point[1] * height)
        self._draw_regions()

    def discard_result_pixmap(self, original: Preview) -> None:
        self._pixmaps = [(source, pixmap) for source, pixmap in self._pixmaps if source is original]

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
            item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            item.setVisible(self._regions_visible)
            self._overlays.append(item)

    def show_regions(self, visible: bool) -> None:
        self._regions_visible = visible
        for item in self._overlays:
            item.setVisible(visible)

    def undo_region(self) -> None:
        if self.regions:
            self.redo_regions.append(self.regions.pop())
            self._draw_regions()
            self.selection_changed.emit()

    def redo_region(self) -> None:
        if self.redo_regions:
            self.regions.append(self.redo_regions.pop())
            self._draw_regions()
            self.selection_changed.emit()

    def clear_regions(self, notify: bool = True) -> None:
        self.cancel_interaction()
        self.regions.clear()
        self.redo_regions.clear()
        self._draw_regions()
        if notify:
            self.selection_changed.emit()

    def set_selection_mode(self, enabled: bool) -> None:
        self.cancel_interaction()
        self.selection_mode = enabled
        self._sync_tool()

    def _sync_tool(self) -> None:
        dragging = self._gesture == "pan" or (
            self._gesture is None and (self._space_down or not self.selection_mode)
        )
        self.setDragMode(self.DragMode.ScrollHandDrag if dragging else self.DragMode.NoDrag)
        self.viewport().setCursor(
            Qt.CursorShape.ClosedHandCursor if self._gesture == "pan"
            else Qt.CursorShape.OpenHandCursor if dragging else Qt.CursorShape.CrossCursor
        )

    def cancel_interaction(self) -> None:
        if self._draft is not None:
            self.scene().removeItem(self._draft)
        self._start = self._draft = None
        self._press_pos = None
        self._space_down = False
        self._gesture = None
        self.setDragMode(self.DragMode.NoDrag)
        self._sync_tool()

    def event(self, event) -> bool:
        if event.type() in (
            QEvent.Type.FocusOut, QEvent.Type.WindowDeactivate, QEvent.Type.Hide,
            QEvent.Type.UngrabMouse,
        ) and hasattr(self, "_gesture"):
            self.cancel_interaction()
        return super().event(event)

    def keyPressEvent(self, event) -> None:
        if (
            event.key() == Qt.Key.Key_Space and not event.modifiers()
            and self.has_document and self.isEnabled()
        ):
            if not event.isAutoRepeat():
                self._space_down = True
                self._sync_tool()
            event.accept()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and self._space_down:
            if not event.isAutoRepeat():
                self._space_down = False
                self._sync_tool()
            event.accept()
        else:
            super().keyReleaseEvent(event)

    def mousePressEvent(self, event) -> None:
        if self.has_document and event.button() == Qt.MouseButton.LeftButton:
            self.setFocus(Qt.FocusReason.MouseFocusReason)
            self._gesture = "select" if self.selection_mode and not self._space_down else "pan"
            self._sync_tool()
            if self._gesture == "select":
                point = self.mapToScene(event.position().toPoint())
                if self._document_rect.contains(point):
                    self._start = point
                    self._press_pos = event.position().toPoint()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._start is not None:
            position = event.position().toPoint()
            if (
                self._draft is None
                and (position - self._press_pos).manhattanLength()
                >= QApplication.startDragDistance()
            ):
                pen = QPen(QColor("#c45b25"), 2)
                pen.setCosmetic(True)
                self._draft = self.scene().addRect(QRectF(), pen)
                self._draft.setZValue(20)
            if self._draft is not None:
                rect = QRectF(self._start, self.mapToScene(position)).normalized()
                self._draft.setRect(rect.intersected(self._document_rect))
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mouseReleaseEvent(event)
            return
        if self._gesture == "select":
            if self._start is not None:
                position = event.position().toPoint()
                dragged = self._draft is not None or (
                    (position - self._press_pos).manhattanLength()
                    >= QApplication.startDragDistance()
                )
                rect = QRectF(self._start, self.mapToScene(position)).normalized()
                rect = rect.intersected(self._document_rect)
                if self._draft is not None:
                    self.scene().removeItem(self._draft)
                self._start = self._draft = self._press_pos = None
                if dragged and rect.width() > 1 and rect.height() > 1:
                    width, height = self._document_rect.width(), self._document_rect.height()
                    self.regions.append((
                        rect.left() / width, rect.top() / height,
                        rect.right() / width, rect.bottom() / height,
                    ))
                    self.redo_regions.clear()
                    self._draw_regions()
                    self.selection_changed.emit()
            event.accept()
        else:
            super().mouseReleaseEvent(event)
        self._gesture = None
        self._sync_tool()

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

    def fit_regions(self) -> None:
        if not self.regions or not self.has_document or self._fitting:
            return
        width, height = self._document_rect.width(), self._document_rect.height()
        bounds = QRectF()
        for left, top, right, bottom in self.regions:
            rect = QRectF(
                left * width, top * height, (right - left) * width, (bottom - top) * height
            )
            bounds = bounds.united(rect.intersected(self._document_rect))
        if bounds.isEmpty():
            return
        self._fitting = True
        try:
            self.cancel_interaction()
            scrollbar = self.style().pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent)
            available_width = max(1, self.contentsRect().width() - scrollbar - 48)
            available_height = max(1, self.contentsRect().height() - scrollbar - 48)
            self.set_zoom(min(available_width / bounds.width(), available_height / bounds.height()))
            self.centerOn(bounds.center())
        finally:
            self._fitting = False

    def clear_highlight(self) -> None:
        if self._highlight is not None:
            self.scene().removeItem(self._highlight)
            self._highlight = None

    def show_highlight(self, highlight, visible: bool) -> None:
        if highlight is not None and visible and self._highlight is None:
            image = QImage(
                highlight.mask, highlight.width, highlight.height, highlight.width,
                QImage.Format.Format_Indexed8,
            )
            image.setColorTable([QColor(36, 100, 210, int(i * 0.55)).rgba() for i in range(256)])
            self._highlight = self.scene().addPixmap(QPixmap.fromImage(image))
            self._highlight.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self._highlight.setZValue(5)
            self._highlight.setTransform(QTransform.fromScale(
                self._document_rect.width() / highlight.width,
                self._document_rect.height() / highlight.height,
            ))
        if self._highlight is not None:
            self._highlight.setVisible(visible)

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
