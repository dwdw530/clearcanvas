from pathlib import Path
from threading import Event

from PySide6.QtCore import QObject, QSize, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QStyle,
    QTabBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from clearcanvas.ai_controller import AiController
from clearcanvas.icons import icon
from clearcanvas.repair import (
    BatchResult,
    RepairResult,
    auto_repair_document,
    batch_export,
    export_result,
    parse_pages,
    repair_document,
    source_stamp,
)


class RepairController(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.result = None
        self.busy = False
        self.dirty = False
        self.stamp = None
        self.cancelled = Event()
        bar = QToolBar("水印处理", window)
        bar.setMovable(False)
        bar.setIconSize(QSize(20, 20))
        window.addToolBarBreak()
        window.addToolBar(bar)
        self.auto_button = QPushButton(icon("scan", "#ffffff"), "一键去水印")
        self.auto_button.setObjectName("primaryButton")
        self.auto_button.setToolTip("自动识别当前图片或当前页的水印并生成预览")
        self.auto_button.clicked.connect(self.start_auto)
        bar.addWidget(self.auto_button)
        bar.addSeparator()
        self.select = bar.addAction(icon("scan"), "框选水印")
        self.select.setToolTip("框选水印；关闭后可拖动画布")
        self.select.setCheckable(True)
        self.select.setChecked(True)
        self.select.toggled.connect(self._selection_mode)
        bar.widgetForAction(self.select).setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.undo = bar.addAction(icon("chevron-left"), "撤销选区")
        self.undo.setShortcut("Ctrl+Z")
        self.undo.triggered.connect(window.canvas.undo_region)
        self.clear = bar.addAction(icon("x"), "清空选区与结果")
        self.clear.triggered.connect(window.canvas.clear_regions)
        bar.addSeparator()
        self.repair_button = QPushButton("修复选区")
        self.repair_button.clicked.connect(self.start)
        bar.addWidget(self.repair_button)
        bar.addSeparator()
        self.tabs = QTabBar()
        self.tabs.addTab("原图")
        self.tabs.addTab("结果")
        self.tabs.currentChanged.connect(self.compare)
        bar.addWidget(self.tabs)
        bar.addSeparator()
        self.export_button = QPushButton(
            window.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton), "导出结果"
        )
        self.export_button.clicked.connect(self.export)
        bar.addWidget(self.export_button)
        self.batch_button = QPushButton("批量导出")
        self.batch_button.clicked.connect(self.batch)
        bar.addWidget(self.batch_button)
        self.cancel_button = QPushButton(icon("x"), "取消")
        self.cancel_button.clicked.connect(self.cancel)
        self.cancel_action = bar.addWidget(self.cancel_button)
        self.range_panel = QWidget()
        range_layout = QVBoxLayout(self.range_panel)
        range_layout.setContentsMargins(0, 0, 0, 12)
        range_layout.addWidget(QLabel("导出处理范围"))
        self.scope = QComboBox()
        self.scope.addItems(["当前页", "全部页", "指定页码"])
        range_layout.addWidget(self.scope)
        self.page_range = QLineEdit()
        self.page_range.setPlaceholderText("1,3-5")
        self.page_range.setAccessibleName("导出页码")
        self.page_range.hide()
        self.scope.currentIndexChanged.connect(lambda i: self.page_range.setVisible(i == 2))
        range_layout.addWidget(self.page_range)
        window.details.layout().insertWidget(0, self.range_panel)
        window.service.progress.connect(self._progress)
        window.canvas.selection_changed.connect(self.selection_changed)
        self.ai = AiController(self)
        self.refresh()

    def cancel(self):
        self.cancelled.set()
        self.ai.cancel()

    def _progress(self, done, total):
        if self.busy:
            self.window.progress.setRange(0, total)
            self.window.progress.setValue(done)
            self.window.status.setText(f"正在处理 {done}/{total}")

    def page_text(self):
        if self.scope.currentIndex() == 1:
            return "all"
        if self.scope.currentIndex() == 2:
            return self.page_range.text()
        return str(self.window._preview.page_index + 1)

    def begin(self, status):
        self.busy = True
        self.cancelled.clear()
        self.window.progress.setRange(0, 0)
        self.window.progress.show()
        self.window.status.setText(status)

    def _selection_mode(self, enabled):
        self.window.canvas.selection_mode = enabled
        self.window.canvas.viewport().setCursor(
            Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.OpenHandCursor
        )

    def selection_changed(self):
        if self.result is not None:
            self.window.canvas.display(self.window._preview, False, True)
            self.result.image.close()
            self.result = None
            self.dirty = False
        self.tabs.setCurrentIndex(0)
        self.window.badge.setText("原件")
        self.refresh()

    def reset(self):
        if self.result is not None:
            self.result.image.close()
        self.result = None
        self.dirty = False
        self.stamp = None
        self.tabs.setCurrentIndex(0)
        self.refresh()

    def note_preview(self):
        self.stamp = source_stamp(self.window._preview.info.path)
        self.ai.note_preview()
        self.refresh()

    def can_leave(self):
        if self.busy:
            return False
        return (
            not self.dirty
            or QMessageBox.question(
                self.window,
                "尚未导出",
                "修复结果尚未导出，确定放弃吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            == QMessageBox.StandardButton.Yes
        )

    def start_auto(self):
        preview = self.window._preview
        if self.busy or preview is None:
            return
        if self.ai.cloud:
            self.ai.start(True)
            return
        self.begin("正在自动识别水印")
        self.window._token = self.window.service.submit(
            auto_repair_document, preview, self.stamp, self.cancelled.is_set
        )
        self.window._update_controls()

    def start(self):
        preview = self.window._preview
        if self.busy or preview is None or not self.window.canvas.regions:
            return
        if self.ai.cloud:
            self.ai.start(False)
            return
        if (
            preview.info.kind == "PDF"
            and QMessageBox.warning(
                self.window,
                "PDF 图像修复",
                "当前页将转为图像，文字将不能复制，链接和批注会移除；其他页面保留。是否继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        self.begin("正在去水印")
        self.window._token = self.window.service.submit(
            repair_document,
            preview,
            list(self.window.canvas.regions),
            self.stamp,
        )
        self.window._update_controls()

    def finish(self, value):
        self.busy = False
        self.window.progress.hide()
        if isinstance(value, BatchResult):
            self.window.status.setText(
                f"批量导出: {len(value.outputs)} 个成功，{len(value.errors)} 个未完成"
            )
            message = f"成功 {len(value.outputs)} 个，未完成 {len(value.errors)} 个。"
            if value.cancelled:
                message += "\n任务已取消，已完成文件保留。"
            if value.errors:
                message += "\n" + "\n".join(value.errors[:8])
            if value.notes:
                message += "\n" + "\n".join(value.notes[:8])
            QMessageBox.information(self.window, "批量处理结果", message)
        elif isinstance(value, RepairResult):
            if self.cancelled.is_set():
                value.image.close()
                self.window.status.setText("已取消")
                self.window._update_controls()
                return
            if self.result is not None:
                self.result.image.close()
            if value.automatic:
                self.window.canvas.regions.clear()
            self.result = value
            self.dirty = True
            self.tabs.setCurrentIndex(1)
            self.compare(1)
            self.window.status.setText(value.message)
        else:
            self.dirty = False
            self.window.status.setText(f"已导出: {value}")
            self.window.status.setToolTip(str(value))
            if self.result is not None and self.result.skipped_pages:
                pages = self.result.skipped_pages
                summary = ", ".join(map(str, pages[:20])) + (" ..." if len(pages) > 20 else "")
                QMessageBox.information(
                    self.window,
                    "导出完成",
                    f"已导出。第 {summary} 页未可靠识别到水印，保留原页未修改。",
                )
        self.window._update_controls()

    def fail(self, error):
        self.busy = False
        self.window.progress.hide()
        self.window.status.setText(str(error))
        if error.code != "cancelled":
            QMessageBox.warning(self.window, "处理未完成", str(error))
        self.window._update_controls()

    def compare(self, index):
        if self.result is not None:
            preview = self.result.preview if index == 1 else self.result.original
            self.window.canvas.display(preview, False, True)
            self.window.canvas.show_regions(index == 0)
            self.window.badge.setText(
                ("云端 AI 结果" if self.result.backend == "cloud_ai" else "结果")
                if index == 1 else "原件"
            )

    def export(self):
        if self.result is None or self.busy:
            return
        source = self.result.original.info.path
        is_pdf = self.result.original.info.kind == "PDF"
        suffix = ".pdf" if is_pdf else ".png"
        name, selected_filter = QFileDialog.getSaveFileName(
            self.window,
            "导出结果",
            str(source.with_name(source.stem + "_净界" + suffix)),
            "PDF (*.pdf)" if is_pdf else "PNG (*.png);;JPEG (*.jpg *.jpeg)",
        )
        if not name:
            return
        destination = Path(name)
        if not destination.suffix:
            destination = destination.with_suffix(
                ".jpg" if selected_filter.startswith("JPEG") else suffix
            )
        pages = None
        if is_pdf:
            try:
                text = self.page_text()
                pages = (
                    list(range(self.result.original.info.page_count))
                    if text == "all"
                    else parse_pages(text, self.result.original.info.page_count)
                )
            except Exception as error:
                QMessageBox.warning(self.window, "页码无效", str(error))
                return
            needs_warning = pages != [self.result.original.page_index] or (
                self.result.automatic and not self.result.structured
            )
            message = (
                f"将逐页识别 {len(pages)} 页。独立水印删除保留文字层；"
                "需要图像修复的页面会失去文字层、链接和批注。"
                "未可靠识别的页面保留不变。继续导出吗？"
                if self.result.automatic
                else f"将修复 {len(pages)} 页；这些页面会转为图像，文字、链接和批注不再保留。"
                "继续导出吗？"
            )
            if (
                needs_warning
                and QMessageBox.warning(
                    self.window,
                    "PDF 导出确认",
                    message,
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                != QMessageBox.StandardButton.Yes
            ):
                return
        self.begin("正在导出")
        self.window._token = self.window.service.submit(
            export_result,
            self.result,
            destination,
            destination.exists(),
            pages,
            self.cancelled.is_set,
            self.window.service.progress.emit,
        )
        self.window._update_controls()

    def batch(self):
        if self.result is None or self.busy:
            return
        if self.result.backend != "local":
            QMessageBox.information(self.window, "暂不支持", "首版云端 AI 暂不支持批量。")
            return
        directory = QFileDialog.getExistingDirectory(self.window, "选择批量输出目录")
        if not directory:
            return
        paths = [
            Path(item.data(Qt.ItemDataRole.UserRole + 1)) for item in self.window._items.values()
        ]
        message = (
            f"将分别识别列表中的 {len(paths)} 个文件，不复用选区。"
            "无法可靠识别的文件跳过；PDF 未识别页面保留。"
            "需要图像修复的 PDF 页面将失去文字层、链接和批注。原件不变。继续吗？"
            if self.result.automatic
            else f"将当前选区应用到列表中的 {len(paths)} 个文件。"
            "尺寸或类型不匹配的文件会跳过，原件不变。PDF 的处理页会转为图像。继续吗？"
        )
        if (
            QMessageBox.question(
                self.window,
                "批量去水印",
                message,
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        self.begin("正在批量导出")
        self.window._token = self.window.service.submit(
            batch_export,
            self.result,
            paths,
            Path(directory),
            "all"
            if self.result.automatic and self.result.original.info.kind != "PDF"
            else self.page_text(),
            self.cancelled.is_set,
            self.window.service.progress.emit,
        )
        self.window._update_controls()

    def refresh(self):
        ready = (
            self.window._preview is not None
            and self.window.stack.currentWidget() is self.window.canvas
        )
        selected = bool(self.window.canvas.regions)
        self.select.setEnabled(ready and not self.busy)
        self.undo.setEnabled(selected and not self.busy)
        self.clear.setEnabled((selected or self.result is not None) and not self.busy)
        self.auto_button.setEnabled(ready and not self.busy)
        self.repair_button.setEnabled(ready and selected and not self.busy)
        self.tabs.setEnabled(self.result is not None and not self.busy)
        self.export_button.setEnabled(self.result is not None and not self.busy)
        self.batch_button.setEnabled(self.result is not None and not self.busy)
        self.cancel_action.setVisible(self.busy)
        self.range_panel.setVisible(ready and self.window._preview.info.kind == "PDF")
        self.range_panel.setEnabled(not self.busy)
        self.window.canvas.setEnabled(not self.busy)
        self.window.files.setEnabled(not self.busy)
        self.window.open_action.setEnabled(not self.busy)
        if hasattr(self, "ai"):
            self.ai.refresh()
        if self.busy:
            for action in (
                self.window.remove_action,
                self.window.previous_action,
                self.window.next_action,
            ):
                action.setEnabled(False)
            self.window.page_spin.setEnabled(False)
