from pathlib import Path
from threading import Event

from PySide6.QtCore import QEvent, QObject, QSize, Qt, QUrl
from PySide6.QtGui import QActionGroup, QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
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
from clearcanvas.change_highlight import compute_change_highlight
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
from clearcanvas.widgets import ElidedLabel


class RepairController(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.result = None
        self.busy = False
        self.dirty = False
        self.stamp = None
        self.cancelled = Event()
        self.temporary_original = False
        self.highlight_cache = None
        self.highlight_task = None
        self.output_path = None
        self.output_kind = ""
        self.output_count = 0
        bar = QToolBar("水印处理", window)
        bar.setMovable(False)
        bar.setIconSize(QSize(20, 20))
        window.addToolBarBreak()
        window.addToolBar(bar)
        self.auto_button = QPushButton(icon("scan", "#ffffff"), "本地一键去水印")
        self.auto_button.setObjectName("primaryButton")
        self.auto_button.setToolTip("自动识别当前图片或当前页的水印并生成预览")
        self.auto_button.clicked.connect(self.start_auto)
        bar.addWidget(self.auto_button)
        bar.addSeparator()
        self.select = bar.addAction(icon("scan"), "框选水印")
        self.select.setToolTip("拖动框选水印；实际修改选区后需重新修复，不会覆盖原文件")
        self.select.setCheckable(True)
        self.select.setChecked(True)
        self.select.toggled.connect(self._selection_mode)
        bar.widgetForAction(self.select).setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.pan = bar.addAction("拖动查看")
        self.pan.setCheckable(True)
        self.pan.setToolTip("直接拖动查看细节，不修改选区或修复结果")
        self.tools = QActionGroup(self)
        self.tools.addAction(self.select)
        self.tools.addAction(self.pan)
        self.undo = bar.addAction(icon("chevron-left"), "撤销选区")
        self.undo.setShortcut("Ctrl+Z")
        self.undo.triggered.connect(window.canvas.undo_region)
        self.undo.setShortcutContext(Qt.ShortcutContext.WidgetShortcut)
        window.canvas.addAction(self.undo)
        self.redo = bar.addAction(icon("chevron-right"), "重做选区")
        self.redo.setShortcut("Ctrl+Y")
        self.redo.setShortcutContext(Qt.ShortcutContext.WidgetShortcut)
        window.canvas.addAction(self.redo)
        self.redo.triggered.connect(window.canvas.redo_region)
        self.clear = bar.addAction(icon("x"), "清空选区与结果")
        self.clear.triggered.connect(lambda: window.canvas.clear_regions())
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
        self.scope.currentIndexChanged.connect(self._scope_changed)
        range_layout.addWidget(self.page_range)
        window.details.layout().insertWidget(0, self.range_panel)
        self.inspect_panel = QWidget()
        inspect_layout = QVBoxLayout(self.inspect_panel)
        inspect_layout.setContentsMargins(0, 0, 0, 12)
        inspect_layout.addWidget(QLabel("检查结果"))
        self.hold_original = QPushButton("按住看原图")
        self.hold_original.setToolTip("按住查看，松开恢复结果；长期检查请切换原图／结果标签")
        self.hold_original.pressed.connect(self.begin_temporary_original)
        self.hold_original.released.connect(self.end_temporary_original)
        self.hold_original.installEventFilter(self)
        inspect_layout.addWidget(self.hold_original)
        hint = QLabel("按住查看，松开恢复结果")
        hint.setWordWrap(True)
        inspect_layout.addWidget(hint)
        self.fit_regions_button = QPushButton(icon("zoom-in"), "放大到选区")
        self.fit_regions_button.clicked.connect(self.fit_regions)
        inspect_layout.addWidget(self.fit_regions_button)
        self.highlight_switch = QCheckBox("高亮实际修改")
        self.highlight_switch.toggled.connect(self.toggle_highlight)
        inspect_layout.addWidget(self.highlight_switch)
        self.highlight_info = QLabel()
        self.highlight_info.setWordWrap(True)
        inspect_layout.addWidget(self.highlight_info)
        self.processing_info = QLabel("尚未生成结果")
        self.processing_info.setWordWrap(True)
        self.processing_info.setTextFormat(Qt.TextFormat.PlainText)
        inspect_layout.addWidget(self.processing_info)
        window.details.layout().insertWidget(1, self.inspect_panel)
        self.output_panel = QWidget()
        output_layout = QVBoxLayout(self.output_panel)
        output_layout.setContentsMargins(0, 0, 0, 12)
        output_layout.addWidget(QLabel("上次成功输出"))
        self.output_title = ElidedLabel("尚无成功输出")
        output_layout.addWidget(self.output_title)
        self.output_info = QLabel()
        self.output_info.setWordWrap(True)
        output_layout.addWidget(self.output_info)
        self.open_file_button = QPushButton("打开文件")
        self.open_file_button.clicked.connect(lambda: self.open_output(False))
        output_layout.addWidget(self.open_file_button)
        self.open_folder_button = QPushButton("打开所在文件夹")
        self.open_folder_button.clicked.connect(lambda: self.open_output(True))
        output_layout.addWidget(self.open_folder_button)
        window.details.layout().insertWidget(2, self.output_panel)
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
        self.end_temporary_original()
        self.window.canvas.cancel_interaction()
        self.busy = True
        self.cancelled.clear()
        self.window.progress.setRange(0, 0)
        self.window.progress.show()
        self.window.status.setText(status)

    def _selection_mode(self, enabled):
        self.end_temporary_original()
        self.window.canvas.set_selection_mode(enabled)
        self.render_result()
        self.refresh()

    def selection_changed(self):
        self.end_temporary_original()
        self.clear_highlight()
        if self.result is not None:
            self.window.canvas.display(self.window._preview, False, True)
            self.result.image.close()
            self.result = None
            self.dirty = False
            self.window.canvas.discard_result_pixmap(self.window._preview)
        self.tabs.setCurrentIndex(0)
        self.select.setChecked(True)
        self.window.canvas.show_regions(True)
        self.window.badge.setText("原件")
        self.window.status.setText(
            "选区已更改，请重新修复；原件未变。" if self.window.canvas.regions
            else "已清空选区与结果；原件未变。"
        )
        self.refresh()

    def reset(self):
        self.end_temporary_original()
        self.clear_highlight()
        self.window.canvas.clear_regions(False)
        if self.result is not None:
            self.result.image.close()
        self.result = None
        self.dirty = False
        self.stamp = None
        self.tabs.setCurrentIndex(0)
        self.select.setChecked(True)
        self.refresh()

    def note_preview(self):
        self.stamp = source_stamp(self.window._preview.info.path)
        self.ai.note_preview()
        self.refresh()

    def can_leave(self):
        self.end_temporary_original()
        self.window.canvas.cancel_interaction()
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
            if value.outputs:
                self.record_output(value.outputs[0].parent, "batch", len(value.outputs))
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
            self.clear_highlight()
            self.window.canvas.discard_result_pixmap(self.window._preview)
            if self.result is not None:
                self.result.image.close()
            if value.automatic:
                self.window.canvas.clear_regions(False)
            self.result = value
            self.dirty = True
            self.pan.setChecked(True)
            if self.tabs.currentIndex() == 1:
                self.compare(1)
            else:
                self.tabs.setCurrentIndex(1)
            self.window.status.setText(value.message)
            self.window.details_dock.widget().verticalScrollBar().setValue(0)
        elif isinstance(value, Path):
            self.record_output(value, "file")
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
        self.end_temporary_original(render=False)
        self.render_result()
        self.refresh()

    def render_result(self):
        if self.result is None:
            return
        showing_result = self.tabs.currentIndex() == 1 and not self.temporary_original
        preview = self.result.preview if showing_result else self.result.original
        self.window.canvas.display(preview, False, True)
        self.window.canvas.show_regions(
            (not showing_result or self.select.isChecked()) and not self.temporary_original
        )
        self.window.canvas.show_highlight(
            self.highlight_cache, showing_result and self.highlight_switch.isChecked()
        )
        self.window.badge.setText(
            "原件 · 临时查看" if self.temporary_original
            else ("云端 AI 结果" if self.result.backend == "cloud_ai" else "结果")
            if showing_result else "原件"
        )

    def begin_temporary_original(self):
        if self.result is not None and self.tabs.currentIndex() == 1 and not self.busy:
            self.temporary_original = True
            self.render_result()

    def end_temporary_original(self, render=True):
        if self.temporary_original:
            self.temporary_original = False
            self.hold_original.setDown(False)
            if render:
                self.render_result()

    def eventFilter(self, watched, event):
        if watched is self.hold_original and event.type() in (
            QEvent.Type.UngrabMouse, QEvent.Type.FocusOut, QEvent.Type.Hide,
            QEvent.Type.WindowDeactivate,
        ):
            self.end_temporary_original()
        return super().eventFilter(watched, event)

    def fit_regions(self):
        if not self.busy:
            self.end_temporary_original()
            self.window.canvas.fit_regions()

    def _scope_changed(self, index):
        self.page_range.setVisible(index == 2)
        self.refresh()

    def record_output(self, path, kind, count=1):
        self.output_path, self.output_kind, self.output_count = Path(path), kind, count
        self.output_title.setText(
            f"批量输出：{count} 个文件" if kind == "batch" else Path(path).name
        )
        self.output_title.setToolTip(str(path))
        self.output_info.setText("完整 AI 返回图 · 未经合成" if kind == "ai_returned" else "")
        self.open_folder_button.setText("打开输出文件夹" if kind == "batch" else "打开所在文件夹")
        self.refresh()

    def open_output(self, folder):
        if self.output_path is None or self.busy:
            return
        self.end_temporary_original()
        target = self.output_path
        if folder and self.output_kind != "batch":
            target = target.parent
        if not (target.is_dir() if folder else target.is_file()):
            QMessageBox.information(
                self.window, "输出目标不存在", "目标已不存在，请检查文件或所在目录。"
            )
        elif not QDesktopServices.openUrl(QUrl.fromLocalFile(str(target.resolve()))):
            QMessageBox.warning(
                self.window, "未能打开", "系统未接受打开请求；此前成功输出不受影响。"
            )

    def clear_highlight(self):
        self.highlight_cache = None
        self.highlight_switch.blockSignals(True)
        self.highlight_switch.setChecked(False)
        self.highlight_switch.blockSignals(False)
        self.window.canvas.clear_highlight()

    def toggle_highlight(self, enabled):
        if self.busy or self.result is None:
            return
        self.end_temporary_original()
        if not enabled or self.highlight_cache is not None:
            self.render_result()
            self.refresh()
            return
        self.begin("正在计算实际修改区域（不联网）")
        token = self.window.service.submit(
            compute_change_highlight, self.result, self.cancelled.is_set
        )
        original = self.result.original
        self.highlight_task = (token, self.result, original.info.path, original.page_index)
        self.window._update_controls()

    def finish_highlight(self, token, value=None, error=None):
        task = self.highlight_task
        if task is None or token != task[0]:
            return False
        self.highlight_task = None
        self.busy = False
        self.window.progress.hide()
        preview = self.window._preview
        current = (
            self.result is task[1] and preview is not None
            and preview.info.path == task[2] and preview.page_index == task[3]
        )
        if error is not None or self.cancelled.is_set() or not current:
            self.clear_highlight()
            text = "高亮未生成，修复结果仍可检查与导出。"
            self.window.status.setText(text)
            if error is not None and error.code != "cancelled":
                QMessageBox.warning(self.window, "高亮未生成", str(error) + "\n" + text)
        else:
            self.highlight_cache = value
            self.render_result()
            self.window.status.setText("修改区域高亮已生成，仅用于定位。")
        self.window._update_controls()
        return True

    def export(self):
        self.end_temporary_original()
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
        self.end_temporary_original()
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
        self.redo.setEnabled(bool(self.window.canvas.redo_regions) and ready and not self.busy)
        self.hold_original.setEnabled(
            self.result is not None and self.tabs.currentIndex() == 1 and not self.busy
        )
        self.fit_regions_button.setEnabled(ready and selected and not self.busy)
        self.window.fit_regions_action.setEnabled(ready and selected and not self.busy)
        self.highlight_switch.setEnabled(self.result is not None and not self.busy)
        self.open_file_button.setVisible(self.output_kind != "batch")
        self.open_file_button.setEnabled(self.output_path is not None and not self.busy)
        self.open_folder_button.setEnabled(self.output_path is not None and not self.busy)
        result = self.result
        layout = self.window.details.layout()
        inspection_index = 0 if result is not None else 2
        if layout.indexOf(self.inspect_panel) != inspection_index:
            layout.removeWidget(self.inspect_panel)
            layout.insertWidget(inspection_index, self.inspect_panel)
        hint = (
            "重新框选将替换当前结果" if self.select.isChecked() and result is not None
            else "拖动框选水印；按住空格拖动" if self.select.isChecked()
            else "拖动查看；按住按钮对比原图" if result is not None
            else "拖动查看；点击“框选水印”编辑"
        )
        self.window.canvas_hint.setText(hint)
        self.window.canvas_hint.setToolTip(hint)
        description = "尚未生成结果"
        if result is not None:
            if result.original.info.kind == "PDF":
                description = (
                    "当前页：独立水印对象删除；保留文字层" if result.structured
                    else "当前页：图像修复；导出后文字不能复制，链接和批注不保留"
                )
                if self.scope.currentIndex() != 0:
                    description += "\n此处仅说明当前页；其他页逐页判断，未可靠识别页保留"
            elif result.backend == "cloud_ai":
                description = "当前图片：已采用的云端 AI 局部合成结果"
            else:
                description = "当前图片：本地像素修复；请检查水印和边缘"
        self.processing_info.setText(description)
        self.highlight_info.setText(
            ("修改区域 · " if self.highlight_switch.isChecked() else "") + (
                "高亮显示当前页渲染差异，不是 PDF 对象列表或文字保真证明"
                if result is not None and result.structured
                else "高亮显示导出前结果的像素变化；JPEG 重编码还可能改变其他像素"
            ) + "；低分辨率预览仅用于定位。" if result is not None else ""
        )
        self.select.setEnabled(ready and not self.busy)
        self.pan.setEnabled(ready and not self.busy)
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
