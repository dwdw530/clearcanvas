from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QSignalBlocker, QSize, Qt
from PySide6.QtGui import QAction, QCloseEvent, QDragEnterEvent, QDropEvent, QKeySequence
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QSystemTrayIcon,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from clearcanvas import __version__
from clearcanvas.canvas import DocumentCanvas
from clearcanvas.documents import DocumentError, Preview, validate_path
from clearcanvas.editor import RepairController
from clearcanvas.icons import icon
from clearcanvas.preview_service import PreviewService
from clearcanvas.repair import BatchResult, RepairResult
from clearcanvas.widgets import DocumentDetails, ElidedLabel, label


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"净界 ClearCanvas {__version__}")
        self.setWindowIcon(icon("scan", "#176a54"))
        self.resize(1360, 880)
        self.setMinimumSize(1000, 680)
        self.setAcceptDrops(True)
        self.service = PreviewService(self)
        self.service.loaded.connect(self._loaded, Qt.ConnectionType.QueuedConnection)
        self.service.failed.connect(self._failed, Qt.ConnectionType.QueuedConnection)
        self._items: dict[str, QListWidgetItem] = {}
        self._passwords: dict[str, str] = {}
        self._pages: dict[str, int] = {}
        self._active_key: str | None = None
        self._token = 0
        self._preview: Preview | None = None
        self._reset_zoom = True
        self._create_actions()
        self._create_toolbar()
        self._create_workspace()
        self._create_docks()
        self._create_menus()
        self.status = ElidedLabel("就绪")
        self.statusBar().addWidget(self.status, 1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setFixedWidth(100)
        self.progress.setTextVisible(False)
        self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().addPermanentWidget(label("本地", muted=True))
        self.editor = RepairController(self)
        self._exit_requested = False
        self._closed = False
        self._close_prompt_open = False
        self.tray = QSystemTrayIcon(self.windowIcon(), self)
        self.tray.setToolTip("净界 ClearCanvas")
        self.tray_menu = QMenu(self)
        self.tray_menu.addAction("打开净界", self.restore_window)
        self.tray_menu.addSeparator()
        self.tray_menu.addAction("退出程序", self.request_exit)
        self.tray.setContextMenu(self.tray_menu)
        self.tray.activated.connect(self._tray_activated)
        self._update_controls()

    def _action(self, title: str, icon_name: str, slot, shortcut=None) -> QAction:
        action = QAction(icon(icon_name), title, self)
        action.triggered.connect(slot)
        if shortcut:
            action.setShortcut(shortcut)
            action.setToolTip(f"{title} ({action.shortcut().toString()})")
        else:
            action.setToolTip(title)
        return action

    def _create_actions(self) -> None:
        self.open_action = self._action(
            "添加文件", "folder-open", self.open_files, QKeySequence.StandardKey.Open
        )
        self.remove_action = self._action("移出列表", "trash-2", self.remove_current)
        self.zoom_in_action = self._action(
            "放大",
            "zoom-in",
            lambda: self.canvas.zoom_by(1.25),
            QKeySequence.StandardKey.ZoomIn,
        )
        self.zoom_out_action = self._action(
            "缩小",
            "zoom-out",
            lambda: self.canvas.zoom_by(0.8),
            QKeySequence.StandardKey.ZoomOut,
        )
        self.fit_action = self._action(
            "适应窗口", "scan", lambda: self.canvas.fit_document(), "Ctrl+0"
        )
        self.actual_action = QAction("原始比例", self)
        self.actual_action.setShortcut("Ctrl+1")
        self.actual_action.triggered.connect(lambda: self.canvas.set_zoom(1))
        self.previous_action = self._action(
            "上一页",
            "chevron-left",
            lambda: self.page_spin.stepDown(),
            "PgUp",
        )
        self.next_action = self._action(
            "下一页",
            "chevron-right",
            lambda: self.page_spin.stepUp(),
            "PgDown",
        )

    def _create_toolbar(self) -> None:
        toolbar = QToolBar("文档工具", self)
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(20, 20))
        self.addToolBar(toolbar)
        brand = label("净界")
        font = brand.font()
        font.setPointSize(19)
        font.setBold(True)
        brand.setFont(font)
        toolbar.addWidget(brand)
        toolbar.addWidget(label("  ClearCanvas", muted=True))
        toolbar.addSeparator()
        toolbar.addAction(self.open_action)
        toolbar.widgetForAction(self.open_action).setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        toolbar.addAction(self.remove_action)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)
        toolbar.addAction(self.zoom_out_action)
        self.zoom_combo = QComboBox()
        self.zoom_combo.setAccessibleName("缩放比例")
        self.zoom_combo.setFixedWidth(92)
        self.zoom_combo.setEditable(True)
        self.zoom_combo.lineEdit().setReadOnly(True)
        for percent in (5, 10, 25, 50, 75, 100, 125, 150, 200, 400, 800):
            self.zoom_combo.addItem(f"{percent}%", percent / 100)
        self.zoom_combo.setCurrentText("100%")
        self.zoom_combo.activated.connect(
            lambda i: self.canvas.set_zoom(self.zoom_combo.itemData(i))
        )
        toolbar.addWidget(self.zoom_combo)
        toolbar.addAction(self.zoom_in_action)
        toolbar.addAction(self.fit_action)
        toolbar.addSeparator()
        self.toolbar = toolbar

    def _create_workspace(self) -> None:
        workbench = QWidget()
        workbench.setObjectName("workbench")
        layout = QVBoxLayout(workbench)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        header = QFrame()
        header.setObjectName("documentHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 13, 18, 13)
        self.document_title = ElidedLabel("工作区")
        header_layout.addWidget(self.document_title, 1)
        self.badge = label("原件")
        self.badge.setObjectName("badge")
        header_layout.addWidget(self.badge)
        layout.addWidget(header)
        self.stack = QStackedWidget()
        self.empty = self._state_widget("files", "尚未打开文件", add_button=True)
        self.loading = self._state_widget("file-text", "正在读取")
        self.error = self._state_widget("file-text", "无法打开文件")
        self.error_message = label()
        self.error_message.setWordWrap(True)
        self.error_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.error_message.setMaximumWidth(360)
        self.error.layout().insertWidget(3, self.error_message, 0, Qt.AlignmentFlag.AlignCenter)
        self.canvas = DocumentCanvas()
        self.canvas.zoom_changed.connect(
            lambda value: self.zoom_combo.setCurrentText(f"{value * 100:.1f}%")
        )
        for widget in (self.empty, self.canvas, self.loading, self.error):
            self.stack.addWidget(widget)
        layout.addWidget(self.stack, 1)
        page_bar = QFrame()
        page_bar.setObjectName("pageBar")
        bar = QHBoxLayout(page_bar)
        bar.setContentsMargins(12, 7, 12, 7)
        bar.addStretch()
        for action in (self.previous_action,):
            button = QToolButton()
            button.setDefaultAction(action)
            button.setFixedSize(34, 34)
            bar.addWidget(button)
        self.page_spin = QSpinBox()
        self.page_spin.setAccessibleName("当前页码")
        self.page_spin.setFixedWidth(82)
        self.page_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.page_spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.page_spin.setRange(1, 1)
        self.page_spin.setKeyboardTracking(False)
        self.page_spin.valueChanged.connect(self._page_changed)
        bar.addWidget(self.page_spin)
        self.page_total = label("/ 0 页", muted=True)
        self.page_total.setMinimumWidth(70)
        bar.addWidget(self.page_total)
        next_button = QToolButton()
        next_button.setDefaultAction(self.next_action)
        next_button.setFixedSize(34, 34)
        bar.addWidget(next_button)
        bar.addStretch()
        layout.addWidget(page_bar)
        self.setCentralWidget(workbench)

    def _state_widget(self, icon_name: str, title: str, add_button: bool = False) -> QWidget:
        state = QWidget()
        layout = QVBoxLayout(state)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        layout.addStretch()
        symbol = label()
        symbol.setPixmap(icon(icon_name, "#73958a").pixmap(48, 48))
        layout.addWidget(symbol, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label(title, muted=True), 0, Qt.AlignmentFlag.AlignCenter)
        if add_button:
            button = QPushButton(icon("folder-open", "#ffffff"), "打开文件")
            button.setObjectName("primaryButton")
            button.clicked.connect(self.open_files)
            layout.addWidget(button, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addStretch()
        return state

    def _create_docks(self) -> None:
        self.files = QListWidget()
        self.files.setAccessibleName("文件列表")
        self.files.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.files.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.files.setIconSize(QSize(24, 24))
        self.files.currentItemChanged.connect(self._selection_changed)
        self.file_dock = QDockWidget("文件 · 0", self)
        self.file_dock.setWidget(self.files)
        self.file_dock.setMinimumWidth(190)
        self.file_dock.setMaximumWidth(400)
        self.file_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.file_dock)
        self.details = DocumentDetails()
        self.details_dock = QDockWidget("文件信息", self)
        self.details_dock.setWidget(self.details)
        self.details_dock.setMinimumWidth(220)
        self.details_dock.setMaximumWidth(350)
        self.details_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.details_dock)
        self.resizeDocks([self.file_dock, self.details_dock], [240, 250], Qt.Orientation.Horizontal)
        for dock, title, name in (
            (self.file_dock, "文件列表", "panel-left"),
            (self.details_dock, "文件信息", "panel-right"),
        ):
            action = dock.toggleViewAction()
            action.setText(title)
            action.setToolTip(title)
            action.setIcon(icon(name))
            self.toolbar.addAction(action)

    def _create_menus(self) -> None:
        file_menu = self.menuBar().addMenu("文件")
        file_menu.addActions([self.open_action, self.remove_action])
        file_menu.addSeparator()
        quit_action = file_menu.addAction("退出")
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        view_menu = self.menuBar().addMenu("视图")
        view_menu.addActions(
            [self.zoom_in_action, self.zoom_out_action, self.fit_action, self.actual_action]
        )
        view_menu.addSeparator()
        view_menu.addActions([self.previous_action, self.next_action])
        view_menu.addSeparator()
        view_menu.addActions(
            [self.file_dock.toggleViewAction(), self.details_dock.toggleViewAction()]
        )
        help_menu = self.menuBar().addMenu("帮助")
        about = help_menu.addAction(icon("info"), "关于净界")
        about.triggered.connect(
            lambda: QMessageBox.about(self, "关于净界", f"净界 ClearCanvas\n{__version__}")
        )

    def open_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "添加文件",
            "",
            "PDF 和图片 (*.pdf *.png *.jpg *.jpeg);;PDF (*.pdf);;图片 (*.png *.jpg *.jpeg)",
        )
        self.add_files(paths)

    def add_files(self, paths: list[str | Path]) -> None:
        first_key = None
        errors = []
        with QSignalBlocker(self.files):
            for value in paths:
                try:
                    path = validate_path(value)
                except DocumentError as exc:
                    errors.append(f"{Path(value).name}: {exc}")
                    continue
                key = os.path.normcase(str(path))
                if first_key is None:
                    first_key = key
                if key in self._items:
                    continue
                kind = path.suffix[1:].upper()
                item = QListWidgetItem(
                    icon("file-text" if kind == "PDF" else "image"), f"{path.name}\n{kind}"
                )
                item.setData(Qt.ItemDataRole.UserRole, key)
                item.setData(Qt.ItemDataRole.UserRole + 1, str(path))
                item.setToolTip(str(path))
                item.setSizeHint(QSize(180, 66))
                self._items[key] = item
                self.files.addItem(item)
        self.file_dock.setWindowTitle(f"文件 · {len(self._items)}")
        if first_key is not None:
            target = self._items[first_key]
            if self.files.currentItem() is target:
                self._selection_changed(target)
            else:
                self.files.setCurrentItem(target)
        if errors:
            message = "\n".join(errors[:8])
            if len(errors) > 8:
                message += f"\n另有 {len(errors) - 8} 个文件无法导入。"
            box = QMessageBox(QMessageBox.Icon.Warning, "部分文件未导入", message, parent=self)
            box.setTextFormat(Qt.TextFormat.PlainText)
            box.exec()

    def _selection_changed(self, item: QListWidgetItem | None, previous=None) -> None:
        if not self.editor.can_leave():
            with QSignalBlocker(self.files):
                self.files.setCurrentItem(previous)
            return
        self.editor.reset()
        self.badge.setText("原件")
        self.service.invalidate()
        self._token = -1
        self._preview = None
        self.canvas.clear()
        self.details.clear()
        self._reset_zoom = True
        self._active_key = item.data(Qt.ItemDataRole.UserRole) if item else None
        with QSignalBlocker(self.page_spin):
            self.page_spin.setRange(1, 1)
            self.page_spin.setValue(1)
        self.page_total.setText("/ 0 页")
        if item is None:
            self.document_title.setText("工作区")
            self.document_title.setToolTip("")
            self.stack.setCurrentWidget(self.empty)
            self.progress.hide()
            self.status.setText("就绪")
            self._update_controls()
            return
        path = Path(item.data(Qt.ItemDataRole.UserRole + 1))
        self.document_title.setText(path.name)
        self.document_title.setToolTip(str(path))
        self._request_page(self._pages.get(self._active_key, 0))

    def _request_page(self, page_index: int) -> None:
        if self._active_key is None:
            return
        path = Path(self._items[self._active_key].data(Qt.ItemDataRole.UserRole + 1))
        self._pages[self._active_key] = page_index
        self.stack.setCurrentWidget(self.loading)
        self.progress.show()
        self.status.setText(f"正在读取: {path.name}")
        self._token = self.service.request(path, page_index, self._passwords.get(self._active_key))
        self._update_controls()

    def _loaded(self, token: int, preview: Preview) -> None:
        if token != self._token or self._active_key is None:
            return
        if isinstance(preview, (RepairResult, BatchResult, Path)):
            self.editor.finish(preview)
            return
        self._preview = preview
        self.stack.setCurrentWidget(self.canvas)
        self.canvas.display(preview, self._reset_zoom)
        self._reset_zoom = False
        self.details.display(preview)
        with QSignalBlocker(self.page_spin):
            self.page_spin.setRange(1, preview.info.page_count)
            self.page_spin.setValue(preview.page_index + 1)
        self.page_total.setText(f"/ {preview.info.page_count} 页")
        item = self._items[self._active_key]
        item.setText(
            f"{preview.info.path.name}\n{preview.info.kind} · {preview.info.page_count} 页"
        )
        self.progress.hide()
        self.status.setText("就绪")
        self.editor.note_preview()
        self._update_controls()

    def _failed(self, token: int, error: DocumentError) -> None:
        if token != self._token or self._active_key is None:
            return
        if self.editor.busy:
            self.editor.fail(error)
            return
        self.progress.hide()
        self._preview = None
        self.canvas.clear()
        self.details.clear()
        self.error_message.setText(str(error))
        self.stack.setCurrentWidget(self.error)
        self.status.setText("未能打开文件")
        self._update_controls()
        if error.code == "password":
            key = self._active_key
            self._passwords.pop(key, None)
            password, accepted = QInputDialog.getText(
                self,
                "PDF 密码",
                str(error),
                QLineEdit.EchoMode.Password,
            )
            if accepted and self._active_key == key:
                self._passwords[key] = password
                self._request_page(self._pages.get(key, 0))
            else:
                self.status.setText("已取消打开加密 PDF")

    def _page_changed(self, value: int) -> None:
        if self._active_key is not None and self._preview is not None:
            if not self.editor.can_leave():
                with QSignalBlocker(self.page_spin):
                    self.page_spin.setValue(self._preview.page_index + 1)
                return
            self.editor.reset()
            self.badge.setText("原件")
            self._request_page(value - 1)

    def _update_controls(self) -> None:
        ready = self._preview is not None and self.stack.currentWidget() is self.canvas
        for action in (
            self.zoom_in_action,
            self.zoom_out_action,
            self.fit_action,
            self.actual_action,
        ):
            action.setEnabled(ready)
        self.zoom_combo.setEnabled(ready)
        self.remove_action.setEnabled(self._active_key is not None)
        count = self._preview.info.page_count if self._preview else 0
        page = self._pages.get(self._active_key, 0)
        self.page_spin.setEnabled(count > 1)
        self.previous_action.setEnabled(count > 1 and page > 0)
        self.next_action.setEnabled(count > 1 and page < count - 1)
        if hasattr(self, "editor"):
            self.editor.refresh()

    def remove_current(self) -> None:
        item = self.files.currentItem()
        if item is None:
            return
        if not self.editor.can_leave():
            return
        self.editor.dirty = False
        key = item.data(Qt.ItemDataRole.UserRole)
        self._passwords.pop(key, None)
        self._pages.pop(key, None)
        self._items.pop(key, None)
        self.files.takeItem(self.files.row(item))
        self.file_dock.setWindowTitle(f"文件 · {len(self._items)}")

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls() and any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        self.add_files([url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()])
        event.acceptProposedAction()

    def restore_window(self) -> None:
        self.setWindowState(self.windowState() & ~Qt.WindowState.WindowMinimized)
        self.show()
        self.raise_()
        self.activateWindow()

    def _tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self.restore_window()

    def request_exit(self) -> None:
        self.restore_window()
        self._exit_requested = True
        try:
            self.close()
        finally:
            self._exit_requested = False

    def _choose_close_action(self) -> str:
        box = QMessageBox(self)
        box.setWindowTitle("关闭净界")
        box.setIcon(QMessageBox.Icon.Question)
        box.setText("关闭窗口后，您希望净界如何运行？")
        available = QSystemTrayIcon.isSystemTrayAvailable()
        box.setInformativeText(
            "最小化后会继续处理当前任务，可点击右下角托盘图标恢复窗口。"
            if available else "系统托盘不可用，将最小化到任务栏，当前任务会继续处理。"
        )
        minimize = box.addButton(
            "最小化到托盘" if available else "最小化到任务栏",
            QMessageBox.ButtonRole.AcceptRole,
        )
        exit_button = box.addButton("退出程序", QMessageBox.ButtonRole.DestructiveRole)
        cancel = box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(minimize)
        box.setEscapeButton(cancel)
        box.exec()
        if box.clickedButton() is minimize:
            return "minimize"
        return "exit" if box.clickedButton() is exit_button else "cancel"

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._closed:
            event.accept()
            return
        event.ignore()
        if self._close_prompt_open:
            return
        self._close_prompt_open = True
        try:
            action = "exit" if self._exit_requested else self._choose_close_action()
        finally:
            self._close_prompt_open = False
        if action == "minimize":
            if QSystemTrayIcon.isSystemTrayAvailable():
                self.tray.show()
                self.hide()
            else:
                self.showMinimized()
            return
        if action != "exit":
            return
        if self.editor.busy:
            QMessageBox.information(self, "任务进行中", "请等待当前任务完成后再退出。")
            return
        if not self.editor.can_leave():
            return
        self._token = -1
        self.service.close()
        self._passwords.clear()
        self.tray.hide()
        self._closed = True
        event.accept()
