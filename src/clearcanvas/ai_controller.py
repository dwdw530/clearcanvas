"""Serial local preparation + independent network + serial validation, with stale guards."""

from __future__ import annotations

from PIL import Image
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QComboBox, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget

from clearcanvas.ai_repair import (
    AiPlan,
    AiValidated,
    compose_result,
    decode_response,
    encode_plan,
    prepare_plan,
    test_plan,
    validate_response,
)
from clearcanvas.ai_service import CANCEL_NOTICE, AiRepairService, AiRequest
from clearcanvas.ai_settings import AiSettings, validate_key
from clearcanvas.ai_settings_dialog import (
    AiSettingsDialog,
    plain_label,
    review_dialog,
)
from clearcanvas.documents import DocumentError
from clearcanvas.repair import RepairResult


def _local(operation, *args):
    try:
        return operation(*args)
    except DocumentError:
        raise
    except Exception:
        raise DocumentError("AI 本地准备或校验失败；原件和上次结果已保留。", "ai_local") from None


def _test_response(raw, cancelled):
    if cancelled():
        raise DocumentError("已取消。", "cancelled")
    with decode_response(raw):
        return True


class AiController(QObject):
    def __init__(self, editor, *, store=None, service=None):
        super().__init__(editor)
        self.editor = editor
        self.window = editor.window
        self.store = store if store is not None else AiSettings()
        self.service = service if service is not None else AiRepairService(self)
        self.service.completed.connect(self._network_done)
        self.service.stage.connect(self._stage)
        self.active = False
        self.local_token = None
        self.network_token = None
        self.plan = None
        self.validated = None
        self.key = ""
        self.config = None
        self.version = None
        self.callback = None
        self.step = ""
        self.panel = QWidget()
        layout = QVBoxLayout(self.panel)
        layout.setContentsMargins(0, 0, 0, 12)
        label = QLabel("修复方式")
        layout.addWidget(label)
        self.mode = QComboBox()
        self.mode.addItems(["本地处理", "云端 AI（实验性）"])
        self.mode.setAccessibleName("修复方式")
        label.setBuddy(self.mode)
        layout.addWidget(self.mode)
        self.notice = plain_label()
        layout.addWidget(self.notice)
        self.settings_button = QPushButton("AI 接口设置…")
        self.settings_button.clicked.connect(self.open_settings)
        layout.addWidget(self.settings_button)
        self.result_info = plain_label()
        layout.addWidget(self.result_info)
        self.window.details.layout().insertWidget(0, self.panel)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.window.ai_settings_action.triggered.connect(self.open_settings)
        self.refresh()

    @property
    def cloud(self):
        return self.mode.currentIndex() == 1

    def _mode_changed(self, *_):
        preview = self.window._preview
        if self.cloud and preview is not None and preview.info.kind == "PDF":
            self.mode.setCurrentIndex(0)
            self.window.status.setText("当前版本 PDF 仅支持本地处理，不会上传。")
        self.refresh()

    def note_preview(self):
        if self.window._preview.info.kind == "PDF":
            self.mode.setCurrentIndex(0)
            self.window.status.setText("当前版本 PDF 仅支持本地处理，不会上传。")
        self.refresh()

    def refresh(self):
        busy = self.editor.busy
        self.mode.setEnabled(not busy)
        self.settings_button.setEnabled(not busy)
        self.window.ai_settings_action.setEnabled(not busy)
        self.notice.setText(
            f"会上传处理图像，可能产生费用。\n服务：{self.store.config.domain}\n"
            + (
                "接口已验证，真实效果待验收。"
                if self.store.verified
                else "实验性：接口及效果尚未验证。"
            )
            if self.cloud
            else "本地处理不联网。"
        )
        self.window.mode_status.setText(self.store.config.domain if self.cloud else "本地")
        result = self.editor.result
        is_ai = result is not None and result.backend == "cloud_ai"
        self.result_info.setText(
            f"云端 AI 结果\n模型：{result.ai_metadata.get('model', '')}\n"
            f"服务：{result.ai_metadata.get('domain', '')}\n"
            "PNG 蒙版外不变；JPEG 重编码可能改变全图像素。"
            if is_ai
            else ""
        )
        self.editor.tabs.setTabText(1, "AI 结果" if is_ai else "结果")
        self.editor.batch_button.setEnabled(result is not None and not busy and not is_ai)
        self.editor.batch_button.setToolTip("首版云端 AI 暂不支持批量。" if is_ai else "批量导出")

    def open_settings(self):
        if self.editor.busy:
            return
        dialog = AiSettingsDialog(self.store, self, self.window)
        dialog.exec()
        dialog.key.clear()
        dialog.receipt = None
        dialog.deleteLater()
        self.refresh()

    def _begin(self, config, key, callback=None):
        if self.editor.busy or self.active:
            raise DocumentError("已有任务进行中。")
        self.config = config.validated()
        self.key = validate_key(key)
        self.version = self.store.version
        self.callback = callback
        self.active = True
        self.editor.begin("本地准备云端请求；尚未上传")
        self.window._update_controls()

    def _submit(self, step, operation, *args):
        self.step = step
        self.local_token = self.window.service.submit(_local, operation, *args)

    def start(self, automatic):
        if self.editor.busy or self.window._preview is None:
            return
        if self.window._preview.info.kind == "PDF":
            self.note_preview()
            return
        try:
            self._begin(self.store.config, self.store.key_for(self.store.config.root))
            self._submit(
                "prepare",
                prepare_plan,
                self.window._preview,
                self.editor.stamp,
                list(self.window.canvas.regions),
                automatic,
                self.editor.cancelled.is_set,
            )
        except DocumentError as error:
            if self.active:
                self._end(error)
            else:
                QMessageBox.warning(self.window, "请配置 AI 接口", str(error))
                self.open_settings()

    def start_test(self, config, key, callback):
        try:
            self._begin(config, key, callback)
            self._submit("prepare", test_plan, self.editor.cancelled.is_set)
        except DocumentError as error:
            if self.active:
                self._end(error)
            else:
                callback(error)

    def confirm_plan(self, plan):
        pixels = plan.pixels.copy()
        pixels[plan.mask != 0, :3] = (240, 90, 60)
        with Image.fromarray(pixels) as overlay:
            text = (
                f"上传至：{self.config.domain}\n模型：{self.config.model}；一次请求，可能计费。\n"
                "请求画布 1536×1024；只在显示的蒙版内修改，矩形内所有内容均可能重建。\n"
                "裁剪仍可能包含脸部及敏感背景；不能保证恢复原始五官。远端保留政策由服务方决定。"
            )
            if plan.downsampled:
                text += "\n上传区域将降采样，修复区可能损失细节；最终文件仍按原尺寸输出。"
            if not self.store.verified:
                text += "\n尚未验证单图接口；建议先在设置中测试生成图。"
            return review_dialog(
                self.window,
                "确认云端上传（可能计费）",
                text,
                [(plan.canvas, "实际上传画布"), (overlay, "原图修改蒙版")],
            )

    def confirm_alignment(self):
        return review_dialog(
            self.window,
            "需核对对齐",
            "无法可靠自动判断构图对应。请比较上传图与完整返回图的五官、边缘与构图。"
            "确认前不生成可导出结果；无法确定请选择取消。",
            [(self.plan.canvas, "上传图"), (self.validated.image, "完整 AI 返回图")],
        )

    def loaded(self, token, value):
        if token != self.local_token:
            return False
        self.local_token = None
        if not self._current():
            self._dispose(value)
            self._end(DocumentError("已取消或任务已过期。", "cancelled"))
            return True
        try:
            if self.step == "prepare":
                self.plan = value
                if self.callback is None and not self.confirm_plan(value):
                    self._end(DocumentError("已取消上传，未发出请求。", "cancelled"))
                    return True
                if not self._current():
                    self._end(DocumentError("已取消。", "cancelled"))
                    return True
                self._submit(
                    "encode",
                    encode_plan,
                    value,
                    self.config.send_mask,
                    self.editor.cancelled.is_set,
                )
            elif self.step == "encode":
                request = AiRequest(self.config, self.key, *value)
                self.key = ""
                self.network_token = self.service.submit(request)
                self.step = "network"
            elif self.step == "test_validate":
                self._end(None)
            elif self.step == "validate":
                self.validated = value
                confirmed = False
                if value.alignment == "uncertain":
                    confirmed = self.confirm_alignment()
                    if not confirmed:
                        self._end(DocumentError("未确认对齐，已放弃本次返回图。", "cancelled"))
                        return True
                self.window.status.setText("本地合成")
                self._submit(
                    "compose",
                    compose_result,
                    self.plan,
                    value,
                    self.config,
                    confirmed,
                    self.editor.cancelled.is_set,
                )
            elif self.step == "compose":
                self._release()
                self.editor.finish(value)
        except DocumentError as error:
            self._end(error)
        return True

    def failed(self, token, error):
        if token != self.local_token:
            return False
        self.local_token = None
        self._end(error)
        return True

    def _current(self):
        return (
            self.active
            and not self.editor.cancelled.is_set()
            and self.store.version == self.version
            and (
                self.callback is not None
                or self.plan is None
                or self.window._preview is self.plan.original
            )
        )

    def _stage(self, token, text):
        if token == self.network_token and self._current():
            self.window.status.setText(text)

    def _network_done(self, token, data, error):
        if token != self.network_token:
            return
        self.network_token = None
        if not self._current():
            self._end(DocumentError("已取消。" + CANCEL_NOTICE, "cancelled"))
        elif error:
            self._end(error)
        else:
            self.window.status.setText("校验结果")
            if self.callback:
                self._submit("test_validate", _test_response, data, self.editor.cancelled.is_set)
            else:
                self._submit(
                    "validate", validate_response, self.plan, data, self.editor.cancelled.is_set
                )

    @staticmethod
    def _dispose(value):
        if isinstance(value, AiPlan):
            value.close()
        elif isinstance(value, (AiValidated, RepairResult)):
            value.image.close()

    def _release(self):
        if self.plan:
            self.plan.close()
        if self.validated:
            self.validated.image.close()
        self.plan = self.validated = None
        self.key = ""
        self.config = None
        self.version = None
        self.active = False
        self.network_token = self.local_token = None
        self.step = ""

    def _end(self, error):
        callback, self.callback = self.callback, None
        self._release()
        if callback:
            self.editor.busy = False
            self.window.progress.hide()
            self.window._update_controls()
            self.window.status.setText(
                str(error) if error else "图片编辑接口验证通过；画质未验收。"
            )
            callback(error)
        elif error:
            self.editor.fail(error)

    def cancel(self):
        self.editor.cancelled.set()
        if self.active:
            self.window.status.setText("正在取消。" + CANCEL_NOTICE)
            self.service.cancel()

    def close(self):
        self.service.close()
        self.store.close()
