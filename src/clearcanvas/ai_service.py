"""Single-request QtNetwork transport; all network objects belong to one event-loop thread."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from PySide6.QtCore import QObject, Qt, QThread, QTimer, QUrl, Signal, Slot
from PySide6.QtNetwork import QHttpMultiPart, QHttpPart, QNetworkAccessManager, QNetworkRequest

from clearcanvas.ai_settings import AiConfig, validate_key
from clearcanvas.documents import DocumentError

MAX_RESPONSE_BYTES = 64 * 1024 * 1024
PROMPT = (
    "删除图中水印并自然填补被覆盖区域。保持人物身份、姿势、构图、鼻孔、鼻翼、嘴唇、"
    "肤色、头发和背景不变；不要美化、换脸、重构五官或添加文字。"
    "保留非水印正文，不推测或补写被遮挡的原文。"
)
CANCEL_NOTICE = "取消只能停止本机等待，服务端可能仍在生成，已发生的费用不保证撤销。"


@dataclass(frozen=True)
class AiRequest:
    config: AiConfig
    key: str = field(repr=False)
    image: bytes = field(repr=False)
    mask: bytes | None = field(default=None, repr=False)


def multipart(request):
    parts = QHttpMultiPart(QHttpMultiPart.ContentType.FormDataType)
    fields = {
        "model": request.config.model,
        "prompt": PROMPT,
        "size": request.config.size,
        "quality": request.config.quality,
        "response_format": "b64_json",
    }
    for name, value in fields.items():
        part = QHttpPart()
        part.setHeader(
            QNetworkRequest.KnownHeaders.ContentDispositionHeader, f'form-data; name="{name}"'
        )
        part.setBody(value.encode("utf-8"))
        parts.append(part)
    for name, data in (("image", request.image), ("mask", request.mask)):
        if data is None:
            continue
        part = QHttpPart()
        filename = "input.png" if name == "image" else "mask.png"
        part.setHeader(
            QNetworkRequest.KnownHeaders.ContentDispositionHeader,
            f'form-data; name="{name}"; filename="{filename}"',
        )
        part.setHeader(QNetworkRequest.KnownHeaders.ContentTypeHeader, "image/png")
        part.setBody(data)
        parts.append(part)
    return parts


def http_error(status, raw=b""):
    # Recognize only bounded, known machine codes; never show arbitrary server text.
    if len(raw) <= 65536:
        try:
            body = json.loads(raw)
            error = body.get("error") if isinstance(body, dict) else None
            code = error.get("code") if isinstance(error, dict) else None
            if code in {"content_policy_violation", "moderation_blocked"}:
                return DocumentError(
                    "服务按内容策略拒绝本次编辑；不会改写指令重试。本次可能已计费。", "policy"
                )
        except (ValueError, TypeError, RecursionError):
            pass
    messages = {
        400: "参数被拒绝，请核对模型、预设或蒙版协议。不会删参数重试。",
        401: "鉴权失败，请检查 API Key。",
        403: "请求被拒绝，请检查权限、额度或服务限制。",
        404: "图片编辑接口或模型不存在，请核对设置。",
        429: "服务限流或额度受限，请在服务方核实。不会自动重试。",
    }
    return DocumentError(
        messages.get(status, f"图片接口返回 HTTP {status}，未取得有效结果。") + " 本次可能已计费。",
        "http",
    )


class _NetworkWorker(QObject):
    completed = Signal(int, object, object)
    stage = Signal(int, str)
    stopped = Signal()

    def __init__(self, manager_factory, response_limit, connect_ms):
        super().__init__()
        self.factory = manager_factory
        self.limit = response_limit
        self.connect_ms = connect_ms
        self.manager = None
        self.reply = None
        self.token = None
        self.buffer = bytearray()
        self.total_timer = None
        self.connect_timer = None

    @Slot(int, object)
    def start(self, token, request):
        if self.reply is not None:
            self.completed.emit(token, None, DocumentError("已有图片请求进行中。"))
            return
        self.token = token
        try:
            config = request.config.validated()
            validate_key(request.key)
            if config.send_mask != (request.mask is not None):
                raise DocumentError("蒙版与请求配置不一致。")
            if self.manager is None:
                self.manager = self.factory(self)
            wire = QNetworkRequest(QUrl(config.endpoint))
            wire.setAttribute(
                QNetworkRequest.Attribute.RedirectPolicyAttribute,
                QNetworkRequest.RedirectPolicy.ManualRedirectPolicy,
            )
            wire.setRawHeader(b"Authorization", b"Bearer " + request.key.encode("ascii"))
            wire.setRawHeader(b"Accept", b"application/json")
            parts = multipart(request)
            reply = self.manager.post(wire, parts)
            parts.setParent(reply)
            self.reply = reply
            self.buffer.clear()
            reply.setReadBufferSize(64 * 1024)
            self.total_timer = QTimer(self)
            self.total_timer.setSingleShot(True)
            self.total_timer.timeout.connect(self._timeout)
            self.connect_timer = QTimer(self)
            self.connect_timer.setSingleShot(True)
            self.connect_timer.timeout.connect(self._timeout)
            reply.readyRead.connect(self._read)
            reply.finished.connect(self._finished)
            reply.sslErrors.connect(self._tls_error)
            reply.encrypted.connect(self._connected)
            reply.uploadProgress.connect(self._upload)
            reply.metaDataChanged.connect(self._headers)
            self.stage.emit(token, "上传中")
            self.total_timer.start(config.timeout * 1000)
            self.connect_timer.start(self.connect_ms)
        except Exception as error:
            safe = (
                error
                if isinstance(error, DocumentError)
                else DocumentError("无法建立安全图片请求。")
            )
            self._end(None, safe)

    def _connected(self):
        if self.connect_timer:
            self.connect_timer.stop()

    def _upload(self, sent, total):
        if self.reply is not None and sent > 0:
            self._connected()
            if sent == total:
                self.stage.emit(self.token, "AI 处理中")

    def _headers(self):
        if self.reply is None:
            return
        self._connected()
        length = self.reply.header(QNetworkRequest.KnownHeaders.ContentLengthHeader)
        try:
            oversized = length is not None and int(length) > self.limit
        except (ValueError, TypeError):
            oversized = True
        if oversized:
            self._end(None, DocumentError("接口响应超过安全预算。", "limit"))

    def _read(self):
        if self.reply is None:
            return
        self.buffer.extend(bytes(self.reply.read(self.limit - len(self.buffer) + 1)))
        if len(self.buffer) > self.limit:
            self._end(None, DocumentError("接口响应超过 64 MiB 安全预算。", "limit"))

    def _finished(self):
        if self.reply is None:
            return
        self._read()
        if self.reply is None:
            return
        status = self.reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute)
        content_type = str(self.reply.header(QNetworkRequest.KnownHeaders.ContentTypeHeader) or "")
        if status is not None and 300 <= int(status) < 400:
            self._end(None, DocumentError("拒绝接口重定向，未向新地址转发凭据。", "redirect"))
        elif status is not None and not 200 <= int(status) < 300:
            self._end(None, http_error(int(status), self.buffer))
        elif self.reply.error() != self.reply.NetworkError.NoError or status is None:
            self._end(
                None, DocumentError("连接中断，未取得有效结果；可能已计费，不自动重试。", "network")
            )
        elif content_type.split(";", 1)[0].strip().lower() != "application/json":
            self._end(None, DocumentError("接口未返回 JSON 图片数据。", "response"))
        else:
            self._end(bytes(self.buffer), None)

    def _tls_error(self, *_):
        self._end(None, DocumentError("TLS 证书验证失败，已停止连接。", "tls"))

    def _timeout(self):
        self._end(None, DocumentError("请求超时，可能已计费；不会自动重试。", "timeout"))

    @Slot(int)
    def cancel(self, token):
        if token == self.token:
            self._end(None, DocumentError("已取消。" + CANCEL_NOTICE, "cancelled"))

    def _end(self, data, error):
        token, self.token = self.token, None
        reply, self.reply = self.reply, None
        for timer in (self.total_timer, self.connect_timer):
            if timer is not None:
                timer.stop()
                timer.deleteLater()
        self.total_timer = self.connect_timer = None
        self.buffer.clear()
        if reply is not None:
            # Disconnect first: abort may synchronously emit finished.
            for signal in (
                reply.readyRead,
                reply.finished,
                reply.sslErrors,
                reply.encrypted,
                reply.uploadProgress,
                reply.metaDataChanged,
            ):
                signal.disconnect()
            if not reply.isFinished():
                reply.abort()
            reply.deleteLater()
        if token is not None:
            self.completed.emit(token, data, error)

    @Slot()
    def stop(self):
        self._end(None, DocumentError("已取消。", "cancelled"))
        if self.manager is not None:
            self.manager.deleteLater()
            self.manager = None
        self.stopped.emit()


class AiRepairService(QObject):
    completed = Signal(int, object, object)
    stage = Signal(int, str)
    _start = Signal(int, object)
    _cancel = Signal(int)
    _stop = Signal()

    def __init__(
        self,
        parent=None,
        *,
        manager_factory=QNetworkAccessManager,
        response_limit=MAX_RESPONSE_BYTES,
        connect_ms=15000,
    ):
        super().__init__(parent)
        self.thread = QThread(self)
        self.worker = _NetworkWorker(manager_factory, response_limit, connect_ms)
        self.worker.moveToThread(self.thread)
        self._start.connect(self.worker.start)
        self._cancel.connect(self.worker.cancel)
        self._stop.connect(self.worker.stop)
        self.worker.completed.connect(self._completed)
        self.worker.stage.connect(self.stage)
        self.worker.stopped.connect(self.thread.quit, Qt.ConnectionType.DirectConnection)
        self.thread.finished.connect(self.worker.deleteLater)
        self._sequence = 0
        self.active = None
        self.closed = False
        self.thread.start()

    def submit(self, request):
        if self.closed or self.active is not None:
            raise DocumentError("图片服务已关闭或已有请求进行中。")
        request.config.validated()
        validate_key(request.key)
        self._sequence += 1
        self.active = self._sequence
        self._start.emit(self.active, request)
        return self.active

    def cancel(self):
        if self.active is not None:
            self._cancel.emit(self.active)

    @Slot(int, object, object)
    def _completed(self, token, data, error):
        if token != self.active or self.closed:
            return
        self.active = None
        self.completed.emit(token, data, error)

    def close(self):
        if not self.closed:
            self.closed = True
            self.active = None
            self._stop.emit()
        if not self.thread.wait(2000):
            raise DocumentError("网络线程尚未停止，请稍后再退出。")
