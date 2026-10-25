import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from PySide6.QtCore import QIODevice, QObject, QThread, QTimer, QUrl
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest

from clearcanvas.ai_service import AiRepairService, AiRequest
from clearcanvas.ai_settings import AiConfig
from clearcanvas.documents import DocumentError

QT_POST = QNetworkAccessManager.post  # Only used by the explicit loopback adapter below.
SECRET = "offline-not-a-real-key"


class Reply(QNetworkReply):
    def __init__(self, parent, spec):
        super().__init__(parent)
        self.spec = spec
        self.data = bytearray()
        self.aborted = False
        self.open(QIODevice.OpenModeFlag.ReadOnly)
        self.setAttribute(
            QNetworkRequest.Attribute.HttpStatusCodeAttribute, spec.get("status", 200)
        )
        self.setHeader(
            QNetworkRequest.KnownHeaders.ContentTypeHeader, spec.get("type", "application/json")
        )
        if "length" in spec:
            self.setHeader(QNetworkRequest.KnownHeaders.ContentLengthHeader, spec["length"])
        QTimer.singleShot(spec.get("delay", 1), self.deliver)

    def deliver(self):
        if self.aborted:
            return
        if self.spec.get("tls"):
            self.sslErrors.emit([])
            return
        if self.spec.get("hang"):
            return
        self.encrypted.emit()
        self.uploadProgress.emit(10, 10)
        self.metaDataChanged.emit()
        for chunk in self.spec.get("chunks", [b'{"data":[]}']):
            if self.aborted:
                return
            self.data.extend(chunk)
            self.readyRead.emit()
        if self.aborted:
            return
        if self.spec.get("disconnect"):
            self.setError(QNetworkReply.NetworkError.RemoteHostClosedError, SECRET)
        self.setFinished(True)
        self.finished.emit()

    def readData(self, maximum):
        result = bytes(self.data[:maximum])
        del self.data[:maximum]
        return result

    def bytesAvailable(self):
        return len(self.data) + super().bytesAvailable()

    def abort(self):
        self.aborted = True
        self.setFinished(True)
        self.finished.emit()


class Manager(QObject):
    def __init__(self, parent, calls, spec):
        super().__init__(parent)
        self.calls, self.spec = calls, spec

    def post(self, request, parts):
        self.calls.append(
            {
                "thread": QThread.currentThread(),
                "request": request,
                "parts": parts.contentType() if hasattr(parts, "contentType") else None,
            }
        )
        self.reply = Reply(self, self.spec)
        return self.reply


@pytest.fixture
def factory(qtbot):
    services = []

    def create(spec=None, **kwargs):
        calls = []

        def manager(parent):
            return Manager(parent, calls, spec or {})

        service = AiRepairService(manager_factory=manager, **kwargs)
        services.append(service)
        return service, calls

    yield create
    for service in services:
        service.close()
        assert not service.thread.isRunning()


def run(qtbot, service, request=None):
    with qtbot.waitSignal(service.completed, timeout=3000) as signal:
        service.submit(request or AiRequest(AiConfig(), SECRET, b"fake-png"))
    return signal.args


def test_worker_thread_request_security_and_single_result(qtbot, factory):
    service, calls = factory({"chunks": [b'{"da', b'ta":[]}']})
    _, data, error = run(qtbot, service)
    assert error is None
    assert data == b'{"data":[]}'
    assert len(calls) == 1
    assert calls[0]["thread"] is service.thread
    request = calls[0]["request"]
    assert request.url().toString() == AiConfig().endpoint
    assert bytes(request.rawHeader("Authorization")) == b"Bearer " + SECRET.encode()
    assert request.attribute(QNetworkRequest.Attribute.RedirectPolicyAttribute) == (
        QNetworkRequest.RedirectPolicy.ManualRedirectPolicy
    )


@pytest.mark.parametrize("status", [400, 401, 403, 404, 429, 500, 502, 503])
def test_http_errors_no_retry_or_body_leak(qtbot, factory, status):
    service, calls = factory({"status": status, "chunks": [SECRET.encode()]})
    _, data, error = run(qtbot, service)
    assert data is None and error.code == "http"
    assert SECRET not in str(error)
    qtbot.wait(20)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "spec,code",
    [
        ({"status": 302}, "redirect"),
        ({"status": 307}, "redirect"),
        ({"tls": True}, "tls"),
        ({"disconnect": True}, "network"),
        ({"type": "text/html", "chunks": [b"<html>oops</html>"]}, "response"),
        ({"length": 21}, "limit"),
        ({"chunks": [b"x" * 12, b"y" * 12]}, "limit"),
    ],
)
def test_transport_boundaries(qtbot, factory, spec, code):
    service, calls = factory(spec, response_limit=20)
    _, data, error = run(qtbot, service)
    assert data is None and error.code == code
    assert len(calls) == 1
    assert not service.worker.buffer


def test_http_policy_refusal_not_reported_as_connection_error(qtbot, factory):
    service, calls = factory(
        {
            "status": 400,
            "chunks": [b'{"error":{"code":"content_policy_violation","message":"PRIVATE"}}'],
        }
    )
    _, data, error = run(qtbot, service)
    assert data is None and error.code == "policy"
    assert "PRIVATE" not in str(error)
    assert len(calls) == 1


def test_connect_timeout(qtbot, factory):
    service, calls = factory({"hang": True}, connect_ms=30)
    _, data, error = run(qtbot, service)
    assert data is None and error.code == "timeout"
    assert len(calls) == 1


def test_total_timeout_on_worker(qtbot, factory):
    service, _ = factory({"hang": True})

    # Trigger the actual total timer in its owner thread without waiting 300 seconds.
    class TimeoutManager(Manager):
        def post(self, request, parts):
            reply = super().post(request, parts)
            QTimer.singleShot(30, lambda: self.parent().total_timer.start(1))
            return reply

    service.worker.factory = lambda parent: TimeoutManager(parent, [], {"hang": True})
    _, data, error = run(qtbot, service)
    assert data is None and error.code == "timeout"


def test_cancel_late_response_singleflight_and_close(qtbot, factory):
    service, calls = factory({"delay": 150})
    results = []
    service.completed.connect(lambda *args: results.append(args))
    token = service.submit(AiRequest(AiConfig(), SECRET, b"png"))
    with pytest.raises(DocumentError):
        service.submit(AiRequest(AiConfig(), SECRET, b"png"))
    with qtbot.waitSignal(service.completed, timeout=1000) as signal:
        service.cancel()
    assert signal.args[2].code == "cancelled"
    qtbot.wait(200)
    assert len(results) == 1
    service._completed(token, b"late", None)
    assert len(results) == 1
    assert len(calls) == 1
    service.close()
    with pytest.raises(DocumentError):
        service.submit(AiRequest(AiConfig(), SECRET, b"png"))


def test_mask_mismatch_does_not_send(qtbot, factory):
    service, calls = factory()
    _, _, error = run(
        qtbot, service, AiRequest(replace(AiConfig(), send_mask=True), SECRET, b"png")
    )
    assert error and not calls


@pytest.mark.parametrize("send_mask", [False, True])
def test_real_multipart_serialization_loopback_only(qtbot, send_mask):
    captured = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            captured.append(
                (self.path, self.headers, self.rfile.read(int(self.headers["Content-Length"])))
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"data":[]}')

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    class Loopback(QNetworkAccessManager):
        def post(self, request, parts):
            assert request.url().scheme() == "https"
            wire = QNetworkRequest(request)
            wire.setUrl(QUrl(f"http://127.0.0.1:{server.server_port}/v1/images/edits"))
            return QT_POST(self, wire, parts)

    service = AiRepairService(manager_factory=Loopback)
    try:
        config = replace(AiConfig(), send_mask=send_mask)
        _, data, error = run(
            qtbot, service, AiRequest(config, SECRET, b"IMAGE", b"MASK" if send_mask else None)
        )
        assert error is None and data == b'{"data":[]}'
        assert len(captured) == 1
        path, headers, body = captured[0]
        assert path == "/v1/images/edits"
        assert headers["Content-Type"].startswith("multipart/form-data;")
        for name in ("model", "prompt", "size", "quality", "response_format", "image"):
            assert body.count(f'name="{name}"'.encode()) == 1
        assert (b'name="mask"' in body) == send_mask
        assert b'filename="input.png"' in body
        assert b"1536x1024" in body and b"b64_json" in body and b"gpt-image-2" in body
        assert b'name="n"' not in body and b"input_fidelity" not in body
    finally:
        service.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
