from __future__ import annotations

from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from clearcanvas.documents import DocumentError, Preview, PreviewCancelled, render_document


class PreviewService(QObject):
    loaded = Signal(int, object)
    failed = Signal(int, object)
    progress = Signal(int, int)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="clearcanvas-preview")
        self._token = 0
        self._future: Future[Preview] | None = None
        self._closed = False

    def request(self, path: Path, page_index: int, password: str | None) -> int:
        if self._closed:
            raise RuntimeError("Preview service is closed")
        self.invalidate()
        token = self._token
        future = self._executor.submit(
            render_document,
            path,
            page_index,
            password,
            lambda: self._closed or token != self._token,
        )
        self._future = future
        future.add_done_callback(lambda result: self._complete(token, result))
        return token

    def invalidate(self) -> None:
        self._token += 1
        if self._future is not None:
            self._future.cancel()

    def submit(self, operation, *args) -> int:
        if self._closed:
            raise RuntimeError("Preview service is closed")
        self.invalidate()
        token = self._token
        self._future = self._executor.submit(operation, *args)
        self._future.add_done_callback(lambda result: self._complete(token, result))
        return token

    def _complete(self, token: int, future: Future[Preview]) -> None:
        if self._closed or token != self._token:
            return
        try:
            preview = future.result()
        except (CancelledError, PreviewCancelled):
            return
        except Exception as exc:
            error = exc if isinstance(exc, DocumentError) else DocumentError(f"操作失败: {exc}")
            if not self._closed and token == self._token:
                self.failed.emit(token, error)
        else:
            if not self._closed and token == self._token:
                self.loaded.emit(token, preview)

    def close(self) -> None:
        self._closed = True
        self.invalidate()
        self._executor.shutdown(wait=True, cancel_futures=True)
