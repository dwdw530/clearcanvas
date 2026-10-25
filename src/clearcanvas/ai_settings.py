"""Local AI preferences. Secrets live only in memory or Windows Credential Manager."""

from __future__ import annotations

import ctypes
import hashlib
import os
import re
from ctypes import wintypes
from dataclasses import asdict, dataclass, replace
from urllib.parse import urlsplit
from uuid import uuid4

from PySide6.QtCore import QSettings

from clearcanvas.documents import DocumentError


def normalize_root(value: str) -> str:
    try:
        if not value or any(c.isspace() or ord(c) < 32 for c in value) or "\\" in value:
            raise ValueError
        url = urlsplit(value)
        if (
            url.scheme != "https"
            or not url.hostname
            or url.username is not None
            or url.password is not None
            or "?" in value
            or "#" in value
            or url.path not in ("", "/", "/v1", "/v1/")
        ):
            raise ValueError
        host = url.hostname.encode("idna").decode("ascii").lower()
        # DNS names / IPv4 only: no ambiguous user-info, escaped hosts or IPv6 literals.
        if not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", host):
            raise ValueError
        if any(
            not label or len(label) > 63 or label.startswith("-") or label.endswith("-")
            for label in host.split(".")
        ):
            raise ValueError
        port = url.port
        if port is not None and not 1 <= port <= 65535:
            raise ValueError
        return f"https://{host}" + (f":{port}" if port and port != 443 else "") + "/v1"
    except (ValueError, UnicodeError):
        raise DocumentError(
            "仅支持 HTTPS 服务根或 /v1 根；不能含账户、查询串或其他路径。", "address"
        ) from None


@dataclass(frozen=True)
class AiConfig:
    root: str = "https://code1.newcli.com/v1"
    model: str = "gpt-image-2"
    size: str = "1536x1024"
    quality: str = "high"
    send_mask: bool = False
    timeout: int = 300

    @property
    def endpoint(self):
        return self.root + "/images/edits"

    @property
    def domain(self):
        return urlsplit(self.root).netloc

    def validated(self):
        root = normalize_root(self.root)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}", self.model):
            raise DocumentError("模型名称必填，只能包含英文、数字和 . _ : / -。", "model")
        if self.size != "1536x1024" or self.quality != "high":
            raise DocumentError("尚未验证其他请求预设。", "preset")
        if type(self.timeout) is not int or not 60 <= self.timeout <= 600:
            raise DocumentError("总超时应为 60–600 秒。", "timeout")
        if type(self.send_mask) is not bool:
            raise DocumentError("蒙版选项无效。", "mask")
        return replace(self, root=root)


def validate_key(key):
    if not isinstance(key, str) or not re.fullmatch(r"[!-~]{1,2048}", key):
        raise DocumentError("请填写有效 API Key（不能含空白或换行）。", "key")
    return key


class CredentialError(DocumentError):
    def __init__(self):
        super().__init__("无法访问 Windows 凭据管理器；不会明文保存 Key。", "credential")


class _Credential(ctypes.Structure):
    _fields_ = [
        ("Flags", wintypes.DWORD),
        ("Type", wintypes.DWORD),
        ("TargetName", wintypes.LPWSTR),
        ("Comment", wintypes.LPWSTR),
        ("LastWritten", wintypes.FILETIME),
        ("CredentialBlobSize", wintypes.DWORD),
        ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
        ("Persist", wintypes.DWORD),
        ("AttributeCount", wintypes.DWORD),
        ("Attributes", ctypes.c_void_p),
        ("TargetAlias", wintypes.LPWSTR),
        ("UserName", wintypes.LPWSTR),
    ]


class WindowsCredentials:
    def __init__(self):
        self.api = None
        if os.name == "nt":
            self.api = ctypes.WinDLL("Advapi32", use_last_error=True)
            self.api.CredWriteW.argtypes = [ctypes.POINTER(_Credential), wintypes.DWORD]
            self.api.CredReadW.argtypes = [
                wintypes.LPCWSTR,
                wintypes.DWORD,
                wintypes.DWORD,
                ctypes.POINTER(ctypes.POINTER(_Credential)),
            ]
            self.api.CredDeleteW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
            for name in ("CredWriteW", "CredReadW", "CredDeleteW"):
                getattr(self.api, name).restype = wintypes.BOOL
            self.api.CredFree.argtypes = [ctypes.c_void_p]
            self.api.CredFree.restype = None

    @staticmethod
    def target(root):
        digest = hashlib.sha256(normalize_root(root).encode()).hexdigest()
        return "ClearCanvas/AI/" + digest

    def read(self, root):
        if self.api is None:
            return ""
        result = ctypes.POINTER(_Credential)()
        if not self.api.CredReadW(self.target(root), 1, 0, ctypes.byref(result)):
            if ctypes.get_last_error() == 1168:  # ERROR_NOT_FOUND
                return ""
            raise CredentialError()
        try:
            data = result.contents
            if data.CredentialBlobSize > 4096:
                raise CredentialError()
            return ctypes.string_at(data.CredentialBlob, data.CredentialBlobSize).decode(
                "utf-16-le"
            )
        except UnicodeError:
            raise CredentialError() from None
        finally:
            self.api.CredFree(result)

    def write(self, root, key):
        if self.api is None:
            raise CredentialError()
        raw = validate_key(key).encode("utf-16-le")
        buffer = (ctypes.c_ubyte * len(raw)).from_buffer_copy(raw)
        cred = _Credential(
            Type=1,
            TargetName=self.target(root),
            UserName="ClearCanvas",
            CredentialBlobSize=len(raw),
            CredentialBlob=buffer,
            Persist=2,
        )
        try:
            if not self.api.CredWriteW(ctypes.byref(cred), 0):
                raise CredentialError()
        finally:
            ctypes.memset(buffer, 0, len(raw))

    def delete(self, root):
        if self.api is not None and not self.api.CredDeleteW(self.target(root), 1, 0):
            if ctypes.get_last_error() != 1168:
                raise CredentialError()


class AiSettings:
    def __init__(self, settings=None, credentials=None):
        self.settings = settings if settings is not None else QSettings("ClearCanvas", "AI")
        self.credentials = credentials if credentials is not None else WindowsCredentials()
        self._keys = {}
        self.warning = ""
        defaults = AiConfig()
        try:
            values = {
                name: self.settings.value(name, value, type=type(value))
                for name, value in asdict(defaults).items()
            }
            self.config = AiConfig(**values).validated()
        except (DocumentError, TypeError, ValueError):
            self.config = defaults
            self.warning = "已保存的 AI 设置无效，已恢复默认值。"
        self.version = uuid4().hex
        self.verified = False  # Do not trust old receipts after a process/key-store change.
        self.remember = self.settings.value("remember", False, type=bool)

    def key_for(self, root):
        root = normalize_root(root)
        if root in self._keys:
            return self._keys[root]
        if not self.settings.value(self._remember_key(root), False, type=bool):
            return ""
        return self.credentials.read(root)

    @staticmethod
    def _remember_key(root):
        return "remembered/" + hashlib.sha256(root.encode()).hexdigest()

    def save(
        self, config, key, remember=False, delete=False, verified=False, *, session_only=False
    ):
        config = config.validated()
        if key:
            validate_key(key)
        elif not delete:
            validate_key(key)
        # Credential operations precede preference writes. Cancel never calls this method.
        if session_only:
            remember = False
        elif remember and key and not delete:
            self.credentials.write(config.root, key)
        else:
            self.credentials.delete(config.root)
        self._keys[config.root] = "" if delete else key
        for name, value in asdict(config).items():
            self.settings.setValue(name, value)
        self.settings.setValue("remember", bool(remember and not delete))
        self.settings.setValue(self._remember_key(config.root), bool(remember and not delete))
        self.settings.setValue("verified", bool(verified and key and not delete))
        self.settings.sync()
        if self.settings.status() != QSettings.Status.NoError:
            raise DocumentError("普通设置保存失败，请检查系统权限；Key 未写入普通配置。")
        self.config = config
        self.remember = bool(remember and not delete)
        self.version = uuid4().hex
        self.verified = bool(verified and key and not delete)

    def close(self):
        self._keys.clear()
