from dataclasses import replace

import pytest
from PySide6.QtCore import QSettings

from clearcanvas.ai_settings import (
    AiConfig,
    AiSettings,
    CredentialError,
    WindowsCredentials,
    normalize_root,
    validate_key,
)
from clearcanvas.documents import DocumentError

SECRET = "offline-test-key-NOT-REAL"


class MemoryCredentials:
    def __init__(self):
        self.values = {}
        self.calls = []
        self.fail = False

    def read(self, root):
        self.calls.append(("read", root))
        if self.fail:
            raise CredentialError()
        return self.values.get(root, "")

    def write(self, root, key):
        self.calls.append(("write", root))
        if self.fail:
            raise CredentialError()
        self.values[root] = key

    def delete(self, root):
        self.calls.append(("delete", root))
        if self.fail:
            raise CredentialError()
        self.values.pop(root, None)


@pytest.fixture
def store(tmp_path):
    return AiSettings(
        QSettings(str(tmp_path / "ai.ini"), QSettings.Format.IniFormat), MemoryCredentials()
    )


@pytest.mark.parametrize(
    "address,expected",
    [
        ("https://EXAMPLE.com/", "https://example.com/v1"),
        ("https://example.com/v1/", "https://example.com/v1"),
        ("https://example.com:443/v1", "https://example.com/v1"),
        ("https://example.com:8443", "https://example.com:8443/v1"),
    ],
)
def test_root_normalization(address, expected):
    assert normalize_root(address) == expected
    assert AiConfig(root=address).validated().endpoint == expected + "/images/edits"


@pytest.mark.parametrize(
    "address",
    [
        "",
        "http://example.com",
        "https://user:pass@example.com",
        "https://u@example.com",
        "https://example.com/v2",
        "https://example.com/v1/images/edits",
        "https://example.com?q=1",
        "https://example.com#x",
        "https://example.com?",
        "https://example.com#",
        "https://example.com\\evil",
        "https://example.com\n",
        "https://example.com:bad",
        "https://example.com:0",
        "https://example.com:65536",
        "https://ex%61mple.com",
        "https://example..com",
        "https://-example.com",
        "https://example.com//v1",
    ],
)
def test_reject_ambiguous_or_insecure_roots(address):
    with pytest.raises(DocumentError):
        normalize_root(address)


@pytest.mark.parametrize(
    "change",
    [
        {"model": ""},
        {"model": "x\nsecret"},
        {"model": "<b>x"},
        {"size": "auto"},
        {"quality": "medium"},
        {"timeout": 59},
        {"timeout": 601},
        {"timeout": True},
        {"send_mask": "true"},
    ],
)
def test_config_validation(change):
    with pytest.raises(DocumentError):
        replace(AiConfig(), **change).validated()


@pytest.mark.parametrize("key", ["", " a", "a b", "a\rb", "中文", "x" * 2049])
def test_key_validation_is_local_and_safe(key):
    with pytest.raises(DocumentError) as error:
        validate_key(key)
    assert key not in str(error.value) if key else True


def test_session_key_and_domain_isolation(store):
    a = AiConfig()
    b = replace(a, root="https://other.example/v1")
    store.save(a, SECRET)
    assert store.key_for(a.root) == SECRET
    assert store.key_for(b.root) == ""
    store.save(b, "other-fake-key")
    assert store.key_for(a.root) == SECRET
    assert store.key_for(b.root) == "other-fake-key"
    assert not store.credentials.values
    store.close()
    assert store.key_for(a.root) == ""
    assert SECRET not in open(store.settings.fileName(), encoding="utf-8").read()


def test_remember_delete_and_verification_invalidation(store):
    config = AiConfig()
    store.save(config, SECRET, remember=True, verified=True)
    version = store.version
    assert store.verified
    assert store.credentials.values[config.root] == SECRET
    store.save(replace(config, model="another-model"), SECRET, remember=True)
    assert store.version != version
    assert not store.verified
    store.save(config, "", delete=True)
    assert not store.credentials.values
    assert store.key_for(config.root) == ""
    assert not store.verified


def test_credentials_failure_never_silently_falls_back(store):
    store.credentials.fail = True
    with pytest.raises(CredentialError):
        store.save(AiConfig(), SECRET, remember=True)
    assert not store.settings.contains("root")
    store.save(AiConfig(), SECRET, session_only=True)
    assert store.key_for(store.config.root) == SECRET
    assert not store.remember
    assert SECRET not in open(store.settings.fileName(), encoding="utf-8").read()


def test_non_windows_session_only_and_target_isolation(monkeypatch):
    import clearcanvas.ai_settings as module

    monkeypatch.setattr(module.os, "name", "posix")
    credentials = WindowsCredentials()
    assert credentials.read(AiConfig().root) == ""
    with pytest.raises(CredentialError):
        credentials.write(AiConfig().root, SECRET)
    assert credentials.target("https://example.com/") == credentials.target(
        "https://example.com/v1"
    )
    assert credentials.target(AiConfig().root) != credentials.target("https://other.example")


def test_restart_does_not_trust_previous_receipt_or_restore_session_key(store):
    store.save(AiConfig(), SECRET, verified=True)
    next_store = AiSettings(store.settings, store.credentials)
    assert not next_store.verified
    assert next_store.key_for(next_store.config.root) == ""


def test_corrupt_settings_fail_safe(store):
    store.settings.setValue("root", "http://insecure.example")
    store.settings.setValue("send_mask", True)
    next_store = AiSettings(store.settings, store.credentials)
    assert next_store.config == AiConfig()
    assert next_store.warning


def test_saved_credentials_root_scoped_reload_and_session_fallback(store):
    config = AiConfig()
    other = replace(config, root="https://other.example/v1")
    store.save(config, SECRET, remember=True)
    store.save(other, "OTHER-FAKE-KEY", remember=True)
    reloaded = AiSettings(store.settings, store.credentials)
    assert reloaded.key_for(config.root) == SECRET
    assert reloaded.key_for(other.root) == "OTHER-FAKE-KEY"
    store.credentials.fail = True
    store.save(config, "SESSION-ONLY", session_only=True)
    reloaded = AiSettings(store.settings, store.credentials)
    # Even if deletion was unavailable and old credentials remain, no implicit reuse.
    assert reloaded.key_for(config.root) == ""


def test_windows_credential_abi_with_fake_api(monkeypatch):
    import ctypes
    import os

    import clearcanvas.ai_settings as module

    if os.name != "nt":
        pytest.skip("Windows ABI adapter test")
    raw = SECRET.encode("utf-16-le")
    blob = (ctypes.c_ubyte * len(raw)).from_buffer_copy(raw)
    value = module._Credential(CredentialBlobSize=len(raw), CredentialBlob=blob)
    calls = []

    class Function:
        def __init__(self, operation):
            self.operation = operation

        def __call__(self, *args):
            return self.operation(*args)

    class Api:
        pass

    api = Api()

    def read(target, kind, flags, output):
        assert kind == 1 and flags == 0
        pointer = ctypes.cast(output, ctypes.POINTER(ctypes.POINTER(module._Credential)))
        pointer[0] = ctypes.pointer(value)
        calls.append("read")
        return True

    def write(pointer, flags):
        cred = ctypes.cast(pointer, ctypes.POINTER(module._Credential)).contents
        assert cred.Type == 1 and cred.Persist == 2
        assert ctypes.string_at(cred.CredentialBlob, cred.CredentialBlobSize) == raw
        calls.append("write")
        return True

    api.CredReadW = Function(read)
    api.CredWriteW = Function(write)
    api.CredDeleteW = Function(lambda *args: calls.append("delete") or True)
    api.CredFree = Function(lambda *args: calls.append("free"))
    monkeypatch.setattr(ctypes, "WinDLL", lambda *a, **k: api)
    credentials = WindowsCredentials()
    credentials.write(AiConfig().root, SECRET)
    assert credentials.read(AiConfig().root) == SECRET
    credentials.delete(AiConfig().root)
    assert calls == ["write", "read", "free", "delete"]
    assert len(api.CredReadW.argtypes) == 4
