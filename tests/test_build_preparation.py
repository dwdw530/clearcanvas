import hashlib
import importlib.util
import io
import zipfile
from pathlib import Path

import pytest


@pytest.fixture
def prepare_module():
    path = Path(__file__).resolve().parents[1] / "tools/prepare_gmic.py"
    spec = importlib.util.spec_from_file_location("prepare_gmic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verified_cached_runtime_is_not_overwritten_or_downloaded(
    prepare_module, monkeypatch, tmp_path,
):
    module = prepare_module
    executable = b"verified executable fixture"
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        files = {"gmic.exe": executable, "COPYING": b"license", "README": b"readme"}
        for name, data in files.items():
            archive.writestr(f"gmic-{module.GMIC_VERSION}-cli-win64/{name}", data)
    payload = stream.getvalue()
    monkeypatch.setattr(module, "GMIC_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(module, "GMIC_EXE_SHA256", hashlib.sha256(executable).hexdigest())
    cache = tmp_path / "downloads" / f"gmic-{module.GMIC_VERSION}-win64.zip"
    cache.parent.mkdir()
    cache.write_bytes(payload)

    def download(*args, **kwargs):
        pytest.fail("The verified local cache must work offline")

    monkeypatch.setattr(module, "urlopen", download)
    runtime = module.prepare_gmic(tmp_path)
    before = {path.name: path.stat().st_mtime_ns for path in runtime.iterdir()}
    write_bytes = Path.write_bytes

    def locked(path, data):
        if path.parent == runtime:
            pytest.fail("Verified in-use runtime files must not be overwritten")
        return write_bytes(path, data)

    monkeypatch.setattr(Path, "write_bytes", locked)
    assert module.prepare_gmic(tmp_path) == runtime
    assert before == {path.name: path.stat().st_mtime_ns for path in runtime.iterdir()}


def test_invalid_download_is_not_extracted(prepare_module, monkeypatch, tmp_path):
    monkeypatch.setattr(prepare_module, "urlopen", lambda *args, **kwargs: io.BytesIO(b"invalid"))
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        prepare_module.prepare_gmic(tmp_path)
    assert not (tmp_path / "gmic").exists()
