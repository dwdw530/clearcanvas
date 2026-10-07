import ast
import sys
from pathlib import Path

import pytest


def test_built_bundle_does_not_capture_foreign_dlls():
    root = Path(__file__).resolve().parents[1]
    manifest = root / "build" / "clearcanvas" / "COLLECT-00.toc"
    bundle = root / "dist" / "ClearCanvas"
    if not manifest.exists() or not (bundle / "ClearCanvas.exe").exists():
        pytest.skip("Build the Windows bundle before running the packaging audit")
    entries = ast.literal_eval(manifest.read_text(encoding="utf-8"))[0]
    trusted = [(root / ".venv").resolve(), Path(sys.base_prefix).resolve()]
    for destination, source, kind in entries:
        if kind in {"BINARY", "EXTENSION"}:
            assert any(Path(source).resolve().is_relative_to(base) for base in trusted), source
            assert not Path(destination).name.lower().startswith("api-ms-win-")
    internal = bundle / "_internal"
    assert not list(internal.rglob("icu*.dll"))
    assert not (internal / "ucrtbase.dll").exists()
    assert (internal / "VCRUNTIME140.dll").is_file()
    assert (internal / "PySide6" / "Qt6Core.dll").is_file()
    assert (internal / "clearcanvas" / "assets" / "lucide" / "LICENSE.txt").is_file()
