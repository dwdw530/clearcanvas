# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import importlib.util
import sys

root = Path(SPECPATH)
assets = root / "artifacts" / "build"
if not (assets / "clearcanvas.ico").is_file():
    raise SystemExit("Run: uv run python tools/prepare_build.py")

a = Analysis(
    [str(root / "packaging" / "entrypoint.py")],
    pathex=[str(root / "src")],
    binaries=[],
    datas=[
        (str(root / "src" / "clearcanvas" / "assets"), "clearcanvas/assets"),
        (str(assets / "licenses"), "licenses"),
        (str(assets / "COMPONENTS.txt"), "."),
        (str(root / "README.md"), "."),
        (str(root / "docs" / "product-design.md"), "docs"),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "pytest"],
    noarchive=False,
)

# Only collect project/runtime libraries, never DLLs from unrelated PATH entries.
# Qt uses Windows' ICU; Poppler's versioned ICU has an incompatible export table.
qt_directory = Path(importlib.util.find_spec("PySide6").origin).parent
trusted_roots = [root / ".venv", Path(sys.base_prefix)]
runtime_names = {
    "vcruntime140.dll", "vcruntime140_1.dll",
    "msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll",
}
a.binaries = [
    entry for entry in a.binaries
    if any(Path(entry[1]).resolve().is_relative_to(directory.resolve()) for directory in trusted_roots)
    and Path(entry[0]).name.lower() not in runtime_names | {"ucrtbase.dll"}
    and not Path(entry[0]).name.lower().startswith("api-ms-win-")
]
for path in qt_directory.glob("*.dll"):
    if path.name.lower() in runtime_names:
        a.binaries.append((path.name, str(path), "BINARY"))

pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ClearCanvas",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(assets / "clearcanvas.ico"),
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="ClearCanvas",
)
