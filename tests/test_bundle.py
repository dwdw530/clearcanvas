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
    trusted = [(root / ".venv").resolve(), Path(sys.base_prefix).resolve(),
               (root / "artifacts/build/gmic").resolve()]
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


def test_built_patchmatch_library_performs_inpainting(monkeypatch):
    import numpy as np
    import patchmatch._lib as bindings
    import patchmatch.patch_match as backend

    root = Path(__file__).resolve().parents[1]
    internal = root / 'dist/ClearCanvas/_internal'
    library = internal / 'patchmatch/patchmatch.dll'
    if not library.exists():
        pytest.skip('Build the PatchMatch-enabled Windows bundle first')
    license_file = internal / 'licenses/pypatchmatch/pypatchmatch-2.1.1.dist-info/licenses/LICENSE'
    assert license_file.exists()
    monkeypatch.setattr(bindings, 'find_library', lambda: library)
    monkeypatch.setattr(backend, '_lib', bindings.load_library())
    marked = np.full((80, 120, 3), (90, 130, 170), np.uint8)
    mask = np.zeros(marked.shape[:2], np.uint8)
    mask[25:50, 40:75] = 255
    marked[mask != 0] = (255, 0, 100)
    result = backend.inpaint(marked, mask, patch_size=3)
    assert np.array_equal(result[mask == 0], marked[mask == 0])
    assert np.max(np.abs(result[mask != 0].astype(int) - (90, 130, 170))) <= 1


def test_built_structure_runtime_performs_offline_inpainting(monkeypatch):
    import hashlib

    import numpy as np

    from clearcanvas import structure_inpaint as structure

    root = Path(__file__).resolve().parents[1]
    internal = root / "dist/ClearCanvas/_internal"
    executable = internal / "gmic/gmic.exe"
    if not executable.is_file():
        pytest.skip("Build the G'MIC-enabled Windows bundle first")
    assert hashlib.sha256(executable.read_bytes()).hexdigest() == structure.GMIC_EXE_SHA256
    assert (internal / "licenses/gmic/COPYING").is_file()
    assert (internal / "licenses/gmic/SOURCE.txt").is_file()
    monkeypatch.setattr(structure, "executable_path", lambda: executable)
    marked = np.full((80, 120, 3), (90, 130, 170), np.uint8)
    mask = np.zeros(marked.shape[:2], np.uint8)
    mask[25:50, 40:75] = 255
    marked[mask != 0] = (255, 0, 100)
    result = structure.restore_photo_structure(marked, mask)
    assert np.array_equal(result[mask == 0], marked[mask == 0])
    assert np.max(np.abs(result[mask != 0].astype(int) - (90, 130, 170))) <= 1
