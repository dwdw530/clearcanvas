import os
import shutil
from pathlib import Path

from PIL import Image
from PySide6.QtGui import QFontMetrics

from clearcanvas.icons import icon


def test_window_screenshots(qtbot, window, sample_pdf, sample_image):
    assert QFontMetrics(window.font()).inFont("净")
    assert QFontMetrics(window.font()).inFont("A")
    root = Path(__file__).resolve().parents[1] / "artifacts" / "qa"
    root.mkdir(parents=True, exist_ok=True)
    scale = os.environ.get("QT_SCALE_FACTOR", "1")

    def capture(name):
        qtbot.wait(100)
        destination = root / f"{name}-{scale}x.png"
        assert window.grab().save(str(destination))
        with Image.open(destination) as image:
            assert image.width >= window.width()
            assert image.convert("L").getextrema()[0] < 100

    window.resize(1360, 880)
    capture("empty")
    window.add_files([sample_pdf, sample_image])
    qtbot.waitUntil(lambda: window._preview is not None, timeout=10_000)
    capture("desktop-pdf")
    assert window.canvas.viewport().grab().toImage().pixelColor(0, 0).name() == "#e9eded"
    window.resize(1000, 680)
    capture("compact-pdf")
    assert window.toolbar.geometry().right() <= window.width()
    window.files.setCurrentRow(1)
    qtbot.waitUntil(
        lambda: window._preview is not None and window._preview.info.kind == "PNG",
        timeout=10_000,
    )
    capture("compact-image")
    window.resize(1360, 880)
    capture("desktop-image")
    long_name = sample_image.with_name("较长文件名与中文路径显示检查_" * 8 + ".png")
    shutil.copyfile(sample_image, long_name)
    window.add_files([long_name])
    qtbot.waitUntil(
        lambda: window._preview is not None and window._preview.info.path == long_name,
        timeout=10_000,
    )
    window.resize(1000, 680)
    capture("long-filename")
    assert window.width() == 1000
    shutil.copyfile(sample_pdf, root / "sample-document.pdf")
    shutil.copyfile(sample_image, root / "sample-image.png")
    for name in ("folder-open", "scan", "zoom-in", "zoom-out", "panel-left", "image"):
        pixels = icon(name).pixmap(24, 24).toImage()
        assert any(pixels.pixelColor(x, y).alpha() > 0 for x in range(24) for y in range(24))
