"""Generate build assets and retain the installed runtime components' licenses."""

import importlib.metadata
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from prepare_gmic import prepare_gmic
from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

from clearcanvas.structure_inpaint import GMIC_VERSION

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "build"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    app = QGuiApplication([])
    source = ROOT / "src" / "clearcanvas" / "assets" / "lucide" / "scan.svg"
    svg = source.read_text(encoding="utf-8").replace("currentColor", "#ffffff")
    renderer = QSvgRenderer(QByteArray(svg.encode()))
    pixmap = QPixmap(256, 256)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor("#176a54"))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawRoundedRect(QRectF(8, 8, 240, 240), 42, 42)
    renderer.render(painter, QRectF(54, 54, 148, 148))
    painter.end()
    png = OUTPUT / "clearcanvas.png"
    assert pixmap.save(str(png))
    with Image.open(png) as image:
        image.save(OUTPUT / "clearcanvas.ico", sizes=[(16, 16), (32, 32), (48, 48), (256, 256)])
    prepare_gmic(OUTPUT)
    components = [f"G'MIC {GMIC_VERSION} (standalone CLI)"]
    for name in (
        "PySide6-Essentials",
        "shiboken6",
        "pypdfium2",
        "Pillow",
        "pypdf",
        "numpy",
        "opencv-python-headless",
        "scikit-image",
        "scipy",
        "pypatchmatch",
        "networkx",
        "imageio",
        "tifffile",
        "lazy-loader",
        "packaging",
    ):
        distribution = importlib.metadata.distribution(name)
        components.append(f"{distribution.metadata['Name']} {distribution.version}")
        for file in distribution.files or []:
            if not any(term in str(file).lower() for term in ("license", "copying", "notice")):
                continue
            source_path = Path(distribution.locate_file(file))
            if source_path.is_file() and source_path.suffix.lower() != ".py":
                target = OUTPUT / "licenses" / name / Path(str(file))
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source_path.read_bytes())
    (OUTPUT / "COMPONENTS.txt").write_text("\n".join(components) + "\n", encoding="utf-8")
    print("Build icon and third-party licenses prepared.")
    app.quit()


if __name__ == "__main__":
    main()
