import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "windows" if os.name == "nt" else "offscreen")

import pytest
from PIL import Image, ImageDraw
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from clearcanvas.theme import apply_theme


def make_pdf(path: Path, password: str | None = None) -> Path:
    writer = PdfWriter()
    font = writer._add_object(
        DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
    )
    for index, (width, height) in enumerate(((595, 842), (842, 595), (420, 595))):
        page = writer.add_blank_page(width, height)
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
            }
        )
        contents = DecodedStreamObject()
        contents.set_data(
            (
                f"0.08 0.36 0.29 rg 0 {height - 24} {width} 24 re f "
                f"BT /F1 28 Tf 44 {height - 100} Td (ClearCanvas) Tj ET "
                f"0.30 0.35 0.33 rg BT /F1 12 Tf 44 {height - 128} Td "
                f"(Sample document / page {index + 1}) Tj ET "
                f"0.88 0.92 0.90 rg 44 {height - 265} {width - 88} 94 re f "
                f"0.64 0.29 0.13 rg 60 {height - 240} 100 40 re f "
                f"0.18 0.46 0.39 rg 178 {height - 240} 160 40 re f "
                f"0.30 0.35 0.33 rg BT /F1 11 Tf 44 {height - 310} Td "
                "(Local preview quality check. Original file remains unchanged.) Tj ET"
            ).encode("ascii")
        )
        page[NameObject("/Contents")] = writer._add_object(contents)
    if password is not None:
        writer.encrypt(password)
    writer.write(path)
    return path


@pytest.fixture
def sample_image(tmp_path):
    path = tmp_path / "测试 图片.png"
    image = Image.new("RGBA", (960, 640), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((50, 60, 450, 560), fill="#176a54")
    draw.rectangle((480, 60, 910, 350), fill="#c76b36")
    draw.rectangle((480, 380, 910, 560), fill="#cbdeda")
    image.save(path)
    return path


@pytest.fixture
def sample_pdf(tmp_path):
    return make_pdf(tmp_path / "测试 文档.pdf")


@pytest.fixture
def encrypted_pdf(tmp_path):
    return make_pdf(tmp_path / "加密 文档.pdf", "clearcanvas-test")


@pytest.fixture
def window(qtbot, qapp, monkeypatch):
    from PySide6.QtCore import Qt

    from clearcanvas.window import MainWindow

    apply_theme(qapp)
    widget = MainWindow()
    monkeypatch.setattr(widget, "_choose_close_action", lambda: "exit")
    widget.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    qtbot.addWidget(widget)
    widget.show()
    yield widget
    widget.request_exit()
