import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from clearcanvas.detection import detect_watermark_mask
from clearcanvas.documents import DocumentError, render_document
from clearcanvas.repair import auto_repair_document, source_stamp


@pytest.mark.parametrize("level", [205, 238, 246])
@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L"])
def test_large_pale_draft_across_dense_body(tmp_path, level, mode):
    clean = Image.new("RGBA", (900, 900), "white")
    layer = Image.new("RGBA", clean.size)
    ImageDraw.Draw(layer).text((95, 325), "DRAFT", fill=(level,) * 3, font_size=210)
    marked = clean.copy()
    marked.alpha_composite(layer.rotate(45, resample=Image.Resampling.BICUBIC))
    for image in (marked, clean):
        draw = ImageDraw.Draw(image)
        for y in range(20, 880, 27):
            draw.text((15, y), "Report body 123456789 - keep all original text. " * 2,
                      fill="black", font_size=18)
    path = tmp_path / "draft.png"
    marked.convert(mode).save(path)
    with Image.open(path) as source:
        mask = detect_watermark_mask(source)
    result = auto_repair_document(render_document(path), source_stamp(path))
    try:
        original, actual, truth = np.array(marked), np.array(result.image), np.array(clean)
        dark = cv2.dilate((original[:, :, 0] < 145).astype(np.uint8), np.ones((3, 3), np.uint8))
        assert np.array_equal(original[dark != 0], actual[dark != 0])
        assert np.array_equal(original[mask == 0], actual[mask == 0])
        mark = (original[:, :, 0] < 251) & (truth[:, :, 0] == 255) & (dark == 0)
        before = np.abs(original[mark].astype(float) - truth[mark]).mean()
        after = np.abs(actual[mark].astype(float) - truth[mark]).mean()
        assert after < before * 0.12
    finally:
        result.image.close()


@pytest.mark.parametrize("jpeg", [False, True])
def test_dense_clean_document_is_not_a_watermark(tmp_path, jpeg):
    image = Image.new("RGB", (700, 550), "white")
    draw = ImageDraw.Draw(image)
    for y in range(5, 540, 20):
        draw.text((4, y), "Ordinary report body text 123456789 " * 3, fill="black", font_size=14)
    path = tmp_path / ("clean.jpg" if jpeg else "clean.png")
    image.save(path, quality=75)
    with Image.open(path) as source, pytest.raises(DocumentError) as error:
        detect_watermark_mask(source)
    assert error.value.code == "no_watermark"


def test_cancel_during_detection():
    calls = 0

    def cancelled():
        nonlocal calls
        calls += 1
        return calls > 1

    image = Image.new("RGBA", (900, 900), "white")
    layer = Image.new("RGBA", image.size)
    ImageDraw.Draw(layer).text((100, 350), "DRAFT", fill="#eeeeee", font_size=180)
    image.alpha_composite(layer.rotate(45))
    with pytest.raises(DocumentError) as error:
        detect_watermark_mask(image, cancelled)
    assert error.value.code == "cancelled"


def test_pale_graphic_behind_body_text(tmp_path):
    clean = Image.new("RGBA", (800, 800), "white")
    marked = clean.copy()
    draw = ImageDraw.Draw(marked)
    draw.ellipse((220, 170, 580, 630), outline="#e0edf5", width=28)
    for image in (clean, marked):
        draw = ImageDraw.Draw(image)
        for y in range(40, 760, 27):
            draw.text((35, y), "Keep this report body 123456789. " * 3, fill="black", font_size=16)
    path = tmp_path / "graphic.png"
    marked.save(path)
    result = auto_repair_document(render_document(path), source_stamp(path))
    try:
        original, actual, truth = np.array(marked), np.array(result.image), np.array(clean)
        protected = cv2.dilate(
            (original[:, :, 0] < 145).astype(np.uint8), np.ones((3, 3), np.uint8)
        )
        assert np.array_equal(original[protected != 0], actual[protected != 0])
        exposed = (truth[:, :, 0] == 255) & (protected == 0)
        before = np.abs(original[exposed].astype(float) - truth[exposed]).mean()
        after = np.abs(actual[exposed].astype(float) - truth[exposed]).mean()
        assert after < before * 0.1
    finally:
        result.image.close()


@pytest.mark.parametrize("panel", [False, True])
def test_ordinary_pale_illustration_and_text_panel_rejected(panel):
    image = Image.new("RGBA", (800, 800), "white")
    draw = ImageDraw.Draw(image)
    if panel:
        draw.rectangle((180, 160, 620, 640), fill="#e0edf5")
        for y in range(180, 620, 27):
            draw.text((200, y), "Ordinary shaded text panel", fill="black", font_size=16)
    else:
        draw.ellipse((220, 170, 580, 630), outline="#e0edf5", width=28)
        draw.text((250, 680), "Ordinary illustration", fill="black", font_size=20)
    with pytest.raises(DocumentError) as error:
        detect_watermark_mask(image)
    assert error.value.code == "no_watermark"


def test_public_draft_regression_when_cached():
    root = Path(__file__).resolve().parents[1] / "artifacts/web-samples-2026-10-06"
    path = root / "01-draft-document.jpg"
    if not path.exists():
        pytest.skip("Public acceptance sample is not redistributed with the project")
    before_hash = hashlib.sha256(path.read_bytes()).digest()
    with Image.open(path).convert("RGBA") as source:
        original = np.array(source)
    result = auto_repair_document(render_document(path), source_stamp(path))
    try:
        actual = np.array(result.image)
        dark = cv2.dilate((original[:, :, 0] < 145).astype(np.uint8), np.ones((3, 3), np.uint8))
        assert np.array_equal(actual[dark != 0], original[dark != 0])
        assert np.count_nonzero(np.any(actual != original, axis=2)) > 25_000
        # Exposed bottom stroke, away from text; no original clean reference exists.
        assert actual[515:539, 165:205, :3].mean() > 253
        assert hashlib.sha256(path.read_bytes()).digest() == before_hash
    finally:
        result.image.close()


def test_tiled_color_watermark_keeps_application_chrome(tmp_path):
    clean = Image.new("RGBA", (800, 800), "#eeeeee")
    draw = ImageDraw.Draw(clean)
    draw.rectangle((35, 110, 765, 765), fill="white")
    draw.text((40, 35), "Application toolbar", fill="black", font_size=22)
    draw.ellipse((340, 25, 367, 60), outline="#a0ccd8", width=2)
    marked = clean.copy()
    draw = ImageDraw.Draw(marked)
    for y in range(155, 700, 125):
        for x in range(75, 710, 125):
            draw.ellipse((x, y, x + 27, y + 35), outline="#a0ccd8", width=2)
    for image in (marked, clean):
        draw = ImageDraw.Draw(image)
        for y in range(160, 730, 23):
            draw.text((60, y), "Report body to preserve 123456789 " * 3,
                      fill="black", font_size=16)
    path = tmp_path / "tiled.png"
    marked.save(path)
    result = auto_repair_document(render_document(path), source_stamp(path))
    try:
        original, actual, truth = np.array(marked), np.array(result.image), np.array(clean)
        assert np.array_equal(actual[:110], original[:110])
        protected = cv2.dilate(
            (original[:, :, :3].min(axis=2) < 145).astype(np.uint8), np.ones((3, 3), np.uint8)
        )
        assert np.array_equal(actual[protected != 0], original[protected != 0])
        exposed = (truth[:, :, 0] == 255) & (protected == 0)
        before = np.abs(original[exposed].astype(float) - truth[exposed]).mean()
        after = np.abs(actual[exposed].astype(float) - truth[exposed]).mean()
        assert after < before * 0.08
    finally:
        result.image.close()


def test_repeated_illustrations_without_overprinted_text_are_not_watermarks():
    image = Image.new("RGBA", (800, 800), "white")
    draw = ImageDraw.Draw(image)
    for y in range(150, 700, 125):
        for x in range(75, 710, 125):
            draw.ellipse((x, y, x + 27, y + 35), outline="#a0ccd8", width=2)
            draw.text((x, y + 55), "Item", fill="black", font_size=12)
    with pytest.raises(DocumentError) as error:
        detect_watermark_mask(image)
    assert error.value.code == "no_watermark"


def test_public_tiled_document_when_cached():
    root = Path(__file__).resolve().parents[1] / "artifacts/web-samples-2026-10-06"
    path = root / "03-confidential-document.png"
    if not path.exists():
        pytest.skip("Public acceptance sample is not redistributed with the project")
    before_hash = hashlib.sha256(path.read_bytes()).digest()
    with Image.open(path).convert("RGBA") as source:
        original = np.array(source)
    result = auto_repair_document(render_document(path), source_stamp(path))
    try:
        actual = np.array(result.image)
        assert np.array_equal(original[:95], actual[:95])
        assert np.array_equal(original[499:], actual[499:])
        protected = cv2.dilate(
            (original[:, :, :3].min(axis=2) < 145).astype(np.uint8), np.ones((3, 3), np.uint8)
        )
        assert np.array_equal(actual[protected != 0], original[protected != 0])
        # Unobscured first word and shield: background should return to white.
        assert actual[125:145, 46:385, :3].mean() > 254
        assert actual[121:149, 200:224, :3].mean() > 254
        assert actual[96:114, 458:477, :3].mean() > 254
        assert hashlib.sha256(path.read_bytes()).digest() == before_hash
    finally:
        result.image.close()
