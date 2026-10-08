import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw
from PySide6.QtWidgets import QMessageBox

from clearcanvas.detection import detect_watermark
from clearcanvas.documents import DocumentError, render_document
from clearcanvas.photo_watermarks import detect_photo_watermark_mask
from clearcanvas.repair import auto_repair_document, export_result, source_stamp


def photo_pair(color=255, opacity=255, corner=False):
    y, x = np.mgrid[:640, :960]
    channels = (
        80 + x / 14 + 12 * np.sin(y / 37),
        100 + y / 16 + 10 * np.cos(x / 43),
        145 + x / 22 + 12 * np.sin((x + y) / 19),
    )
    clean = Image.fromarray(np.stack(channels, axis=2).astype(np.uint8)).convert("RGBA")
    layer = Image.new("RGBA", clean.size)
    draw = ImageDraw.Draw(layer)
    ink = ((color,) * 3 if isinstance(color, int) else color) + (opacity,)
    if corner:
        draw.text((230, 590), "PHOTOGRAPHY.COM", fill=ink, font_size=28)
    else:
        for yy in (-40, 180, 400, 620):
            for xx in (-110, 440):
                draw.text((xx, yy), "WATERMARK", fill=ink, font_size=46)
        layer = layer.rotate(20, resample=Image.Resampling.BICUBIC)
    return clean, Image.alpha_composite(clean, layer)


@pytest.mark.parametrize("hue", range(0, 180, 15))
@pytest.mark.parametrize("opacity,corner", [(255, False), (160, True)])
def test_photo_colored_ink_across_hue_wheel(hue, opacity, corner):
    color = tuple(int(c) for c in cv2.cvtColor(
        np.uint8([[[hue, 230, 240]]]), cv2.COLOR_HSV2RGB
    )[0, 0])
    clean, marked = photo_pair(color, opacity, corner)
    with clean, marked:
        ink = np.max(np.abs(np.array(marked).astype(int) - np.array(clean)), axis=2) > 20
        mask = detect_photo_watermark_mask(marked)
        assert np.mean(mask[ink] != 0) > 0.98
        nearby = cv2.dilate(ink.astype(np.uint8), np.ones((11, 11), np.uint8)) != 0
        assert np.count_nonzero((mask != 0) & ~nearby) < ink.sum() * 0.05


@pytest.mark.parametrize("filename,digest", [
    ("06-photo-tiled-changed.png",
     "3006df31ddf393133ee94da2c11acffbd745877c2a13a8e5f744f8d950e5d2ea"),
    ("10-photo-corner-changed.png",
     "59c0496202f4799781e2691605a4bec18d3807cd730b20cbdfd68c17f63fc4ff"),
])
def test_user_supplied_color_images_are_accepted_unchanged(filename, digest):
    path = (Path(__file__).resolve().parents[1]
            / "artifacts/non-ai-followup-2026-10-07" / filename)
    if not path.exists():
        pytest.skip("User acceptance images are not redistributed")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    with Image.open(path).convert("RGB") as image:
        ink = np.all(np.array(image) == (255, 0, 100), axis=2)
        mask = detect_photo_watermark_mask(image)
        assert np.count_nonzero(ink) > 9000
        assert np.all(mask[ink])
        nearby = cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8)) != 0
        assert np.count_nonzero((mask != 0) & ~nearby) < ink.sum() * 0.01


@pytest.mark.parametrize("color", [(240, 20, 80), (20, 200, 60), (20, 30, 240)])
def test_colored_headings_and_interior_captions_are_preserved(color):
    clean, marked = photo_pair()
    with clean, marked:
        draw = ImageDraw.Draw(clean)
        draw.text((140, 15), "PHOTOGRAPHY TUTORIAL", fill=color, font_size=28)
        draw.text((150, 280), "Ordinary photo caption", fill=color, font_size=32)
        with pytest.raises(DocumentError) as error:
            detect_photo_watermark_mask(clean)
        assert error.value.code == "no_watermark"


@pytest.mark.parametrize("scale", [1, 2])
@pytest.mark.parametrize("color", [(240, 20, 80), (20, 200, 60), (20, 30, 240)])
def test_opaque_strokes_do_not_erase_the_clean_band(color, scale):
    clean, unused = photo_pair()
    unused.close()
    with clean, clean.resize((960 * scale, 640 * scale)) as background:
        with background.copy() as marked:
            ImageDraw.Draw(marked).text(
                (300 * scale, 530 * scale), "WATERMARK", fill=color, font_size=48 * scale
            )
            original = np.array(marked).astype(np.int16)
            truth = np.array(background).astype(np.int16)
            ink = np.max(np.abs(original - truth), axis=2) > 10
            mask = detect_photo_watermark_mask(marked)
            assert np.mean(mask[ink] != 0) > 0.98
            fringe = cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8)) != 0
            assert np.count_nonzero((mask != 0) & ~fringe) < ink.sum() * 0.02


def test_color_refinement_keeps_both_colors_of_bottom_lettering():
    with Image.new('RGBA', (1200, 800), (95, 110, 130, 255)) as image:
        draw = ImageDraw.Draw(image)
        for x, color in [(110, (240, 20, 80)), (640, (20, 210, 80))]:
            draw.text((x, 690), 'WATERMARK', fill=color, font_size=52)
        pixels = np.array(image)
        mask = detect_photo_watermark_mask(image)
        for color in [(240, 20, 80), (20, 210, 80)]:
            ink = np.all(pixels[:, :, :3] == color, axis=2)
            assert ink.sum() > 1000
            assert np.all(mask[ink])


@pytest.mark.parametrize(
    "color,opacity,corner", [(255, 255, False), (0, 255, False), (255, 180, True)]
)
def test_photo_strokes_quality_and_outside_protection(tmp_path, color, opacity, corner):
    clean, marked = photo_pair(color, opacity, corner)
    path = tmp_path / "photo.png"
    marked.save(path)
    digest = hashlib.sha256(path.read_bytes()).digest()
    detection = detect_watermark(marked)
    assert detection.photographic
    result = auto_repair_document(render_document(path), source_stamp(path))
    try:
        original, actual, truth = np.array(marked), np.array(result.image), np.array(clean)
        ink = np.max(np.abs(original.astype(float) - truth), axis=2) > 20
        assert np.mean(detection.mask[ink] != 0) > 0.95
        assert np.array_equal(original[detection.mask == 0], actual[detection.mask == 0])
        before = np.abs(original[ink].astype(float) - truth[ink]).mean()
        after = np.abs(actual[ink].astype(float) - truth[ink]).mean()
        assert after < before * 0.25
        surrounding = cv2.dilate(ink.astype(np.uint8), np.ones((9, 9), np.uint8)) != 0
        assert np.count_nonzero((detection.mask != 0) & ~surrounding) < ink.sum() * 0.05
        assert hashlib.sha256(path.read_bytes()).digest() == digest
    finally:
        result.image.close()
        marked.close()
        clean.close()


def test_clean_photo_and_single_interior_caption_are_rejected():
    clean, marked = photo_pair()
    with clean, marked:
        with pytest.raises(DocumentError) as error:
            detect_watermark(clean)
        assert error.value.code == "no_watermark"
        ImageDraw.Draw(clean).text((150, 280), "Ordinary photo caption", fill="white", font_size=32)
        with pytest.raises(DocumentError) as error:
            detect_watermark(clean)
        assert error.value.code == "no_watermark"


def test_photo_cancellation_is_not_reported_as_no_watermark():
    clean, marked = photo_pair()
    calls = 0

    def cancelled():
        nonlocal calls
        calls += 1
        return calls > 3

    with clean, marked, pytest.raises(DocumentError) as error:
        detect_photo_watermark_mask(marked, cancelled)
    assert error.value.code == "cancelled"


def test_single_top_heading_is_not_a_watermark():
    clean, marked = photo_pair()
    with clean, marked:
        ImageDraw.Draw(clean).text(
            (140, 15), "PHOTOGRAPHY TUTORIAL", fill="white", font_size=28
        )
        with pytest.raises(DocumentError) as error:
            detect_photo_watermark_mask(clean)
        assert error.value.code == "no_watermark"


def test_different_interior_caption_is_not_a_repeated_corner_watermark():
    clean, marked = photo_pair(corner=True)
    with clean, marked:
        ImageDraw.Draw(marked).text(
            (150, 280), "ORDINARY CAPTION", fill="#e6e6e6", font_size=28
        )
        mask = detect_photo_watermark_mask(marked)
        assert np.any(mask[580:])
        assert not np.any(mask[270:330])


def test_dark_letter_counters_are_preserved():
    with Image.new("RGBA", (960, 640), (25, 25, 25, 255)) as image:
        ImageDraw.Draw(image).text(
            (220, 540), "BBOOBBOOBBOO", fill="white", font_size=40
        )
        ink = np.array(image)[:, :, 0] > 100
        mask = detect_photo_watermark_mask(image)
        allowed = cv2.dilate(ink.astype(np.uint8), np.ones((7, 7), np.uint8)) != 0
        assert np.mean(mask[ink] != 0) > 0.98
        assert not np.any(mask[~allowed])


def test_detached_tinted_crossbar_is_completed():
    with Image.new("RGBA", (960, 640), (30, 35, 80, 255)) as image:
        draw = ImageDraw.Draw(image)
        color = (155, 147, 172, 255)
        draw.text((570, 530), "WATERMARK", fill=color, font_size=40)
        draw.rectangle((578, 531, 593, 533), fill=color)
        mask = detect_photo_watermark_mask(image)
        assert np.all(mask[531:534, 578:594])


def test_corner_position_uses_lettering_bounds_before_dilation():
    # The ink starts below 82% of the image, but its grouping halo starts above it.
    with Image.new("RGBA", (960, 640), (60, 110, 170, 255)) as image:
        ImageDraw.Draw(image).text(
            (570, 515), "WATERMARK", fill=(185, 210, 240, 255), font_size=40
        )
        mask = detect_photo_watermark_mask(image)
        ink = np.array(image)[:, :, 0] > 100
        assert np.mean(mask[ink] != 0) > 0.98
        assert not np.any(mask[:520])
        assert not np.any(mask[:, :560])


def test_corner_disconnected_short_stroke_is_completed():
    with Image.new("RGBA", (960, 640), (60, 110, 170, 255)) as image:
        draw = ImageDraw.Draw(image)
        draw.text((570, 530), "WATERMARK", fill=(185, 210, 240, 255), font_size=40)
        draw.rectangle((585, 536, 591, 538), fill=(185, 210, 240, 255))
        # Nearby unrelated content must remain outside the text mask.
        draw.rectangle((840, 530, 855, 550), fill=(185, 210, 240, 255))
        mask = detect_photo_watermark_mask(image)
        assert np.all(mask[536:539, 585:592])
        assert not np.any(mask[530:551, 840:856])


def test_corner_low_contrast_letters_and_bright_background_gaps(tmp_path):
    # The first letter lies on bright water, the rest on a dark reflection.
    # Bright gaps alongside the line must not be treated as extra lettering.
    y, x = np.mgrid[:640, :960]
    gray = 75 + 88 / (1 + np.exp(np.clip((x - 470) / 5, -50, 50)))
    gray = np.clip(gray + 3 * np.sin(y / 4), 0, 255).astype(np.uint8)
    clean = Image.fromarray(np.repeat(gray[:, :, None], 3, axis=2)).convert("RGBA")
    draw = ImageDraw.Draw(clean)
    for yy in (548, 557, 566):
        draw.rectangle((724, yy, 744, yy + 3), fill=(160, 160, 160, 255))
    marked = clean.copy()
    ImageDraw.Draw(marked).text((420, 530), "WATERMARK", fill="#cccccc", font_size=40)
    path = tmp_path / "mixed-contrast.png"
    marked.save(path)
    try:
        original, truth = np.array(marked), np.array(clean)
        ink = np.max(np.abs(original.astype(int) - truth.astype(int)), axis=2) > 15
        first_letter = ink & (x < 450)
        assert np.count_nonzero(first_letter) > 30
        mask = detect_photo_watermark_mask(marked)
        assert np.mean(mask[first_letter] != 0) > 0.98
        assert np.mean(mask[ink] != 0) > 0.98
        assert not np.any(mask[545:575, 724:745])
        result = auto_repair_document(render_document(path), source_stamp(path))
        try:
            actual = np.array(result.image)
            assert np.array_equal(actual[mask == 0], original[mask == 0])
            before = np.abs(original[ink].astype(float) - truth[ink]).mean()
            after = np.abs(actual[ink].astype(float) - truth[ink]).mean()
            assert after < before * 0.25
        finally:
            result.image.close()
    finally:
        marked.close()
        clean.close()


def test_photo_mask_dilation_preserves_transparent_pixels():
    clean, marked = photo_pair(corner=True)
    with clean, marked:
        pixels = np.array(marked)
        # A thin translucent seam beside the lettering must not be inpainted.
        pixels[:, 300, 3] = 120
        with Image.fromarray(pixels) as image:
            mask = detect_photo_watermark_mask(image)
            assert np.any(mask)
            assert not np.any(mask[:, 300])


def test_cached_tutorial_heading_is_preserved():
    root = Path(__file__).resolve().parents[1] / "artifacts/web-samples-2026-10-06"
    paths = list(root.glob("08-photo-tutorial.*"))
    paths = [path for path in paths if path.suffix.lower() in {".jpg", ".png", ".webp"}]
    if not paths:
        pytest.skip("Public acceptance sample is not redistributed with the project")
    with Image.open(paths[0]) as image:
        try:
            mask = detect_photo_watermark_mask(image)
        except DocumentError as error:
            assert error.code == "no_watermark"
        else:
            assert not np.any(mask[: int(image.height * 0.22)])


def test_cached_tiled_portrait_matches_clipped_copies():
    path = (
        Path(__file__).resolve().parents[1]
        / "artifacts/web-samples-2026-10-06/06-photo-tiled.jpg"
    )
    if not path.exists():
        pytest.skip("Public acceptance sample is not redistributed with the project")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == "1035451ebe77da32b9deec446386219b665825805a21cdd68c15cbce80c3ca3e"
    with Image.open(path).convert("RGB") as image:
        mask = detect_photo_watermark_mask(image)
        # The complete central word was missed when its only partners on other
        # rows were cropped by the image border. These are original white pixels,
        # not the magenta change-overlay artifact from the acceptance tooling.
        rgb = np.array(image)[540:985, 1370:2300]
        white = rgb.min(axis=2) >= 248
        assert np.count_nonzero(white) > 20_000
        assert np.mean(mask[540:985, 1370:2300][white] != 0) > 0.98


@pytest.mark.parametrize("filename", ["09-clean-control.png", "11-office-photo.jpg"])
def test_cached_clean_controls_remain_rejected(filename):
    path = Path(__file__).resolve().parents[1] / "artifacts/web-samples-2026-10-06" / filename
    if not path.exists():
        pytest.skip("Public acceptance sample is not redistributed with the project")
    with Image.open(path) as image, pytest.raises(DocumentError) as error:
        detect_watermark(image)
    assert error.value.code == "no_watermark"


def test_photo_ui_one_click_and_export_without_selection(qtbot, window, tmp_path, monkeypatch):
    clean, marked = photo_pair(corner=True)
    path = tmp_path / "photo.png"
    marked.save(path)
    clean.close()
    marked.close()
    digest = hashlib.sha256(path.read_bytes()).digest()
    messages = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a: messages.append(a[2]) or QMessageBox.Ok)
    window.add_files([path])
    qtbot.waitUntil(lambda: window._preview is not None)
    assert window.editor.auto_button.text() == "一键去水印"
    assert not window.canvas.regions
    window.editor.auto_button.click()
    qtbot.waitUntil(lambda: not window.editor.busy, timeout=15000)
    assert not messages
    assert window.editor.result is not None
    assert window.editor.export_button.isEnabled()
    output = tmp_path / "exported.png"
    monkeypatch.setattr(
        "clearcanvas.editor.QFileDialog.getSaveFileName", lambda *a: (str(output), "PNG (*.png)")
    )
    window.editor.export_button.click()
    qtbot.waitUntil(lambda: not window.editor.busy and output.exists(), timeout=15000)
    assert hashlib.sha256(path.read_bytes()).digest() == digest
    with Image.open(output) as exported:
        assert exported.size == (960, 640)
    window.editor.dirty = False


def test_photo_pdf_selected_page_export(tmp_path):
    clean, marked = photo_pair(corner=True)
    path = tmp_path / "photos.pdf"
    with clean, marked, clean.convert("RGB") as first, marked.convert("RGB") as second:
        first.save(path, save_all=True, append_images=[second], resolution=100)
    digest = hashlib.sha256(path.read_bytes()).digest()
    original = render_document(path, 0)
    result = auto_repair_document(render_document(path, 1), source_stamp(path))
    try:
        output = export_result(result, tmp_path / "out.pdf", pages=[1])
        assert render_document(output, 0).rgba == original.rgba
        assert hashlib.sha256(path.read_bytes()).digest() == digest
    finally:
        result.image.close()
