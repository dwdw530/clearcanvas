import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from clearcanvas.documents import DocumentError
from clearcanvas.photo_repair import interpolate_wide_photo_strokes, restore_photo_texture
from clearcanvas.repair import _repair_image


def textured_pair():
    y, x = np.mgrid[:180, :320]
    texture = 12 * np.sin(x * 2 * np.pi / 13) + 9 * np.cos(y * 2 * np.pi / 11)
    rgb = np.stack((80 + texture + x / 10, 100 + texture, 145 + texture + y / 20), axis=2)
    clean = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).convert("RGBA")
    marked = clean.copy()
    ImageDraw.Draw(marked).text((80, 68), "TEXTURE", fill="white", font_size=28)
    ink = np.any(np.array(marked) != np.array(clean), axis=2)
    mask = cv2.dilate(ink.astype(np.uint8) * 255, np.ones((5, 5), np.uint8))
    return clean, marked, mask


def test_texture_transfer_reduces_reference_error_and_preserves_surroundings(monkeypatch):
    clean, marked, mask = textured_pair()
    with clean, marked, _repair_image(marked, mask, photographic=True) as repaired:
        monkeypatch.setattr(
            "clearcanvas.repair.restore_photo_texture",
            lambda rgb, mask, excluded, fallback, cancelled, budget: (fallback, 0),
        )
        with _repair_image(marked, mask, photographic=True) as interpolation:
            truth, actual, smooth = np.array(clean), np.array(repaired), np.array(interpolation)
            selected = mask != 0
            texture_error = np.abs(actual[selected].astype(float) - truth[selected]).mean()
            smooth_error = np.abs(smooth[selected].astype(float) - truth[selected]).mean()
            assert texture_error < smooth_error * 0.65
            assert np.array_equal(actual[~selected], np.array(marked)[~selected])


def test_texture_transfer_cancellation_during_matching():
    clean, marked, mask = textured_pair()
    calls = 0

    def cancelled():
        nonlocal calls
        calls += 1
        return calls == 5

    with clean, marked, pytest.raises(DocumentError) as error:
        restore_photo_texture(
            np.array(marked)[:, :, :3], mask, mask,
            np.array(clean)[:, :, :3], cancelled, 100,
        )
    assert error.value.code == "cancelled"


def test_texture_transfer_budget_and_unavailable_donors():
    clean, marked, mask = textured_pair()
    with clean, marked:
        rgb, fallback = np.array(marked)[:, :, :3], np.array(clean)[:, :, :3]
        result, steps = restore_photo_texture(rgb, mask, mask, fallback, lambda: False, 1)
        assert steps <= 1
        assert np.array_equal(result[mask == 0], rgb[mask == 0])
        result, steps = restore_photo_texture(
            rgb, mask, np.full_like(mask, 255), fallback, lambda: False, 100
        )
        assert steps == 0
        assert np.array_equal(result, fallback)


def test_one_unmatched_patch_does_not_abort_other_regions(monkeypatch):
    clean, marked, mask = textured_pair()
    match = cv2.matchTemplate
    calls = 0

    def reject_first(*args, **kwargs):
        nonlocal calls
        calls += 1
        scores = match(*args, **kwargs)
        return np.full_like(scores, np.inf) if calls == 1 else scores

    monkeypatch.setattr(cv2, "matchTemplate", reject_first)
    with clean, marked:
        rgb = np.array(marked)[:, :, :3]
        truth = np.array(clean)[:, :, :3]
        fallback = cv2.inpaint(rgb, mask, 3, cv2.INPAINT_NS)
        actual, steps = restore_photo_texture(rgb, mask, mask, fallback, lambda: False, 1000)
        selected = mask != 0
        assert steps > 1
        error = np.abs(actual[selected].astype(float) - truth[selected]).mean()
        baseline_error = np.abs(fallback[selected].astype(float) - truth[selected]).mean()
        assert error < baseline_error * 0.65


def test_wide_strokes_reconstruct_gradient_without_patch_seams():
    y, x = np.mgrid[:160, :240]
    truth = np.stack((70 + x // 2, 60 + y // 2, 80 + (x + y) // 4), axis=-1).astype(np.uint8)
    mask = np.zeros((160, 240), np.uint8)
    mask[40:115, 70:110] = 255
    mask[55:80, 130:175] = 255
    marked = truth.copy()
    marked[mask != 0] = (255, 0, 100)
    with Image.fromarray(marked).convert('RGBA') as image:
        with _repair_image(image, mask, photographic=True) as result:
            actual = np.array(result)[:, :, :3]
    assert np.array_equal(actual[mask == 0], truth[mask == 0])
    assert np.abs(actual[mask != 0].astype(float) - truth[mask != 0]).mean() < 1.0


def test_wide_fill_does_not_use_other_marked_pixels_or_black_canvas_padding():
    rgb = np.full((80, 160, 3), (70, 120, 180), np.uint8)
    mask = np.zeros((80, 160), np.uint8)
    mask[0:35, 20:70] = 255
    mask[40:70, 80:130] = 255
    rgb[mask != 0] = (255, 0, 100)
    result = interpolate_wide_photo_strokes(rgb, mask)
    assert np.max(np.abs(result.astype(int) - (70, 120, 180))) <= 1


def test_wide_fill_cancellation_inside_solver():
    rgb = np.full((100, 160, 3), 90, np.uint8)
    mask = np.zeros((100, 160), np.uint8)
    mask[25:70, 40:120] = 255
    rgb[mask != 0] = (255, 0, 100)
    calls = 0

    def cancelled():
        nonlocal calls
        calls += 1
        return calls >= 8

    with pytest.raises(DocumentError) as error:
        interpolate_wide_photo_strokes(rgb, mask, cancelled)
    assert error.value.code == 'cancelled'
