import numpy as np
import pytest
from PIL import Image

from clearcanvas import structure_inpaint as structure
from clearcanvas.documents import DocumentError
from clearcanvas.photo_repair import interpolate_wide_photo_strokes
from clearcanvas.repair import _repair_image
from clearcanvas.texture_inpaint import restore_wide_texture


@pytest.fixture
def runtime():
    if not structure.executable_path().is_file():
        pytest.skip("Run tools/prepare_gmic.py to test the pinned native runtime")


def broad_texture():
    yy, xx = np.mgrid[:180, :320]
    texture = 18 * np.sin(xx * 2 * np.pi / 13) + 14 * np.cos(yy * 2 * np.pi / 11)
    rgb = np.stack((90 + texture, 120 + texture, 160 + texture), axis=2).astype(np.uint8)
    mask = np.zeros((180, 320), np.uint8)
    mask[60:115, 115:175] = 255
    marked = rgb.copy()
    marked[mask != 0] = (255, 0, 100)
    return rgb, marked, mask


def test_upstream_texture_fill_improves_broad_reference_and_preserves_surroundings(runtime):
    truth, marked, mask = broad_texture()
    baseline = interpolate_wide_photo_strokes(marked, mask)
    snapshot = mask.copy()
    with Image.fromarray(marked).convert('RGBA') as image:
        with _repair_image(image, mask, photographic=True) as result:
            actual = np.array(result)[:, :, :3]
    selected = mask != 0
    assert np.array_equal(mask, snapshot)
    assert np.array_equal(actual[~selected], marked[~selected])
    error = np.abs(actual[selected].astype(float) - truth[selected]).mean()
    old_error = np.abs(baseline[selected].astype(float) - truth[selected]).mean()
    assert error < old_error * 0.8


def test_marked_and_transparent_donors_do_not_influence_texture():
    truth, marked, mask = broad_texture()
    excluded = mask.copy()
    excluded[20:140, 220:250] = 255
    first = marked.copy()
    second = marked.copy()
    first[excluded != 0] = (255, 0, 100)
    second[excluded != 0] = (0, 255, 0)
    a = restore_wide_texture(first, mask, excluded, truth)
    b = restore_wide_texture(second, mask, excluded, truth)
    assert np.array_equal(a, b)
    assert np.array_equal(a[mask == 0], truth[mask == 0])


def test_clipped_strokes_have_no_watermark_or_black_canvas_fringe():
    truth, _, _ = broad_texture()
    mask = np.zeros(truth.shape[:2], np.uint8)
    mask[:35, 110:175] = 255
    marked = truth.copy()
    marked[mask != 0] = (255, 0, 100)
    baseline = interpolate_wide_photo_strokes(marked, mask)
    actual = restore_wide_texture(marked, mask, mask, baseline)
    assert not np.any(np.all(actual == (255, 0, 100), axis=2))
    assert actual[mask != 0].min() > 30
    assert np.array_equal(actual[mask == 0], marked[mask == 0])


def test_cancellation_after_native_step_discards_result(monkeypatch):
    truth, marked, mask = broad_texture()
    completed = False

    def native(*args, **kwargs):
        nonlocal completed
        completed = True
        return truth.copy()

    monkeypatch.setattr('clearcanvas.texture_inpaint.patchmatch.inpaint', native)
    with pytest.raises(DocumentError) as error:
        restore_wide_texture(marked, mask, mask, truth, lambda: completed)
    assert error.value.code == 'cancelled'


def test_smooth_region_keeps_interpolation_without_native_call(monkeypatch):
    truth = np.full((100, 160, 3), (80, 120, 160), np.uint8)
    mask = np.zeros((100, 160), np.uint8)
    mask[30:70, 40:100] = 255
    marked = truth.copy()
    marked[mask != 0] = (255, 0, 100)

    def native(*args, **kwargs):
        pytest.fail('A constant background needs no patch search')

    monkeypatch.setattr('clearcanvas.texture_inpaint.patchmatch.inpaint', native)
    assert np.array_equal(restore_wide_texture(marked, mask, mask, truth), truth)
