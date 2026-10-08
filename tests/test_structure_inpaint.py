import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from clearcanvas import structure_inpaint as structure
from clearcanvas.documents import DocumentError
from clearcanvas.photo_repair import interpolate_wide_photo_strokes
from clearcanvas.repair import _repair_image


@pytest.fixture
def runtime():
    if not structure.executable_path().is_file():
        pytest.skip("Run tools/prepare_gmic.py to test the pinned native runtime")


def contour_pair():
    yy, xx = np.mgrid[:180, :280]
    edge = 84 + 0.12 * (xx - 140) + 0.0008 * (xx - 140) ** 2
    upper = (1 + np.tanh((yy - edge) / 1.4)) / 2
    rgb = np.stack((
        175 + xx * 0.08 + upper * 25,
        125 + yy * 0.08 - upper * 65,
        85 + yy * 0.03 - upper * 30,
    ), axis=-1).astype(np.uint8)
    mask = np.zeros(rgb.shape[:2], np.uint8)
    mask[50:128, 116:164] = 255
    marked = rgb.copy()
    marked[mask != 0] = (255, 0, 100)
    return rgb, marked, mask


def test_edge_transport_improves_observed_contour_without_changing_surroundings(runtime):
    truth, marked, mask = contour_pair()
    original, selection = marked.copy(), mask.copy()
    baseline = interpolate_wide_photo_strokes(marked, mask)
    with Image.fromarray(marked).convert("RGBA") as image:
        with _repair_image(image, mask, photographic=True) as result:
            actual = np.array(result)[:, :, :3]
    selected = mask != 0
    error = np.abs(actual[selected].astype(float) - truth[selected]).mean()
    old_error = np.abs(baseline[selected].astype(float) - truth[selected]).mean()
    assert error < old_error * 0.7
    assert np.array_equal(actual[~selected], original[~selected])
    assert np.array_equal(marked, original)
    assert np.array_equal(mask, selection)


def test_all_excluded_colors_are_ignored_and_canvas_edges_have_no_black_fringe(runtime):
    rgb = np.full((90, 150, 3), (80, 120, 160), np.uint8)
    mask = np.zeros(rgb.shape[:2], np.uint8)
    mask[:35, 20:65] = 255
    mask[45:80, 75:120] = 255
    first, second = rgb.copy(), rgb.copy()
    first[mask != 0] = (255, 0, 100)
    second[mask != 0] = (0, 255, 0)
    a = structure.restore_photo_structure(first, mask)
    b = structure.restore_photo_structure(second, mask)
    assert np.array_equal(a, b)
    assert np.max(np.abs(a.astype(int) - rgb)) <= 1


def test_missing_runtime_reports_error_instead_of_a_successful_fallback(monkeypatch, tmp_path):
    _, marked, mask = contour_pair()
    monkeypatch.setattr(structure, "executable_path", lambda: tmp_path / "missing.exe")
    with pytest.raises(DocumentError) as error:
        structure.restore_photo_structure(marked, mask)
    assert error.value.code == "repair_unavailable"


def test_invalid_native_output_is_rejected_and_temporary_files_are_removed(monkeypatch, tmp_path):
    _, marked, mask = contour_pair()
    executable = tmp_path / "gmic.exe"
    executable.touch()
    monkeypatch.setattr(structure, "executable_path", lambda: executable)
    directories = []

    def run(executable, directory, cancelled):
        directories.append(directory)
        with Image.new("RGB", (2, 2), "white") as image:
            image.save(directory / "result.ppm")

    monkeypatch.setattr(structure, "_run_structure", run)
    with pytest.raises(DocumentError) as error:
        structure.restore_photo_structure(marked, mask)
    assert error.value.code == "repair_failed"
    assert directories and not directories[0].exists()


@pytest.mark.parametrize("reason", ["cancelled", "timeout", "failed"])
def test_child_process_is_reaped_on_cancel_timeout_and_failure(monkeypatch, tmp_path, reason):
    calls = 0

    class Process:
        returncode = None
        killed = False
        communicated = False

        def __init__(self, command, **options):
            assert "-inpaint_pde[0]" in command
            assert not any("http" in argument for argument in command)
            assert options["env"]["GMIC_PATH"] == str(tmp_path)
            assert options["cwd"] == tmp_path
            assert options["stdout"] == subprocess.DEVNULL

        def __enter__(self):
            return self

        def __exit__(self, *args):
            assert self.returncode is not None

        def communicate(self, timeout=None):
            self.communicated = True
            if reason == "failed":
                self.returncode = 1
            elif not self.killed:
                raise subprocess.TimeoutExpired("gmic", timeout)
            return b"", b""

        def poll(self):
            return self.returncode

        def kill(self):
            self.killed = True
            self.returncode = -1

    processes = []

    def start(*args, **kwargs):
        process = Process(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(structure.subprocess, "Popen", start)
    if reason == "timeout":
        monkeypatch.setattr(structure, "STRUCTURE_TIMEOUT", 0)

    def cancelled():
        nonlocal calls
        calls += 1
        return reason == "cancelled" and calls > 1

    with pytest.raises(DocumentError) as error:
        structure._run_structure(Path("gmic.exe"), tmp_path, cancelled)
    assert error.value.code == {"cancelled": "cancelled", "timeout": "repair_timeout",
                                "failed": "repair_failed"}[reason]
    process = processes[0]
    assert process.communicated
    assert process.killed == (reason != "failed")


def test_large_or_empty_source_regions_do_not_start_native_process(monkeypatch):
    _, marked, mask = contour_pair()

    def run(*args):
        pytest.fail("A refused region must not start the native runtime")

    monkeypatch.setattr(structure, "_run_structure", run)
    with pytest.raises(DocumentError, match="干净背景"):
        structure.restore_photo_structure(marked, np.full_like(mask, 255))
    monkeypatch.setattr(structure, "MAX_STRUCTURE_PIXELS", 1)
    with pytest.raises(DocumentError) as error:
        structure.restore_photo_structure(marked, mask)
    assert error.value.code == "repair_limit"
    assert np.array_equal(structure.restore_photo_structure(marked, np.zeros_like(mask)), marked)


def test_transparent_pixels_are_preserved_by_complete_photo_pipeline(runtime):
    _, marked, mask = contour_pair()
    rgba = cv2.cvtColor(marked, cv2.COLOR_RGB2RGBA)
    rgba[10:40, 30:90, :3] = (0, 255, 0)
    rgba[10:40, 30:90, 3] = 0
    with Image.fromarray(rgba) as image:
        with _repair_image(image, mask, photographic=True) as result:
            actual = np.array(result)
    assert np.array_equal(actual[mask == 0], rgba[mask == 0])
    assert not np.any(np.all(actual[mask != 0, :3] == (0, 255, 0), axis=1))
