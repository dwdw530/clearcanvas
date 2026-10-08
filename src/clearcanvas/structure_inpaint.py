"""Use G'MIC's non-neural, multiscale edge transport for broad photo strokes."""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from clearcanvas.documents import DocumentError

GMIC_VERSION = "4.0.5"
GMIC_URL = "https://gmic.eu/get_file.php?file=windows/gmic_4.0.5_cli_win64.zip"
GMIC_SHA256 = "3f7ae429fa9758db41dae89d3fdc048e60d5d7e820e9ff03419c5bd7a10c2d95"
GMIC_EXE_SHA256 = "f5c93b69d74650b69b18aabf997b9a27c206508fda32dff767641cd3e0c56bd8"
MAX_STRUCTURE_PIXELS = 1_500_000
STRUCTURE_TIMEOUT = 30.0


def executable_path():
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "gmic" / "gmic.exe"
    return Path(__file__).resolve().parents[2] / "artifacts" / "build" / "gmic" / "gmic.exe"


def _check_cancelled(cancelled):
    if cancelled():
        raise DocumentError("已取消。", "cancelled")


def _run_structure(executable, directory, cancelled):
    # Isolate user.gmic and cached command definitions. Only the bundled standard
    # library is used; there is no update command, user script, or network input.
    environment = os.environ.copy()
    environment["GMIC_PATH"] = str(directory)
    environment["OMP_NUM_THREADS"] = str(min(4, os.cpu_count() or 1))
    command = [
        str(executable), "-v", "-1", "-input", "source.ppm", "-input", "mask.pgm",
        "-inpaint_pde[0]", "[1],-9,2,20", "-output[0]", "result.ppm",
    ]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    deadline = time.monotonic() + STRUCTURE_TIMEOUT
    with subprocess.Popen(
        command, cwd=directory, env=environment, creationflags=flags,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    ) as process:
        try:
            while True:
                _check_cancelled(cancelled)
                if time.monotonic() >= deadline:
                    raise DocumentError("轮廓修复超时，请缩小处理区域后重试。", "repair_timeout")
                try:
                    process.communicate(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    continue
            if process.returncode:
                raise DocumentError("轮廓修复失败，请检查安装组件后重试。", "repair_failed")
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
    _check_cancelled(cancelled)


def restore_photo_structure(rgb, excluded, cancelled=lambda: False):
    """Extend observed contours without copying unrelated facial features.

    The caller provides a bounded crop and excludes every watermark/transparent
    pixel in it, not just the connected stroke being committed to the result.
    """
    _check_cancelled(cancelled)
    missing = excluded != 0
    if not np.any(missing):
        return rgb.copy()
    if np.all(missing) or excluded.size > MAX_STRUCTURE_PIXELS:
        raise DocumentError("轮廓修复区域过大或缺少干净背景，请缩小选区。", "repair_limit")
    executable = executable_path()
    if not executable.is_file():
        raise DocumentError("轮廓修复组件缺失，请重新安装完整版本。", "repair_unavailable")
    try:
        with tempfile.TemporaryDirectory(prefix="clearcanvas-structure-") as temporary:
            directory = Path(temporary)
            source = rgb.copy()
            source[missing] = 0
            with Image.fromarray(source) as image:
                image.save(directory / "source.ppm")
            with Image.fromarray(missing.astype(np.uint8)) as image:
                image.save(directory / "mask.pgm")
            _run_structure(executable, directory, cancelled)
            with Image.open(directory / "result.ppm") as image:
                if image.size != (rgb.shape[1], rgb.shape[0]) or image.mode != "RGB":
                    raise DocumentError("轮廓修复返回了无效图像。", "repair_failed")
                repaired = np.array(image)
    except OSError as error:
        raise DocumentError(
            "无法运行轮廓修复组件，请重新安装完整版本。", "repair_unavailable"
        ) from error
    repaired[~missing] = rgb[~missing]
    _check_cancelled(cancelled)
    return repaired
