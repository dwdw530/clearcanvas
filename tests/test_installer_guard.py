"""Exercise the actual NSIS uninstall guard, without uninstalling the user's app."""

import ctypes
import os
import subprocess
import time
from contextlib import contextmanager
from ctypes import wintypes
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows installer integration")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def guard_bundle(tmp_path_factory):
    compiler = ROOT / "artifacts/build/nsis/nsis-3.11/makensis.exe"
    if not compiler.exists():
        pytest.skip("The local NSIS build tool is not installed")
    output = tmp_path_factory.mktemp("installer-guard")
    for name in ("installer_guard_probe.nsi", "installer_running_stub.nsi"):
        subprocess.run(
            [str(compiler), "-INPUTCHARSET", "UTF8", "-V2", f"-DQA_OUTPUT={output}",
             str(ROOT / "tests" / name)],
            check=True, capture_output=True, timeout=30,
        )
    subprocess.run([str(output / "guard-setup.exe"), "/S"], check=True, timeout=10)
    return output


@pytest.fixture
def probe(guard_bundle):
    marker = guard_bundle / "guard-passed.txt"
    marker.write_text("untouched", encoding="ascii")
    return guard_bundle, marker


@pytest.fixture
def running_stub(guard_bundle):
    # This disposable NSIS program has no user documents or application state.
    process = subprocess.Popen([str(guard_bundle / "ClearCanvas.exe")])
    time.sleep(0.2)
    assert process.poll() is None
    try:
        yield process
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=5)


@contextmanager
def start_guard(output, *, silent=False):
    process = subprocess.Popen(
        [str(output / "guard-uninstall.exe"), *(["/S"] if silent else []), f"_?={output}"]
    )
    try:
        yield process
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=5)


def dialog_for(process, contains="请先保存处理结果并退出软件"):
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
    user32.EnumChildWindows.argtypes = [wintypes.HWND, callback_type, wintypes.LPARAM]
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.PostMessageW.argtypes = [
        wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
    ]
    found = []

    @callback_type
    def visit(hwnd, _):
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value == process.pid:
            texts = []

            @callback_type
            def child(control, _):
                buffer = ctypes.create_unicode_buffer(1024)
                user32.GetWindowTextW(control, buffer, len(buffer))
                texts.append(buffer.value)
                return True

            user32.EnumChildWindows(hwnd, child, 0)
            if any(contains in text for text in texts):
                found.append(hwnd)
        return True

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        user32.EnumWindows(visit, 0)
        if found:
            return user32, found[0]
        assert process.poll() is None, f"Process exited without the expected text: {contains}"
        time.sleep(0.05)
    raise AssertionError(f"Expected a dialog containing: {contains}")


def test_uninstall_guard_allows_closed_app(probe):
    output, marker = probe
    with start_guard(output, silent=True) as process:
        assert process.wait(timeout=10) == 0
    assert marker.read_text() == "passed"


def test_silent_uninstall_stops_without_touching_files(probe, running_stub):
    output, marker = probe
    with start_guard(output, silent=True) as process:
        assert process.wait(timeout=10) == 2
    assert marker.read_text() == "untouched"
    assert running_stub.poll() is None


def test_cancel_uninstall_keeps_app_and_files(probe, running_stub):
    output, marker = probe
    with start_guard(output) as process:
        api, dialog = dialog_for(process)
        api.PostMessageW(dialog, 0x0111, 2, 0)  # WM_COMMAND / IDCANCEL
        assert process.wait(timeout=10) == 2
    assert marker.read_text() == "untouched"
    assert running_stub.poll() is None


def test_retry_waits_for_app_exit_before_continuing(probe, running_stub):
    output, marker = probe
    with start_guard(output) as process:
        api, dialog = dialog_for(process)
        api.PostMessageW(dialog, 0x0111, 4, 0)  # WM_COMMAND / IDRETRY
        time.sleep(0.2)
        api, dialog = dialog_for(process)
        assert marker.read_text() == "untouched"
        assert running_stub.poll() is None
        running_stub.terminate()
        running_stub.wait(timeout=5)
        api.PostMessageW(dialog, 0x0111, 4, 0)
        assert process.wait(timeout=10) == 0
    assert marker.read_text() == "passed"


def test_built_installer_directory_page_accepts_custom_path(tmp_path):
    from clearcanvas import __version__

    installer = ROOT / "dist" / f"ClearCanvas-{__version__}-Setup.exe"
    if not installer.exists():
        pytest.skip("Build the current installer before testing its directory page")
    startup = subprocess.STARTUPINFO()
    startup.dwFlags = subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 0
    process = subprocess.Popen([str(installer)], startupinfo=startup)
    try:
        api, window = dialog_for(process, "欢迎使用")
        api.PostMessageW(window, 0x0111, 1, 0)  # Next, from welcome to directory.
        api, window = dialog_for(process, "选择安装位置")
        api.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        api.SendMessageW.argtypes = [
            wintypes.HWND, wintypes.UINT, wintypes.WPARAM, ctypes.c_void_p
        ]
        edits = []
        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        @callback_type
        def collect_edit(hwnd, _):
            name = ctypes.create_unicode_buffer(128)
            api.GetClassNameW(hwnd, name, len(name))
            if name.value == "Edit":
                edits.append(hwnd)
            return True

        api.EnumChildWindows(window, collect_edit, 0)
        assert len(edits) == 1
        custom = tmp_path / "自选安装目录" / "ClearCanvas"
        text = ctypes.create_unicode_buffer(str(custom))
        api.SendMessageW(edits[0], 0x000C, 0, ctypes.cast(text, ctypes.c_void_p))
        actual = ctypes.create_unicode_buffer(1024)
        api.SendMessageW(edits[0], 0x000D, len(actual), ctypes.cast(actual, ctypes.c_void_p))
        assert actual.value == str(custom)
        # Cancel before installing: never replace the user's installation or registry.
        api.PostMessageW(window, 0x0111, 2, 0)
        api, warning = dialog_for(process, "退出")
        api.PostMessageW(warning, 0x0111, 6, 0)  # IDYES
        assert process.wait(timeout=10) == 1  # NSIS user-cancel exit code.
        assert not custom.exists()
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=5)
