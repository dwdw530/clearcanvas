"""Prepare a pinned, verified offline G'MIC runtime for development and builds."""

import hashlib
import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

from clearcanvas.structure_inpaint import GMIC_EXE_SHA256, GMIC_SHA256, GMIC_URL, GMIC_VERSION

ROOT = Path(__file__).resolve().parents[1]


def prepare_gmic(output=ROOT / "artifacts" / "build"):
    cache = output / "downloads" / f"gmic-{GMIC_VERSION}-win64.zip"
    data = cache.read_bytes() if cache.is_file() else None
    if data is None or hashlib.sha256(data).hexdigest() != GMIC_SHA256:
        with urlopen(GMIC_URL, timeout=60) as response:
            data = response.read(40_000_001)
        if hashlib.sha256(data).hexdigest() != GMIC_SHA256:
            raise RuntimeError("G'MIC archive checksum mismatch; no runtime was installed")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(data)
    runtime = output / "gmic"
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        prefix = f"gmic-{GMIC_VERSION}-cli-win64/"
        files = [entry for entry in archive.infolist() if not entry.is_dir()]
        for entry in files:
            if not entry.filename.startswith(prefix):
                raise RuntimeError("Unexpected G'MIC archive layout")
            name = entry.filename[len(prefix):]
            if not name or "/" in name or "\\" in name or ":" in name or name in {".", ".."}:
                raise RuntimeError("Unsafe G'MIC archive path")
        runtime.mkdir(parents=True, exist_ok=True)
        for entry in files:
            target = runtime / entry.filename[len(prefix):]
            payload = archive.read(entry)
            # Windows locks a running EXE/DLL against writes. Reusing an exactly
            # verified file also keeps preparation idempotent during local QA.
            if not target.is_file() or target.read_bytes() != payload:
                target.write_bytes(payload)
    executable = runtime / "gmic.exe"
    if hashlib.sha256(executable.read_bytes()).hexdigest() != GMIC_EXE_SHA256:
        raise RuntimeError("G'MIC executable checksum mismatch")
    licenses = output / "licenses" / "gmic"
    licenses.mkdir(parents=True, exist_ok=True)
    for name in ("COPYING", "README"):
        (licenses / name).write_bytes((runtime / name).read_bytes())
    (licenses / "SOURCE.txt").write_text(
        f"G'MIC {GMIC_VERSION}, unmodified upstream Windows CLI distribution.\n"
        f"Archive: {GMIC_URL}\nSHA256: {GMIC_SHA256}\n"
        f"Corresponding source: https://github.com/GreycLab/gmic/tree/v.{GMIC_VERSION}\n"
        "The runtime is invoked as a separate process. See COPYING for CeCILL/CeCILL-C terms.\n",
        encoding="utf-8",
    )
    print(f"G'MIC {GMIC_VERSION} runtime verified: {runtime}")
    return runtime


if __name__ == "__main__":
    prepare_gmic()
