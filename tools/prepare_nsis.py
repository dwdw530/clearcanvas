"""Fetch the pinned portable installer compiler without changing the system."""

import hashlib
import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

VERSION = "3.11"
URL = "https://downloads.sourceforge.net/project/nsis/NSIS%203/3.11/nsis-3.11.zip"
SHA256 = "c7d27f780ddb6cffb4730138cd1591e841f4b7edb155856901cdf5f214394fa1"
ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "build"


def main():
    cache = ROOT / "downloads" / f"nsis-{VERSION}.zip"
    data = cache.read_bytes() if cache.is_file() else None
    if data is None or hashlib.sha256(data).hexdigest() != SHA256:
        with urlopen(URL, timeout=60) as response:
            data = response.read(30_000_001)
        if hashlib.sha256(data).hexdigest() != SHA256:
            raise RuntimeError("NSIS archive checksum mismatch; nothing was installed")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(data)
    destination = (ROOT / "nsis").resolve()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for entry in archive.infolist():
            if not (destination / entry.filename).resolve().is_relative_to(destination):
                raise RuntimeError("Unsafe NSIS archive path")
            if not entry.filename.startswith(f"nsis-{VERSION}/"):
                raise RuntimeError("Unexpected NSIS archive layout")
        archive.extractall(destination)
    compiler = destination / f"nsis-{VERSION}" / "makensis.exe"
    if not compiler.is_file():
        raise RuntimeError("The NSIS compiler is missing")
    print(f"NSIS {VERSION} verified: {compiler}")


if __name__ == "__main__":
    main()
