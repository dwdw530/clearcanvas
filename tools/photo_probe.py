"""Evaluate the photo detector on cached, unmodified acceptance images."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from clearcanvas.documents import DocumentError
from clearcanvas.photo_watermarks import detect_photo_watermark_mask
from clearcanvas.repair import _repair_image


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "artifacts/web-samples-2026-10-06"
    output = root / "artifacts/photo-probe"
    output.mkdir(parents=True, exist_ok=True)
    records = json.loads((source / "sources.json").read_text(encoding="utf-8"))
    for record in records:
        if "file" not in record:
            continue
        path = source / record["file"]
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        with Image.open(path).convert("RGBA") as image:
            try:
                mask = detect_photo_watermark_mask(image)
                with _repair_image(image, mask, photographic=True) as repaired:
                    repaired.save(output / (record["id"] + "-result.png"))
                overlay = np.array(image)
                overlay[mask != 0] = [255, 0, 100, 255]
                Image.fromarray(overlay).save(output / (record["id"] + "-mask.png"))
                print(record["id"], "selected", np.count_nonzero(mask), flush=True)
            except DocumentError as error:
                print(record["id"], error.code, flush=True)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before == record["sha256"]


if __name__ == "__main__":
    main()
