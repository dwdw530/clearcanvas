"""Inspect automatic repair of a user-supplied corner-watermark image."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from clearcanvas.detection import detect_watermark
from clearcanvas.documents import render_document
from clearcanvas.repair import auto_repair_document, export_result, source_stamp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(args.source.read_bytes()).hexdigest()
    with Image.open(args.source).convert("RGBA") as original:
        detection = detect_watermark(original)
        mask = detection.mask
        yy, xx = np.where(mask)
        box = (int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1)
        crop = (
            max(0, box[0] - 40), max(0, box[1] - 40),
            min(original.width, box[2] + 40), min(original.height, box[3] + 40),
        )
        result = auto_repair_document(render_document(args.source), source_stamp(args.source))
        try:
            export_result(result, args.output / "result.png", overwrite=True)
            before, after = np.array(original), np.array(result.image)
            assert np.array_equal(before[mask == 0], after[mask == 0])
            overlay = before.copy()
            overlay[mask != 0] = [255, 0, 100, 255]
            with Image.fromarray(overlay) as highlighted:
                panels = [original.crop(crop), highlighted.crop(crop), result.image.crop(crop)]
            width, height = panels[0].size
            with Image.new("RGB", (width, (height + 28) * 3), "#eef2f4") as comparison:
                draw = ImageDraw.Draw(comparison)
                labels = ("Original", "Detected strokes", "Result")
                for i, (label, panel) in enumerate(zip(labels, panels, strict=True)):
                    draw.text((8, i * (height + 28) + 6), label, fill="#24373e")
                    comparison.paste(panel, (0, i * (height + 28) + 28))
                    panel.close()
                comparison.save(args.output / "comparison.png")
            report = {
                "source": str(args.source), "sha256": digest, "size": original.size,
                "mask_pixels": int(np.count_nonzero(mask)), "bounds": box,
                "outside_mask_unchanged": True,
                "source_unchanged": hashlib.sha256(args.source.read_bytes()).hexdigest() == digest,
                "quality_limit": "No clean reference; original texture recovery is unverified.",
            }
            assert report["source_unchanged"]
            (args.output / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(json.dumps(report, ensure_ascii=False))
        finally:
            result.image.close()


if __name__ == "__main__":
    main()
