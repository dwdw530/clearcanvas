"""Reproduce user inputs unchanged and save actual exports plus quality evidence."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from clearcanvas import __version__
from clearcanvas.documents import render_document
from clearcanvas.photo_watermarks import detect_photo_watermark_mask
from clearcanvas.repair import auto_repair_document, export_result, source_stamp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--detail", type=int, nargs=4, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"))
    parser.add_argument("paths", type=Path, nargs="+")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for path in args.paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        start = time.monotonic()
        result = auto_repair_document(render_document(path), source_stamp(path))
        try:
            destination = args.output / (path.stem + "-result.png")
            export_result(result, destination, overwrite=args.overwrite)
            with Image.open(path).convert("RGBA") as original:
                before = np.array(original)
                after = np.array(result.image)
                changed = np.any(before != after, axis=2)
                ink = np.all(before[:, :, :3] == (255, 0, 100), axis=2)
                remaining = np.all(after[:, :, :3] == (255, 0, 100), axis=2)
                mask = detect_photo_watermark_mask(original)
                outside_unchanged = np.array_equal(before[mask == 0], after[mask == 0])
                assert outside_unchanged
                with Image.open(destination).convert('RGBA') as exported:
                    assert np.array_equal(np.array(exported), after)
                Image.fromarray(mask).save(args.output / (path.stem + '-mask.png'))
                record = {
                    "version": __version__,
                    "input": str(path.resolve()), "sha256": digest,
                    "output": str(destination.resolve()), "size": original.size,
                    "changed_pixels": int(changed.sum()),
                    "mask_pixels": int(np.count_nonzero(mask)),
                    "magenta_input_pixels": int(ink.sum()),
                    "magenta_remaining_pixels": int(remaining.sum()),
                    "outside_mask_unchanged": outside_unchanged,
                    "export_matches_result": True,
                    "seconds": round(time.monotonic() - start, 2),
                }
                if args.baseline:
                    with Image.open(args.baseline / destination.name).convert("RGBA") as previous:
                        record["same_as_baseline"] = np.array_equal(np.array(previous), after)
                width = 1100
                height = round(original.height * width / original.width)
                comparison = Image.new("RGB", (width * 2, height + 36), "white")
                draw = ImageDraw.Draw(comparison)
                draw.text((12, 8), "UNCHANGED INPUT", fill="black", font_size=18)
                draw.text((width + 12, 8), "ACTUAL EXPORT", fill="black", font_size=18)
                comparison.paste(original.resize((width, height)), (0, 36))
                comparison.paste(result.image.resize((width, height)), (width, 36))
                comparison.save(args.output / (path.stem + "-comparison.jpg"))
                comparison.close()
                if args.detail:
                    box = tuple(args.detail)
                    width, height = box[2] - box[0], box[3] - box[1]
                    sources = [("INPUT", original), ("CURRENT", result.image)]
                    baseline = None
                    if args.baseline:
                        baseline = Image.open(args.baseline / (path.stem + "-result.png"))
                        sources.insert(1, ("PREVIOUS", baseline))
                    with Image.new("RGB", (width * len(sources), height + 32), "white") as detail:
                        draw = ImageDraw.Draw(detail)
                        for index, (label, source) in enumerate(sources):
                            draw.text((index * width + 8, 6), label, fill="black", font_size=18)
                            with source.crop(box) as cropped:
                                detail.paste(cropped, (index * width, 32))
                                detail_name = path.stem + f"-detail-{label.lower()}.png"
                                cropped.save(args.output / detail_name)
                        detail.save(args.output / (path.stem + "-detail.jpg"))
                    if baseline:
                        baseline.close()
                assert original.size == result.image.size
                assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
                records.append(record)
                print(json.dumps(record), flush=True)
        finally:
            result.image.close()
    (args.output / "report.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
