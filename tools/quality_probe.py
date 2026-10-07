"""Compare real repair output with known clean references; no product changes."""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw

from clearcanvas.documents import render_document
from clearcanvas.repair import export_result, repair_document, source_stamp

ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "quality"
WIDTH, HEIGHT = 640, 360
BOX = (240, 130, 400, 190)
REGIONS = [(BOX[0] / WIDTH, BOX[1] / HEIGHT, BOX[2] / WIDTH, BOX[3] / HEIGHT)]


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    y, x = np.mgrid[:HEIGHT, :WIDTH]
    flat = np.full((HEIGHT, WIDTH, 3), [185, 210, 224], dtype=np.uint8)
    gradient = np.stack((100 + x / 6, 160 + y / 5, 200 - x / 10), axis=-1).astype("uint8")
    texture = np.stack(
        (120 + 65 * np.sin(x / 5), 130 + 55 * np.cos(y / 7), 140 + 55 * np.sin((x + y) / 9)),
        axis=-1,
    ).astype("uint8")
    document = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(document)
    for top in (135, 163):
        draw.text((190, top), "Original document text 12345", fill="#252525", font_size=19)
    cases = [
        ("Solid background", Image.fromarray(flat)),
        ("Gradient", Image.fromarray(gradient)),
        ("Texture", Image.fromarray(texture)),
        ("Over document text", document),
    ]
    gallery = Image.new("RGB", (1080, 880), "#ffffff")
    labels = ImageDraw.Draw(gallery)
    for index, title in enumerate(("Watermarked", "Repaired", "Clean reference")):
        labels.text((index * 360 + 12, 12), title, fill="#202020", font_size=20)
    report = []
    for index, (title, clean) in enumerate(cases):
        marked = clean.convert("RGBA")
        overlay = Image.new("RGBA", clean.size)
        ImageDraw.Draw(overlay).text((253, 142), "SAMPLE", fill=(20, 30, 40, 150), font_size=28)
        marked = Image.alpha_composite(marked, overlay)
        source = ROOT / f"case-{index + 1}-source.png"
        marked.save(source)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        started = perf_counter()
        result = repair_document(render_document(source), REGIONS, source_stamp(source))
        output = export_result(result, ROOT / f"case-{index + 1}-result.png", overwrite=True)
        actual = np.array(result.image)[:, :, :3]
        original = np.array(clean)
        watermarked = np.array(marked)[:, :, :3]
        mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
        mask[130:190, 240:400] = True
        before = np.abs(watermarked[mask].astype(float) - original[mask]).mean()
        after = np.abs(actual[mask].astype(float) - original[mask]).mean()
        report.append(
            {
                "case": title,
                "seconds": round(perf_counter() - started, 3),
                "masked_error_before": round(float(before), 2),
                "masked_error_after": round(float(after), 2),
                "outside_unchanged": bool(np.array_equal(actual[~mask], watermarked[~mask])),
                "source_unchanged": digest == hashlib.sha256(source.read_bytes()).hexdigest(),
                "exported": output.is_file(),
            }
        )
        top = 55 + index * 205
        labels.text((12, top), title, fill="#202020", font_size=19)
        for column, image in enumerate((marked, result.image, clean)):
            crop = image.convert("RGB").crop((200, 110, 440, 220))
            crop = crop.resize((336, 154), Image.Resampling.NEAREST)
            gallery.paste(crop, (column * 360 + 12, top + 29))
    gallery.save(ROOT / "quality-check.png")
    (ROOT / "quality-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
