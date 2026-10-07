"""Download public examples and evaluate the installed local repair pipeline."""

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from time import perf_counter, sleep
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from clearcanvas import __version__
from clearcanvas.documents import DocumentError, render_document
from clearcanvas.repair import auto_repair_document, export_result, source_stamp

ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "web-samples-2026-10-06"
OUTPUT = ROOT
GITHUB = "https://raw.githubusercontent.com/braindotai/Watermark-Removal-Pytorch/master/"
SAMPLES = [
    {
        "id": "01-draft-document",
        "category": "Document: diagonal DRAFT over text",
        "page": "https://www.timeatlas.com/word-watermarks/",
        "url": "https://timeatlas.com/wp-content/uploads/sample-draft-background.jpg",
    },
    {
        "id": "02-logo-document",
        "category": "Document: large logo behind text",
        "page": "https://www.timeatlas.com/word-watermarks/",
        "url": "https://timeatlas.com/wp-content/uploads/image-watermark-example.jpg",
    },
    {
        "id": "03-confidential-document",
        "category": "Document: confidential watermark",
        "page": "https://www.locklizard.com/wp-content/uploads/2021/10/confidential-image-watermark3.png",
        "url": "https://www.locklizard.com/wp-content/uploads/2021/10/confidential-image-watermark3.png",
    },
    {
        "id": "04-dynamic-document",
        "category": "Document: dynamic watermark",
        "page": "https://www.locklizard.com/",
        "url": "https://www.locklizard.com/wp-content/uploads/2021/11/dynamic-watermark.png",
        "excluded": "Product diagram, not a watermarked source document",
    },
    {
        "id": "05-word-document",
        "category": "Document: Word tutorial screenshot",
        "page": "https://thesoftwarepro.com/",
        "url": "https://thesoftwarepro.com/wp-content/uploads/2018/10/insert-watermark-in-word-document-small.png",
        "excluded": "Standalone stamp graphic, not a watermark over content",
    },
    {
        "id": "06-photo-tiled",
        "category": "Photo: portrait with tiled watermark",
        "page": "https://www.fotor.com/features/remove-watermark/",
        "url": "https://imgv3.fotor.com/images/slider-image/fotor-watermark-remover-before-tiled-watermark-banner-image_2026-03-19-063437_aabk.jpg",
    },
    {
        "id": "07-photo-paired",
        "category": "Photo: public research watermarked example",
        "page": "https://github.com/braindotai/Watermark-Removal-Pytorch",
        "url": GITHUB + "outputs/watermark-available/watermarked-image.png",
        "reference_id": "09-clean-control",
    },
    {
        "id": "08-photo-tutorial",
        "category": "Photo: public before/after tutorial composite",
        "page": "https://www.aiseesoft.com/how-to/remove-watermark-from-photo.html",
        "url": "https://www.aiseesoft.com/images/how-to/remove-watermark-from-photo/remove-watermark-from-photo.jpg",
        "excluded": "Marketing before/after composite, not a source photo",
    },
    {
        "id": "09-clean-control",
        "category": "Negative control: unwatermarked reference",
        "page": "https://github.com/braindotai/Watermark-Removal-Pytorch",
        "url": GITHUB + "outputs/watermark-available/original-image.png",
        "control": True,
    },
    {
        "id": "10-photo-corner",
        "category": "Photo: photographer watermark near lower edge",
        "page": "https://visualwilderness.com/",
        "url": "https://cdn.visualwilderness.com/wp-content/uploads/2023/02/Lace-Andersen-Watermark-4.jpg",
    },
    {
        "id": "11-office-photo",
        "category": "Negative control: illustration without visible watermark",
        "page": "https://bbs.wps.cn/",
        "url": "https://bbs.wps.cn/cms/manage/file/8.jpg",
        "control": True,
    },
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contact_sheet(records, output_name, results=False):
    tiles = [record for record in records if record.get("file")]
    sheet = Image.new("RGB", (1200, ((len(tiles) + 2) // 3) * 380), "#eeeeee")
    draw = ImageDraw.Draw(sheet)
    for index, record in enumerate(tiles):
        x, y = (index % 3) * 400, (index // 3) * 380
        path = (
            OUTPUT / record["output"]
            if results and record.get("output") else ROOT / record["file"]
        )
        with Image.open(path) as source:
            with ImageOps.contain(source.convert("RGB"), (384, 302)) as thumb:
                sheet.paste(thumb, (x + (400 - thumb.width) // 2, y + 60))
        draw.text((x + 8, y + 8), record["id"], fill="#202020", font_size=18)
        label = record.get("status", "downloaded") if results else record["category"]
        draw.text((x + 8, y + 32), label, fill="#202020", font_size=12)
    sheet.save(OUTPUT / output_name)


def download():
    ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    for sample in SAMPLES:
        record = dict(sample)
        try:
            cached = next(
                (
                    ROOT / (sample["id"] + suffix)
                    for suffix in (".png", ".jpg")
                    if (ROOT / (sample["id"] + suffix)).exists()
                ),
                None,
            )
            if cached is not None:
                with Image.open(cached) as image:
                    record["size"] = list(image.size)
                    image.verify()
                record.update(
                    file=cached.name,
                    sha256=digest(cached),
                    bytes=cached.stat().st_size,
                    cached=True,
                )
                records.append(record)
                print(json.dumps(record, ensure_ascii=False), flush=True)
                continue
            request = Request(sample["url"], headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=40) as response:
                data = response.read(12 * 1024 * 1024 + 1)
                record["resolved_url"] = response.url
            if len(data) > 12 * 1024 * 1024:
                raise ValueError("Sample exceeds 12 MiB limit")
            with Image.open(BytesIO(data)) as image:
                if image.format not in ("PNG", "JPEG") or image.width * image.height > 40_000_000:
                    raise ValueError("Unsupported image format or dimensions")
                suffix = ".png" if image.format == "PNG" else ".jpg"
                record["size"] = list(image.size)
                image.verify()
            path = ROOT / (sample["id"] + suffix)
            if path.exists() and path.read_bytes() != data:
                raise ValueError("Remote asset changed; refusing to overwrite existing sample")
            path.write_bytes(data)
            record.update(file=path.name, sha256=digest(path), bytes=len(data))
        except Exception as error:
            record["download_error"] = str(error)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        records.append(record)
    (ROOT / "sources.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    contact_sheet(records, "sources-contact-sheet.jpg")


def evaluate():
    records = json.loads((ROOT / "sources.json").read_text(encoding="utf-8"))
    by_id = {record["id"]: record for record in records}
    source_root = Path(__file__).resolve().parents[1] / "src" / "clearcanvas"
    report = {
        "version": __version__,
        "date": datetime.now(timezone(timedelta(hours=8))).date().isoformat(),
        "cases": [],
        "source_sha256": {
            name: digest(source_root / name)
            for name in (
                "detection.py",
                "photo_watermarks.py",
                "repair.py",
                "pdf_watermarks.py",
                "editor.py",
            )
        },
    }
    for sample in records:
        record = dict(sample)
        if "file" not in sample:
            record["status"] = "download_failed"
            report["cases"].append(record)
            continue
        path = ROOT / sample["file"]
        started = perf_counter()
        result = None
        try:
            if digest(path) != sample["sha256"]:
                raise ValueError("Downloaded sample changed before testing")
            preview = render_document(path)
            result = auto_repair_document(preview, source_stamp(path))
            destination = OUTPUT / (sample["id"] + "-result.png")
            export_result(result, destination, overwrite=destination.exists())
            with Image.open(path).convert("RGBA") as source:
                original = np.array(source)
            with Image.open(destination) as exported:
                actual = np.array(exported.convert("RGBA"))
            changed = np.any(original != actual, axis=2)
            record.update(
                status="result_created_needs_visual_review",
                output=destination.name,
                changed_pixels=int(changed.sum()),
                changed_percent=round(float(changed.mean() * 100), 4),
                dark_pixels_unchanged=bool(
                    np.array_equal(
                        original[original[:, :, :3].max(axis=2) < 145],
                        actual[original[:, :, :3].max(axis=2) < 145],
                    )
                ),
            )
            highlight = original.copy()
            highlight[changed] = [255, 0, 100, 255]
            Image.fromarray(highlight).save(OUTPUT / (sample["id"] + "-changed.png"))
        except DocumentError as error:
            record.update(
                status="not_detected" if error.code == "no_watermark" else "error",
                message=str(error),
                error_code=error.code,
            )
        except Exception as error:
            record.update(status="error", message=str(error))
        finally:
            if result is not None:
                result.image.close()
        record["seconds"] = round(perf_counter() - started, 3)
        record["source_unchanged"] = digest(path) == sample["sha256"]
        reference = by_id.get(sample.get("reference_id"))
        if reference and reference.get("file"):
            with Image.open(path).convert("RGB") as source:
                original = np.array(source)
            with Image.open(ROOT / reference["file"]).convert("RGB") as clean:
                truth = np.array(clean)
            if original.shape == truth.shape:
                before = float(np.abs(original.astype(float) - truth).mean())
                record["reference_mae_before"] = round(before, 4)
                if record.get("output"):
                    with Image.open(OUTPUT / record["output"]).convert("RGB") as output:
                        after = float(np.abs(np.array(output).astype(float) - truth).mean())
                    record["reference_mae_after"] = round(after, 4)
                else:
                    record["reference_mae_after"] = round(before, 4)
            else:
                record["reference_comparison"] = (
                    "Not computed: published source and reference dimensions differ; "
                    "no rescaling or registration was applied"
                )
        report["cases"].append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    qualified = [
        case
        for case in report["cases"]
        if case.get("file") and not case.get("control") and not case.get("excluded")
    ]
    report["summary"] = {
        "qualified_watermarked_images": len(qualified),
        "not_detected": sum(case["status"] == "not_detected" for case in qualified),
        "result_created": sum(bool(case.get("output")) for case in qualified),
        "runtime_errors": sum(case["status"] == "error" for case in qualified),
        "controls": sum(bool(case.get("control")) for case in report["cases"]),
        "excluded": sum(bool(case.get("excluded")) for case in report["cases"]),
        "all_downloaded_sources_unchanged": all(
            case["source_unchanged"] for case in report["cases"] if case.get("file")
        ),
    }
    (OUTPUT / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    contact_sheet(report["cases"], "results-contact-sheet.jpg", results=True)
    contact_sheet(
        [record for record in report["cases"] if not record.get("excluded")],
        "acceptance-contact-sheet.jpg",
        results=True,
    )


def ui_probe():
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QMessageBox

    from clearcanvas.theme import apply_theme
    from clearcanvas.window import MainWindow

    app = QApplication([])
    apply_theme(app)
    messages = []
    original_warning = QMessageBox.warning
    QMessageBox.warning = lambda *args: messages.append(str(args[2])) or QMessageBox.Ok
    records = []

    def wait_for(condition):
        deadline = perf_counter() + 20
        while not condition():
            app.processEvents()
            if perf_counter() > deadline:
                raise TimeoutError("UI probe timed out")
            sleep(0.01)

    try:
        for filename in ("01-draft-document.jpg", "06-photo-tiled.jpg"):
            window = MainWindow()
            try:
                window.setAttribute(Qt.WA_DontShowOnScreen, True)
                window.resize(1200, 800)
                window.show()
                window.add_files([ROOT / filename])
                wait_for(lambda current=window: current._preview is not None)
                assert window.editor.auto_button.isEnabled()
                assert not window.canvas.regions
                window.editor.auto_button.click()
                wait_for(lambda current=window: not current.editor.busy)
                app.processEvents()
                screenshot = OUTPUT / (Path(filename).stem + "-ui.png")
                assert window.grab().save(str(screenshot))
                records.append(
                    {
                        "file": filename,
                        "status": window.status.text(),
                        "result_created": window.editor.result is not None,
                        "export_enabled": window.editor.export_button.isEnabled(),
                        "badge": window.badge.text(),
                        "screenshot": screenshot.name,
                        "messages": list(messages),
                    }
                )
                messages.clear()
            finally:
                window.editor.dirty = False
                window.close()
    finally:
        QMessageBox.warning = original_warning
        app.quit()
    (OUTPUT / "ui-results.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("download", "evaluate", "ui"))
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    if arguments.output_dir:
        OUTPUT = arguments.output_dir.resolve()
        OUTPUT.mkdir(parents=True, exist_ok=True)
    {"download": download, "evaluate": evaluate, "ui": ui_probe}[arguments.operation]()
