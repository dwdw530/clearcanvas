import base64
import hashlib
import json
from dataclasses import replace
from io import BytesIO

import cv2
import numpy as np
import pytest
from PIL import Image

from clearcanvas import ai_repair as ai
from clearcanvas.ai_settings import AiConfig
from clearcanvas.documents import DocumentError, render_document
from clearcanvas.repair import _export_pdf, batch_export, export_result, source_stamp


def response(image=None, *, raw=None):
    if raw is None:
        raw = ai.png_bytes(image)
    return json.dumps({"data": [{"b64_json": base64.b64encode(raw).decode()}]}).encode()


@pytest.fixture
def plan(sample_image):
    preview = render_document(sample_image)
    value = ai.prepare_plan(preview, source_stamp(sample_image), [(0.55, 0.65, 0.65, 0.7)], False)
    yield value
    value.close()


def test_crop_padding_mask_and_metadata(plan):
    assert plan.canvas.size == ai.CANVAS == plan.remote_mask.size
    x1, y1, x2, y2 = plan.content
    alpha = np.array(plan.remote_mask)[:, :, 3]
    outside = np.ones(alpha.shape, bool)
    outside[y1:y2, x1:x2] = False
    assert np.all(alpha[outside] == 255)
    assert np.any(alpha[y1:y2, x1:x2] == 0)
    image, mask = ai.encode_plan(plan, True)
    for data in (image, mask):
        with Image.open(BytesIO(data)) as decoded:
            assert decoded.size == ai.CANVAS
            assert not decoded.getexif()
            assert not decoded.info
    assert ai.encode_plan(plan, False)[1] is None


@pytest.mark.parametrize("size", [(200, 700), (2700, 1700), (2000, 2000), (700, 200)])
def test_mapping_truth_and_strict_rgba_protection(tmp_path, size):
    width, height = size
    yy, xx = np.mgrid[:height, :width]
    pixels = np.stack((xx % 256, yy % 256, (xx + yy) % 256, (xx * 3) % 256), axis=2).astype(
        np.uint8
    )
    path = tmp_path / "alpha.png"
    Image.fromarray(pixels).save(path)
    digest = hashlib.sha256(path.read_bytes()).digest()
    mask = np.zeros((height, width), np.uint8)
    mask[height // 3 : height // 3 + 20, width // 3 : width // 3 + 50] = 255
    mask[height - 40 : height - 20, width - 80 : width - 30] = 255
    with Image.fromarray(pixels) as image:
        plan = ai.plan_image(image, mask, preview=render_document(path), stamp=source_stamp(path))
    with Image.new("RGB", ai.CANVAS, (7, 111, 229)) as returned:
        result = ai.compose_result(plan, ai.AiValidated(returned, "uncertain"), AiConfig(), True)
    output = np.array(result.image)
    assert np.array_equal(output[mask == 0], pixels[mask == 0])
    assert np.array_equal(output[:, :, 3], pixels[:, :, 3])
    assert np.all(output[:, :, :3][mask != 0] == (7, 111, 229))
    dest = tmp_path / "result.png"
    export_result(result, dest)
    assert np.array_equal(np.array(Image.open(dest)), output)
    assert hashlib.sha256(path.read_bytes()).digest() == digest
    assert result.backend == "cloud_ai"
    assert "key" not in str(result.ai_metadata).lower()
    assert plan.downsampled == (max(size) > 1536)
    result.image.close()
    plan.close()


def test_inverse_transform_uses_content_not_padding(plan):
    returned = Image.new("RGB", ai.CANVAS, "red")
    x1, y1, x2, y2 = plan.content
    with Image.new("RGB", (x2 - x1, y2 - y1), (10, 90, 180)) as content:
        returned.paste(content, (x1, y1))
    result = ai.compose_result(plan, ai.AiValidated(returned, "aligned"), AiConfig())
    assert np.all(np.array(result.image)[:, :, :3][plan.mask != 0] == (10, 90, 180))
    returned.close()
    result.image.close()


def test_exif_orientation_and_manual_selection(tmp_path):
    path = tmp_path / "rotated.jpg"
    image = Image.new("RGB", (100, 200), "white")
    exif = Image.Exif()
    exif[274] = 6
    exif[270] = "PRIVATE"
    image.save(path, exif=exif)
    preview = render_document(path)
    plan = ai.prepare_plan(preview, source_stamp(path), [(0.1, 0.1, 0.2, 0.2)], False)
    assert plan.pixels.shape == (100, 200, 4)
    assert np.count_nonzero(plan.mask) == 200
    assert b"PRIVATE" not in ai.encode_plan(plan, False)[0]
    plan.close()


@pytest.mark.parametrize(
    "raw",
    [
        b"<html>oops</html>",
        b"[]",
        b"{}",
        b'{"error":{"message":"SECRET"}}',
        b'{"data":[]}',
        b'{"data":[{},{}]}',
        b'{"data":[{"url":"https://example.com/x"}]}',
        b'{"data":[{"b64_json":"abc!"}]}',
        b'{"data":[{"b64_json":""}]}',
        b'{"data":[{"b64_json":123}]}',
        b'{"data":["a"]}',
    ],
)
def test_bad_response_rejected_without_server_text(raw):
    with pytest.raises(DocumentError) as error:
        ai.decode_response(raw)
    assert "SECRET" not in str(error.value)


def test_explicit_policy_refusal():
    with pytest.raises(DocumentError) as error:
        ai.decode_response(b'{"error":{"code":"content_policy_violation"}}')
    assert error.value.code == "policy"


@pytest.mark.parametrize(
    "format,size",
    [("PNG", (100, 100)), ("JPEG", (1024, 1536)), ("GIF", ai.CANVAS), ("BMP", ai.CANVAS)],
)
def test_wrong_format_and_size(format, size):
    with Image.new("RGB", size, "white") as image, BytesIO() as stream:
        image.save(stream, format=format)
        with pytest.raises(DocumentError):
            ai.decode_response(response(raw=stream.getvalue()))


def test_animation_limits_and_external_url(plan, monkeypatch):
    with BytesIO() as stream:
        plan.canvas.save(
            stream, format="PNG", save_all=True, append_images=[Image.new("RGB", ai.CANVAS, "blue")]
        )
        with pytest.raises(DocumentError):
            ai.decode_response(response(raw=stream.getvalue()))
    raw = response(plan.canvas)
    body = json.loads(raw)
    body["data"][0]["url"] = "https://example.com/image.png"
    with pytest.raises(DocumentError):
        ai.decode_response(json.dumps(body).encode())
    monkeypatch.setattr(ai, "MAX_IMAGE_BYTES", 10)
    with pytest.raises(DocumentError):
        ai.decode_response(raw)
    monkeypatch.setattr(ai, "MAX_RESPONSE_BYTES", 10)
    with pytest.raises(DocumentError):
        ai.decode_response(raw)


def test_valid_png_jpeg_and_mask_outside_changed(plan):
    for fmt in ("PNG", "JPEG"):
        with BytesIO() as stream:
            plan.canvas.save(stream, format=fmt)
            image = ai.decode_response(response(raw=stream.getvalue()))
            assert image.size == ai.CANVAS
            image.close()
    validated = ai.validate_response(plan, response(plan.canvas))
    assert validated.alignment in {"aligned", "uncertain"}
    validated.image.close()


def test_geometry_shift_rejected_and_flat_content_requires_review(tmp_path):
    rng = np.random.default_rng(123)
    pixels = rng.integers(0, 256, (1024, 1536, 3), np.uint8)
    mask = np.zeros((1024, 1536), np.uint8)
    mask[10:30, 10:30] = 255
    mask[990:1010, 1500:1520] = 255
    with Image.fromarray(pixels) as image:
        plan = ai.plan_image(image, mask)
    assert ai.alignment_status(plan, plan.canvas) == "aligned"
    shifted = cv2.warpAffine(np.array(plan.canvas), np.float32([[1, 0, 22], [0, 1, 15]]), ai.CANVAS)
    with Image.fromarray(shifted) as image:
        assert ai.alignment_status(plan, image) == "rejected"
        with pytest.raises(DocumentError, match="位移"):
            ai.validate_response(plan, response(image))
    with Image.new("RGB", ai.CANVAS, "white") as image:
        assert ai.alignment_status(plan, image) == "uncertain"
    plan.close()
    with Image.new("RGB", ai.CANVAS, "white") as image:
        flat_plan = ai.plan_image(image, mask)
        assert ai.alignment_status(flat_plan, flat_plan.canvas) == "uncertain"
        flat_plan.close()


def test_uncertain_not_exportable_and_ai_lower_layer_guards(plan, tmp_path):
    checked = ai.AiValidated(plan.canvas.copy(), "uncertain")
    with pytest.raises(DocumentError, match="核对对齐"):
        ai.compose_result(plan, checked, AiConfig())
    result = ai.compose_result(plan, checked, AiConfig(), True)
    with pytest.raises(DocumentError, match="批量"):
        batch_export(result, [], tmp_path, "all", lambda: False, lambda *a: None)
    with pytest.raises(DocumentError, match="PDF"):
        _export_pdf(result, tmp_path / "bad.pdf", None, lambda: False, lambda *a: None)
    result.original = replace(result.original, info=replace(result.original.info, kind="PDF"))
    with pytest.raises(DocumentError, match="PDF"):
        export_result(result, tmp_path / "bad.pdf")
    result.image.close()
    checked.image.close()


def test_source_change_cancel_no_watermark_and_area_limits(plan, tmp_path):
    checked = ai.AiValidated(plan.canvas.copy(), "aligned")
    result = ai.compose_result(plan, checked, AiConfig())
    plan.original.info.path.write_bytes(b"changed")
    for op in (
        lambda: ai.encode_plan(plan, False),
        lambda: ai.compose_result(plan, checked, AiConfig()),
        lambda: export_result(result, tmp_path / "out.png"),
    ):
        with pytest.raises(DocumentError):
            op()
    assert not (tmp_path / "out.png").exists()
    with Image.new("RGB", (100, 100), "white") as image:
        for mask in (np.zeros((100, 100), np.uint8), np.ones((100, 100), np.uint8)):
            with pytest.raises(DocumentError):
                ai.plan_image(image, mask)
    with pytest.raises(DocumentError) as error:
        ai.encode_plan(plan, False, lambda: True)
    assert error.value.code == "cancelled"
    result.image.close()
    checked.image.close()


def test_cloud_pdf_never_loads_image(sample_pdf):
    with pytest.raises(DocumentError, match="PDF"):
        ai.prepare_plan(render_document(sample_pdf), source_stamp(sample_pdf), [], True)
