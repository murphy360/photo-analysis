import io

from PIL import Image

from app.integrations.compreface import _crop_face


def _jpeg(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color="blue").save(buf, format="JPEG")
    return buf.getvalue()


def test_crop_face_stays_within_image_bounds():
    image_bytes = _jpeg(100, 100)
    # Box near the edge: padding must clamp to the image, not go negative or
    # past the far edge (both would raise / produce a malformed crop).
    box = {"x_min": 90, "y_min": 90, "x_max": 100, "y_max": 100}

    cropped_bytes = _crop_face(image_bytes, box)
    cropped = Image.open(io.BytesIO(cropped_bytes))
    assert cropped.width > 0
    assert cropped.height > 0
    assert cropped.width <= 100
    assert cropped.height <= 100


def test_crop_face_pads_around_a_central_box():
    image_bytes = _jpeg(200, 200)
    box = {"x_min": 80, "y_min": 80, "x_max": 120, "y_max": 120}

    cropped_bytes = _crop_face(image_bytes, box)
    cropped = Image.open(io.BytesIO(cropped_bytes))
    # 40x40 box with 15% padding each side should end up noticeably larger
    # than the raw box but still smaller than the whole frame.
    assert cropped.width > 40
    assert cropped.width < 200


def test_crop_face_without_a_box_returns_original_bytes():
    image_bytes = _jpeg(50, 50)
    assert _crop_face(image_bytes, None) == image_bytes
