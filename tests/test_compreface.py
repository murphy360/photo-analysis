import io

import httpx
import pytest
from PIL import Image

from app.integrations.compreface import CompreFaceClient, _crop_face


_RealAsyncClient = httpx.AsyncClient


def _mock_client(handler):
    """Monkeypatches httpx.AsyncClient (as used inside CompreFaceClient) to
    always route through a MockTransport — exercises the real httpx request/
    response machinery (headers, raise_for_status, JSON parsing) against a
    fake server response, rather than hand-rolling fake response objects.
    Calls the real class captured above, not httpx.AsyncClient itself —
    that name is what's being patched, so using it here would recurse."""

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return _RealAsyncClient(*args, **kwargs)

    return factory


def _fake_client() -> CompreFaceClient:
    client = CompreFaceClient()
    client._base_url = "http://fake-compreface"
    client._api_key = "test-key"
    return client


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


async def test_recognize_treats_no_face_response_as_empty_result(monkeypatch):
    """CompreFace responds with HTTP 400 + error code 28 for a completely
    normal case (no detectable face in the photo, e.g. one facing away from
    the camera) — this must come back as an empty list, not an exception."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400, json={"message": "No face is found in the given image", "code": 28}
        )

    monkeypatch.setattr(
        "app.integrations.compreface.httpx.AsyncClient", _mock_client(handler)
    )

    result = await _fake_client().recognize(_jpeg(50, 50))
    assert result == []


async def test_recognize_still_raises_on_a_genuine_error(monkeypatch):
    """A different 400 (or any other error status) must still propagate —
    only the specific "no face found" code is treated as a non-error."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"message": "Some other problem", "code": 1})

    monkeypatch.setattr(
        "app.integrations.compreface.httpx.AsyncClient", _mock_client(handler)
    )

    with pytest.raises(httpx.HTTPStatusError):
        await _fake_client().recognize(_jpeg(50, 50))
