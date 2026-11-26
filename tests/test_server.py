"""The local web server, exercised in-process on an ephemeral port."""

from __future__ import annotations

import csv
import io
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
from PIL import Image

from rodcurve.server import RodHandler


@pytest.fixture(scope="module")
def server():
    from .synthetic import parabola, render

    truth, anchors = parabola(samples=600, width=2048, height=1536)
    image = render(truth, width=2048, height=1536)
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    payload = buffer.getvalue()

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), RodHandler)
    httpd.startup_photo = None
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield base, payload, anchors.tolist()
    httpd.shutdown()
    httpd.server_close()


def get(url: str):
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.status, response.headers.get_content_type(), response.read()


def post(url: str, body: bytes, headers: dict | None = None):
    request = urllib.request.Request(url, data=body, method="POST", headers=headers or {})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.status, response.headers.get_content_type(), response.read()


def test_pages_are_served(server):
    base, _, _ = server
    for path, kind in (("/", "text/html"), ("/app.js", "text/javascript"), ("/model", "text/html")):
        status, content_type, body = get(base + path)
        assert status == 200
        assert content_type.startswith(kind)
        assert len(body) > 500


def test_startup_without_photo(server):
    base, _, _ = server
    status, _, body = get(base + "/startup")
    assert status == 200
    assert json.loads(body) == {"photo": None}


def test_preview_keeps_the_working_pixel_grid(server):
    base, payload, _ = server
    status, content_type, body = post(base + "/preview", payload)
    assert status == 200 and content_type == "image/jpeg"
    with Image.open(io.BytesIO(body)) as preview:
        assert preview.size == (2048, 1536)


def test_extract_returns_the_documented_payload(server):
    base, payload, anchors = server
    status, _, body = post(base + "/extract", payload, {"X-Rod-Anchors": json.dumps(anchors)})
    assert status == 200
    data = json.loads(body)
    assert set(data) == {"samples", "fitted", "lengthPx", "reviewCount", "imageSize"}
    assert data["imageSize"] == [2048, 1536]
    sample = data["samples"][0]
    assert set(sample) == {"s", "x", "y", "u", "v", "evidence", "review"}
    assert data["samples"][0]["s"] == pytest.approx(0.0)
    assert data["samples"][-1]["s"] == pytest.approx(data["lengthPx"])
    assert len(data["fitted"]) > len(data["samples"])
    assert 0 <= data["reviewCount"] <= len(data["samples"])


def test_bad_anchors_are_rejected_as_json(server):
    base, payload, anchors = server
    with pytest.raises(urllib.error.HTTPError) as error:
        post(base + "/extract", payload, {"X-Rod-Anchors": json.dumps([anchors[0]])})
    assert error.value.code == 400
    assert "参照点" in json.loads(error.value.read())["error"]


def test_empty_body_is_rejected(server):
    base, _, _ = server
    with pytest.raises(urllib.error.HTTPError) as error:
        post(base + "/preview", b"")
    assert error.value.code == 400


def test_unknown_path_is_404(server):
    base, _, _ = server
    with pytest.raises(urllib.error.HTTPError) as error:
        get(base + "/nope")
    assert error.value.code == 404
