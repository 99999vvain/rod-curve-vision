"""Local web front end: the same interface as the shipped Windows build.

The browser does the annotation and the plotting; this server only decodes the
photograph, runs the tracer and returns JSON.  Everything stays on the machine
and nothing is uploaded anywhere.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from .imageio import decode_image, load_image, preview_bytes
from .tracing import extract_centerline

WEB_ROOT = Path(__file__).resolve().parent / "webui"
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/model": ("model.html", "text/html; charset=utf-8"),
}

# One-entry decode cache: the browser posts the same photograph once for the
# preview and once for the trace.  Decoding a 12 MP JPEG costs more than the
# whole tracing step, so the second request reuses the first decode.
_CACHE_LOCK = threading.Lock()
_CACHE: dict[str, np.ndarray] = {}


def decode_cached(data: bytes) -> np.ndarray:
    key = hashlib.sha1(data).hexdigest()
    with _CACHE_LOCK:
        if key in _CACHE:
            return _CACHE[key]
    image = decode_image(data)
    with _CACHE_LOCK:
        _CACHE.clear()
        _CACHE[key] = image
    return image


class RodHandler(BaseHTTPRequestHandler):
    server_version = "rodcurve"
    startup_photo: Path | None = None

    def log_message(self, fmt, *args):  # keep the console readable
        if "/extract" in fmt % args or "/preview" in fmt % args:
            return
        super().log_message(fmt, *args)

    # -- helpers ---------------------------------------------------------
    def _send(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK):
        self._send(json.dumps(payload).encode("utf-8"), "application/json", status)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            raise ValueError("请求没有附带照片数据。")
        return self.rfile.read(length)

    # -- routes ----------------------------------------------------------
    def do_GET(self):  # noqa: N802 - required by BaseHTTPRequestHandler
        path = self.path.split("?", 1)[0]
        if path in ASSETS:
            name, content_type = ASSETS[path]
            body = (WEB_ROOT / name).read_bytes()
            self._send(body, content_type)
            return
        if path == "/startup":
            photo = self.server.startup_photo
            self._send_json({"photo": photo.name if photo else None})
            return
        if path == "/initial-photo":
            photo = self.server.startup_photo
            if photo is None or not photo.exists():
                self._send_json({"error": "没有启动照片。"}, HTTPStatus.NOT_FOUND)
                return
            self._send(preview_bytes(load_image(photo)), "image/jpeg")
            return
        self._send_json({"error": "未知路径。"}, HTTPStatus.NOT_FOUND)

    def do_POST(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        try:
            data = self._read_body()
            if path == "/preview":
                self._send(preview_bytes(decode_cached(data)), "image/jpeg")
                return
            if path == "/extract":
                self._send_json(self._extract(data))
                return
        except ValueError as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            return
        except Exception as error:  # noqa: BLE001 - surface the reason to the UI
            self._send_json({"error": f"识别失败：{error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self._send_json({"error": "未知路径。"}, HTTPStatus.NOT_FOUND)

    def _extract(self, data: bytes) -> dict:
        try:
            anchors = json.loads(self.headers.get("X-Rod-Anchors") or "")
        except json.JSONDecodeError as error:
            raise ValueError("参照点格式无效。") from error
        if not isinstance(anchors, list) or len(anchors) < 2:
            raise ValueError("至少需要夹具出口和竿梢两个参照点。")
        image = decode_cached(data)
        result = extract_centerline(image, anchors)
        return {
            "samples": [
                {
                    "s": float(s), "x": float(x), "y": float(y),
                    "u": float(u), "v": float(v),
                    "evidence": float(e), "review": int(r),
                }
                for s, x, y, (u, v), e, r in zip(
                    result.arc_px, result.coordinates_px[:, 0], result.coordinates_px[:, 1],
                    result.pixels, result.evidence, result.review,
                )
            ],
            "fitted": result.fitted.tolist(),
            "lengthPx": float(result.arc_px[-1]),
            "reviewCount": int(result.review.sum()),
            "imageSize": [int(image.shape[1]), int(image.shape[0])],
        }


def serve(photo: Path | None = None, port: int = 8765, open_browser: bool = True) -> int:
    handler = type("Handler", (RodHandler,), {})
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    httpd.startup_photo = photo
    address = f"http://127.0.0.1:{httpd.server_address[1]}/"
    print(f"鱼竿曲线识别已启动：{address}")
    print("关闭此窗口即退出；照片和计算结果只在本机处理。")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(address)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出。")
    finally:
        httpd.server_close()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="rodcurve-web", description="启动本地图形界面。")
    parser.add_argument("photo", nargs="?", type=Path, help="启动时载入的照片，可省略")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args(argv)
    if args.photo is not None and not args.photo.exists():
        parser.error(f"照片不存在：{args.photo}")
    if not (WEB_ROOT / "index.html").exists():
        parser.error(f"找不到前端文件：{WEB_ROOT}")
    return serve(args.photo, args.port, not args.no_browser)


if __name__ == "__main__":
    raise SystemExit(main())
