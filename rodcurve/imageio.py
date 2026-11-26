"""Loading photographs and producing a preview for the browser front end."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# A side-view photograph of a 4 m rod needs roughly 1000 px per metre to resolve
# the tip, so full-resolution frames are kept by default.  The cap only guards
# against pathological phone output (e.g. 8000 px panoramas) that would make the
# search strip needlessly wide.
MAX_SIDE_PX = 4096


def load_image(path: str | Path) -> np.ndarray:
    """Read an image as an RGB array, honouring the EXIF orientation tag."""
    with Image.open(path) as source:
        return np.asarray(ImageOps.exif_transpose(source).convert("RGB"))


def decode_image(data: bytes, max_side: int | None = MAX_SIDE_PX) -> np.ndarray:
    """Decode encoded image bytes (as received by the local web server)."""
    with Image.open(BytesIO(data)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        return _cap_size(image, max_side)


def _cap_size(image: Image.Image, max_side: int | None) -> np.ndarray:
    if max_side and max(image.size) > max_side:
        scale = max_side / max(image.size)
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(size, Image.LANCZOS)
    return np.asarray(image)


def preview_bytes(image: np.ndarray, quality: int = 88) -> bytes:
    """Full-size JPEG used by the annotation canvas.

    The preview deliberately keeps the working image's own pixel grid.  The
    browser reports anchor clicks in bitmap pixels and the server traces the
    same array, so no rescaling sits between the click and the sample
    coordinates -- a preview downscaled for transfer would shift every anchor by
    up to a pixel and quietly bias the arc length.
    """
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()
