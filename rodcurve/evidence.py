"""Image evidence: where a pixel looks like a rod body or a thin dark line.

A fishing rod in a side view offers two very different appearances:

* the butt and middle sections are painted, low-saturation-to-yellow, and
  several pixels wide;
* the tip is a bare, dark, one-to-two-pixel carbon line.

One scoring function has to accept both.  The body term rewards yellow pixels
that sit deep inside a yellow region (so a broad yellow object such as a wall or
a bucket edge scores far less than a rod of the same colour); the line term
rewards a pixel that is darker than *both* of its flanks, which rejects a single
high-contrast boundary such as a curtain edge.
"""

from __future__ import annotations

import cv2
import numpy as np

# HSV window of the painted rod body.  Hue is expressed on OpenCV's 0-179 scale.
YELLOW_LOW = (9, 58, 65)
YELLOW_HIGH = (43, 255, 255)

# Half-widths (in reference pixels) of the flank pairs used by the line detector.
LINE_RADII_PX = (2, 4, 7, 11)

# Contrast (grey levels) below which a dip is not called a line, and the
# contrast that saturates the score.
LINE_CONTRAST_FLOOR = 3.0
LINE_CONTRAST_SPAN = 24.0
LINE_WEIGHT = 0.88
BODY_BASE = 0.55
BODY_DEPTH_WEIGHT = 0.45
BODY_DEPTH_SCALE = 6.0


def image_evidence(strip: np.ndarray, pitch: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score every pixel of a sampling strip.

    Parameters
    ----------
    strip
        RGB strip with rows running along the rod and columns across it.
    pitch
        Reference pixel size, see :func:`rodcurve.geometry.pixel_pitch`.

    Returns
    -------
    score, yellow mask, blurred grey
    """
    hsv = cv2.cvtColor(strip, cv2.COLOR_RGB2HSV)
    yellow = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)
    depth = cv2.distanceTransform(yellow, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    body = (yellow > 0) * (BODY_BASE + BODY_DEPTH_WEIGHT * np.clip(depth / BODY_DEPTH_SCALE, 0, 1))

    grey = cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY).astype(np.float32)
    grey = cv2.GaussianBlur(grey, (3, 3), 0.65)
    contrast = np.zeros_like(grey)
    for radius_px in LINE_RADII_PX:
        radius = max(1, round(radius_px / pitch))
        if 2 * radius >= grey.shape[1]:
            continue
        centre = grey[:, radius:-radius]
        flanks = np.minimum(grey[:, :-2 * radius], grey[:, 2 * radius:])
        np.maximum(contrast[:, radius:-radius], flanks - centre, out=contrast[:, radius:-radius])
    line = LINE_WEIGHT * np.clip((contrast - LINE_CONTRAST_FLOOR) / LINE_CONTRAST_SPAN, 0, 1)
    return np.maximum(body, line).astype(np.float32), yellow > 0, grey
