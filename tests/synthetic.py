"""Synthetic photographs with a known centerline.

The real experiment photographs cannot serve as a regression baseline for
*accuracy*, because the true centerline is unknown -- only another program's
estimate of it is.  These images have an analytic centerline, so the tests can
assert an absolute error budget instead of comparing against a previous output.
"""

from __future__ import annotations

import numpy as np

BACKGROUND = (196, 188, 182)  # cool light grey, like the curtain in the lab
PAINT = (238, 196, 58)        # rod body
CARBON = (26, 26, 28)         # bare tip


def parabola(samples: int = 1200, width: int = 4096, height: int = 3072):
    """A 3.5 m rod clamped on the right, sagging toward the left.

    The geometry is defined in fractions of the frame, so the same rod appears on
    canvases of other sizes.  Returns the analytic centerline as ``(x, y)`` and
    the anchor pair the user would click: the clamp exit and the tip.
    """
    x = np.linspace(0.928 * width, 0.078 * width, samples)
    t = np.linspace(0.0, 1.0, samples)           # 0 at the butt, 1 at the tip
    y = height * (520.0 + 60.0 * t + 780.0 * t ** 2) / 3072.0
    return np.column_stack((x, y)), np.array([[x[0], y[0]], [x[-1], y[-1]]])


def render(centerline: np.ndarray, width: int = 4096, height: int = 3072,
           paint_fraction: float = 0.72) -> np.ndarray:
    """Draw a tapered rod: painted over the first part, bare carbon at the tip.

    The painted section is stamped as discs of a linearly tapering radius, which
    is both faster and less biased than connecting successive points with short
    line segments: every disc is centred exactly on the analytic centerline, so
    the ground truth stays exact.
    """
    import cv2

    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = BACKGROUND
    count = len(centerline)
    split = int(count * paint_fraction)
    points = np.rint(centerline).astype(np.int32)

    pitch = width / 4096.0                   # keep the rod's relative width
    for index in range(split):
        along = index / max(1, split - 1)
        radius = (11.0 - 7.5 * along) * pitch   # 22 px at the butt, 7 px at the swap
        cv2.circle(image, tuple(points[index]), int(round(radius)), PAINT, -1, cv2.LINE_AA)
    # Bare tip: a two pixel carbon line.
    cv2.polylines(image, [points[split:]], False, CARBON, 2, cv2.LINE_AA)
    return image


def distance_to_polyline(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Shortest distance from each point to a polyline (dense sampling)."""
    dense = []
    for a, b in zip(polyline, polyline[1:]):
        step = max(1, int(np.linalg.norm(b - a)))
        dense.append(np.linspace(a, b, step))
    dense = np.concatenate(dense)
    return np.array([np.min(np.linalg.norm(dense - p, axis=1)) for p in points])
