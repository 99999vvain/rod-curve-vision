"""Small geometry helpers shared by tracing and calibration."""

from __future__ import annotations

import numpy as np


def cumulative_length(points: np.ndarray) -> np.ndarray:
    """Cumulative arc length of a polyline, starting at 0."""
    points = np.asarray(points, dtype=float)
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def pixel_pitch(image: np.ndarray) -> float:
    """Length of one image pixel relative to the 4096-px reference frame.

    All scale-dependent constants in this package are quoted in millimetres at
    the tolerance they were tuned for on a 4096 px wide photograph.  Expressing
    them through this pitch keeps the behaviour identical on smaller images
    instead of silently changing the search width in pixel terms.
    """
    return max(1.0, max(image.shape[:2]) / 4096)


def segment_frame(start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Unit tangent, unit normal and chord length between two anchor points."""
    chord = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    distance = float(np.linalg.norm(chord))
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("相邻参照点重合，无法建立追踪方向。")
    tangent = chord / distance
    normal = np.array([-tangent[1], tangent[0]])
    return tangent, normal, distance
