"""Turning pixels into millimetres, and measuring the clamp angle."""

from __future__ import annotations

import numpy as np
import cv2

from .geometry import cumulative_length
from .tracing import Centerline, validate_anchors


def parse_length_mm(value) -> float:
    """Validate a user-supplied free-span length."""
    try:
        length = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("曲线总长度必须是以毫米为单位的正数。") from error
    if not np.isfinite(length) or length <= 0:
        raise ValueError("曲线总长度必须是以毫米为单位的正数。")
    return length


def measure_clamp_angle(image: np.ndarray, back, front, homography=None) -> float:
    """Inclination from the handle side toward the tip side, upward positive.

    ``back`` is the point nearer the butt, ``front`` the point nearer the tip.
    With a planar calibration the angle is measured in world coordinates, so a
    tilted camera does not leak into the clamp angle.
    """
    points = validate_anchors(image, [back, front])
    if homography is None:
        dx = points[1, 0] - points[0, 0]
        dy = points[0, 1] - points[1, 1]
    else:
        matrix = np.asarray(homography, dtype=float)
        denominator = np.c_[points, np.ones(2)] @ matrix[2]
        if np.any(np.abs(denominator) < 1e-9) or np.sign(denominator[0]) != np.sign(denominator[1]):
            raise ValueError("投影在夹持段上不连续，请检查平面标定。")
        back_mm, front_mm = cv2.perspectiveTransform(points[None], matrix)[0]
        dx, dy = front_mm - back_mm
    return float(np.degrees(np.arctan2(dy, abs(dx))))


def physical_coordinates(result: Centerline, calibration, scale, length_mm=None):
    """Return ``(s, x, y)`` in millimetres plus a description of the calibration.

    Three mutually exclusive routes are supported: one uniform mm-per-pixel
    factor, one known free-span length, or a planar homography built from four or
    more reference points.  Returns ``(None, None)`` when the caller asked for
    pixel units only.
    """
    if calibration is None and scale is None and length_mm is None:
        return None, None

    if scale is not None or length_mm is not None:
        if length_mm is not None:
            length_mm = parse_length_mm(length_mm)
            scale = length_mm / result.arc_px[-1]
            info = {"mode": "known_arc_length", "input_length_mm": length_mm,
                    "mm_per_px": float(scale)}
        else:
            info = {"mode": "uniform_scale", "mm_per_px": float(scale)}
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("mm/px 比例必须为正数。")
        return np.column_stack((result.arc_px, result.coordinates_px)) * scale, info

    return _homography_coordinates(result, calibration)


def _homography_coordinates(result: Centerline, calibration):
    if not isinstance(calibration, dict) or not {"image_px", "world_mm"} <= calibration.keys():
        raise ValueError("标定文件必须包含 image_px 与 world_mm 两组坐标。")
    pixels = np.asarray(calibration["image_px"], dtype=float)
    world = np.asarray(calibration["world_mm"], dtype=float)
    if (pixels.ndim != 2 or pixels.shape[1] != 2 or pixels.shape != world.shape
            or len(pixels) < 4 or not np.isfinite([pixels, world]).all()):
        raise ValueError("平面标定需要至少四组一一对应的有限坐标。")
    # Reject repeated points and geometrically degenerate point sets.
    for points in (pixels, world):
        if len(np.unique(points, axis=0)) != len(points):
            raise ValueError("平面标定点不能重复。")
        if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
            raise ValueError("平面标定点不能共线。")
    matrix, _ = cv2.findHomography(pixels, world, method=0)
    if matrix is None or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) < 3:
        raise ValueError("无法计算平面标定矩阵，请检查点位及顺序。")
    projected = cv2.perspectiveTransform(pixels[None], matrix)[0]
    rmse = np.sqrt(np.mean(np.sum((projected - world) ** 2, axis=1)))
    denominator = np.c_[result.fitted, np.ones(len(result.fitted))] @ matrix[2]
    if np.any(np.abs(denominator) < 1e-9) or np.min(denominator) * np.max(denominator) < 0:
        raise ValueError("投影在曲线上不连续，请检查平面标定。")
    dense_mm = cv2.perspectiveTransform(result.fitted[None], matrix)[0]
    dense_arc = cumulative_length(result.fitted)
    arc_mm = np.interp(result.arc_px, dense_arc, cumulative_length(dense_mm))
    curve_mm = cv2.perspectiveTransform(result.pixels[None], matrix)[0]
    curve_mm -= curve_mm[0].copy()
    return np.column_stack((arc_mm, curve_mm)), {
        "mode": "planar_homography",
        "fit_rmse_mm": float(rmse),
        "homography": matrix.tolist(),
    }
