"""CSV, overlay image and curve plot writers."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .tracing import Centerline

OVERLAY_COLOR = (0, 255, 197)
REVIEW_COLOR = (255, 159, 20)
CLAMP_COLOR = (200, 143, 255)


def write_csv(path: Path, result: Centerline, physical, clamp_angle=None) -> None:
    """Write one row per sample.

    Pixel columns are always present; millimetre columns appear once a scale is
    known, and every row carries the same clamp angle so a single CSV is enough
    to drive the mechanical model.
    """
    headings = ["s_px", "x_px", "y_px", "image_u_px", "image_v_px", "evidence", "needs_review"]
    values = np.column_stack((result.arc_px, result.coordinates_px, result.pixels,
                              result.evidence, result.review))
    if physical is not None:
        headings.extend(("s_mm", "x_mm", "y_mm"))
        values = np.column_stack((values, physical))
    if clamp_angle is not None:
        headings.append("clamp_angle_deg")
        values = np.column_stack((values, np.full(len(values), clamp_angle)))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(headings)
        for row in values:
            writer.writerow([str(int(v)) if i == 6 else f"{v:.5f}" for i, v in enumerate(row)])


def render_overlay(image: np.ndarray, result: Centerline, clamp_points=None) -> np.ndarray:
    """Draw the fitted centerline, the review points and the root/tip markers."""
    canvas = image.copy()
    thickness = max(2, round(image.shape[1] / 1350))
    dense = np.rint(result.fitted).astype(np.int32)
    cv2.polylines(canvas, [dense], False, OVERLAY_COLOR, thickness, cv2.LINE_AA)
    if clamp_points is not None:
        back, front = map(tuple, np.rint(clamp_points).astype(int))
        cv2.line(canvas, back, front, CLAMP_COLOR, thickness, cv2.LINE_AA)
        cv2.circle(canvas, back, thickness * 2, CLAMP_COLOR, -1, cv2.LINE_AA)
        cv2.circle(canvas, front, thickness * 2, CLAMP_COLOR, -1, cv2.LINE_AA)
    for point in np.rint(result.pixels[result.review]).astype(np.int32):
        cv2.circle(canvas, tuple(point), thickness + 1, REVIEW_COLOR, -1, cv2.LINE_AA)
    for label, point in (("ROOT", result.anchors[0]), ("TIP", result.anchors[-1])):
        x, y = np.rint(point).astype(int)
        cv2.circle(canvas, (x, y), thickness * 3, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), thickness * 3, (17, 119, 126), 1, cv2.LINE_AA)
        scale = max(0.5, image.shape[1] / 3600)
        size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        origin = (max(0, min(x + 16, image.shape[1] - size[0] - 4)), max(size[1] + 4, y - 12))
        cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 28, 40), thickness + 3, cv2.LINE_AA)
        cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)
    return canvas


def save_curve_plot(path: Path, result: Centerline, physical) -> None:
    """Plot the centerline with equal aspect ratio, plus the tip drop."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if physical is None:
        xy, unit = result.coordinates_px, "px"
    else:
        xy, unit = physical[:, 1:], "mm"
    figure = Figure(figsize=(9, 4.3), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.set_title("Fishing rod centerline", loc="left")
    if physical is not None:
        tip_x, tip_y = xy[-1]
        figure.suptitle(f"Tip vertical distance to clamp level: {abs(tip_y):.1f} mm", fontsize=12)
        axes.axhline(0, color="#768690", linestyle="--", linewidth=0.8)
        axes.vlines(tip_x, min(0, tip_y), max(0, tip_y), color="#768690",
                    linestyle="--", linewidth=0.8)
    axes.plot(xy[:, 0], xy[:, 1], color="#176b77", linewidth=1.7)
    if result.review.any():
        points = xy[result.review]
        axes.scatter(points[:, 0], points[:, 1], s=12, color="#c36e1b", zorder=3, label="Review")
        axes.legend(loc="lower left", frameon=False)
    axes.set(xlabel=f"x ({unit})", ylabel=f"y ({unit})", aspect="equal")
    axes.grid(True, color="#e2e7ec", linewidth=0.7)
    axes.spines[["top", "right"]].set_visible(False)
    figure.savefig(path, dpi=160)


def save_overlay_png(path: Path, canvas: np.ndarray, compress_level: int = 6) -> None:
    """Write the overlay.

    ``compress_level`` is exposed because the overlay is a full-resolution frame:
    level 1 typically halves the write time at twice the file size, which matters
    when a session produces dozens of them.
    """
    Image.fromarray(canvas).save(path, compress_level=compress_level)
