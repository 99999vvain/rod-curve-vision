"""Centerline tracing: strip sampling, path search, cross-section refinement."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.interpolate import make_splprep

from .evidence import image_evidence
from .geometry import cumulative_length, pixel_pitch, segment_frame

# Fraction of the anchor-to-anchor distance used as the half-width of the search
# strip.  Wide enough for a strongly bent rod between two guide points, narrow
# enough to keep background structure out of the path search.
CORRIDOR = 0.16

# Lateral motion allowed per strip row, in reference pixels.  The rows advance
# along the chord, so this caps the tangent angle the path may follow.
MAX_SHIFT_PX = 3

# Penalty on the change of lateral shift between consecutive rows.  A physical
# rod changes curvature smoothly, whereas a fold in a curtain or a shadow edge
# enters and leaves the strip almost vertically, and a first-order path search
# cannot tell those apart because both stay inside the slope limit.
#
# Measured on the six reference cases in ``docs/validation.md``, a non-zero value
# did not help consistently: it reduced the worst deviation on one case
# (0.58 -> 0.34 px) and raised it on another (1.43 -> 14.7 px).  It is therefore
# off by default, and ``best_path`` keeps a first-order fast path for that case.
CURVATURE_PENALTY = 0.0

# Reward for staying near the middle of the strip.
CENTER_PRIOR = 0.055
# Reward for staying near the straight chord between the two anchors.
CHORD_PRIOR = 0.025

EVIDENCE_REVIEW_THRESHOLD = 0.16
FIT_REVIEW_PX = 2.5


@dataclass
class Centerline:
    """Result of :func:`extract_centerline`.

    Attributes
    ----------
    anchors
        User-supplied points in image pixels, from clamp exit to tip.
    raw
        Traced points before smoothing, one per strip row.
    fitted
        Dense smoothed polyline (used for drawing and for calibration).
    pixels
        Samples resampled at constant arc length, image pixel coordinates.
    arc_px
        Arc length of every sample, starting at zero.
    coordinates_px
        Samples relative to the clamp exit; ``y`` points up, ``x`` follows the
        direction from the clamp exit toward the tip.
    evidence
        Local image score of every sample, in ``[0, 1]``.
    review
        Boolean mask of samples a human should check.
    """

    anchors: np.ndarray
    raw: np.ndarray
    fitted: np.ndarray
    pixels: np.ndarray
    arc_px: np.ndarray
    coordinates_px: np.ndarray
    evidence: np.ndarray
    review: np.ndarray


def validate_anchors(image: np.ndarray, anchors) -> np.ndarray:
    """Check the user's anchor points and return them as a float array."""
    anchors = np.asarray(anchors, dtype=float)
    if anchors.ndim != 2 or anchors.shape[1] != 2 or len(anchors) < 2:
        raise ValueError("至少需要夹具出口和实际竿梢两个点。")
    height, width = image.shape[:2]
    if not np.isfinite(anchors).all() or (anchors < 0).any():
        raise ValueError("点位必须是照片内的有效像素坐标。")
    if (anchors[:, 0] > width - 1).any() or (anchors[:, 1] > height - 1).any():
        raise ValueError("有点位超出照片范围。")
    if (np.diff(cumulative_length(anchors)) < 8).any():
        raise ValueError("相邻点至少相距 8 像素，请撤销重复或过近的点。")
    return anchors


def _first_order_path(evidence: np.ndarray, offsets: np.ndarray, max_shift: int) -> np.ndarray:
    """Plain path search: no coupling between consecutive lateral shifts."""
    count, width = evidence.shape
    middle = width // 2
    shifts = np.arange(-max_shift, max_shift + 1, dtype=np.int32)
    columns = np.arange(width, dtype=np.int32)
    previous_columns = columns[None, :] - shifts[:, None]
    valid = (previous_columns >= 0) & (previous_columns < width)
    previous_columns = np.clip(previous_columns, 0, width - 1)
    span = max(1.0, abs(float(offsets[-1])))
    prior = CENTER_PRIOR * (shifts / max(1, max_shift)) ** 2
    displacement = CHORD_PRIOR * (offsets / span) ** 2
    history = np.empty((count, width), dtype=np.int8)
    previous = np.full(width, -np.inf)
    previous[middle] = 0.0

    for row in range(1, count):
        candidates = previous[previous_columns] - prior[:, None]
        candidates[~valid] = -np.inf
        choice = np.argmax(candidates, axis=0)
        history[row] = shifts[choice]
        previous = candidates[choice, columns] + evidence[row] - displacement

    if not np.isfinite(previous[middle]):
        raise ValueError("两点间找不到连续路径，请加一个沿竿身的引导点。")
    track = np.empty(count, dtype=np.int32)
    track[-1] = middle
    for row in range(count - 1, 0, -1):
        track[row - 1] = track[row] - history[row, track[row]]
    return track


def refine_centers(track: np.ndarray, yellow: np.ndarray, grey: np.ndarray, pitch: float) -> np.ndarray:
    """Replace ridge positions by the middle of the locally bounded cross section.

    The path search follows whichever side of the rod scores best, which leaves a
    systematic offset of about half the rod diameter.  Here each row is replaced
    by the midpoint between the two edges of the cross section that contains the
    tracked point: yellow edges where the body is painted, the dark band between
    two bright flanks on the bare tip.
    """
    centers = track.astype(float)
    reach = max(6, round(16 / pitch))
    near = max(1, round(3 / pitch))
    for row, offset in enumerate(track[1:-1], start=1):
        first = max(0, offset - reach)
        last = min(grey.shape[1], offset + reach + 1)
        local = offset - first
        if local < near or last - offset <= near:
            continue
        color = yellow[row, first:last]
        nearby_yellow = np.flatnonzero(color[local - near:local + near + 1]) + local - near
        if nearby_yellow.size:
            seed = nearby_yellow[np.argmin(abs(nearby_yellow - local))]
            edges = _band_edges(color, seed)
        else:
            profile = grey[row, first:last]
            seed = int(np.argmin(profile[local - near:local + near + 1])) + local - near
            if seed < 2 or len(profile) - seed < 3:
                continue
            baseline = min(
                np.percentile(profile[:seed], 80),
                np.percentile(profile[seed + 1:], 80),
            )
            contrast = baseline - profile[seed]
            if contrast < 9:
                continue
            edges = _band_edges(profile <= baseline - 0.42 * contrast, seed)
        if edges is not None:
            center = 0.5 * (edges[0] + edges[1])
            if abs(center - local) * pitch <= 7:
                centers[row] = first + center
    return centers


def _band_edges(mask: np.ndarray, seed: int) -> tuple[int, int] | None:
    left_breaks = np.flatnonzero(~mask[:seed])
    right_breaks = np.flatnonzero(~mask[seed + 1:])
    if not len(left_breaks) or not len(right_breaks):
        return None
    return int(left_breaks[-1] + 1), int(seed + right_breaks[0])


def best_path(
    evidence: np.ndarray,
    offsets: np.ndarray,
    max_shift: int,
    curvature_penalty: float = 0.0,
) -> np.ndarray:
    """Strongest continuous path through the strip with fixed ends.

    A dynamic program over ``(column, previous shift)`` states.  With
    ``curvature_penalty = 0`` the result equals a plain first-order path search,
    so that case takes the cheaper one-state-per-column route.  A positive value
    adds the cost of changing the lateral shift between rows, which a straight
    but slanted edge cannot afford while a curving rod can.
    """
    if not curvature_penalty:
        return _first_order_path(evidence, offsets, max_shift)

    count, width = evidence.shape
    middle = width // 2
    shifts = np.arange(-max_shift, max_shift + 1, dtype=np.int32)
    states = len(shifts)
    columns = np.arange(width, dtype=np.int32)
    span = max(1.0, abs(float(offsets[-1])))
    prior = CENTER_PRIOR * (shifts / max(1, max_shift)) ** 2
    displacement = CHORD_PRIOR * (offsets / span) ** 2
    change = (shifts[:, None] - shifts[None, :]).astype(np.float64) ** 2

    history = np.empty((count, width, states), dtype=np.int8)
    score = np.full((width, states), -np.inf, dtype=np.float64)
    score[middle, int(np.argmin(np.abs(shifts)))] = 0.0

    for row in range(1, count):
        updated = np.empty_like(score)
        for k, shift in enumerate(shifts):
            source = columns - shift
            valid = (source >= 0) & (source < width)
            gathered = score[np.clip(source, 0, width - 1), :]
            if curvature_penalty:
                gathered = gathered - curvature_penalty * change[k][None, :]
            choice = np.argmax(gathered, axis=1)
            best = gathered[columns, choice]
            best[~valid] = -np.inf
            history[row, :, k] = choice
            updated[:, k] = best - prior[k] + evidence[row] - displacement
        score = updated

    if not np.isfinite(score[middle]).any():
        raise ValueError("两点间找不到连续路径，请加一个沿竿身的引导点。")
    state = int(np.argmax(score[middle]))
    track = np.empty(count, dtype=np.int32)
    track[-1] = middle
    for row in range(count - 1, 0, -1):
        track[row - 1] = track[row] - shifts[state]
        state = int(history[row, track[row], state])
    return track


def trace_segment(
    image: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    corridor: float = CORRIDOR,
    curvature_penalty: float = CURVATURE_PENALTY,
) -> tuple[np.ndarray, np.ndarray]:
    """Trace the rod between two anchors and return points plus their evidence."""
    _, normal, distance = segment_frame(start, end)
    pitch = pixel_pitch(image)
    count = max(5, int(np.ceil(distance / (2 * pitch))) + 1)
    centers = np.linspace(start, end, count)
    half_width = int(np.ceil(max(16 * pitch, corridor * distance) / pitch))
    offsets = np.arange(-half_width, half_width + 1) * pitch
    map_x = (centers[:, 0, None] + offsets * normal[0]).astype(np.float32)
    map_y = (centers[:, 1, None] + offsets * normal[1]).astype(np.float32)
    strip = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    score, yellow, grey = image_evidence(strip, pitch)
    # Average over nearby longitudinal samples to favour sustained evidence.
    score = cv2.GaussianBlur(score, (3, 5), sigmaX=0.55, sigmaY=1.0)
    height, width = image.shape[:2]
    valid = (map_x >= 0) & (map_x <= width - 1) & (map_y >= 0) & (map_y <= height - 1)
    track = best_path(np.where(valid, score, -np.inf), offsets, MAX_SHIFT_PX, curvature_penalty)
    refined = refine_centers(track, yellow, grey, pitch)
    points = centers + ((refined - half_width) * pitch)[:, None] * normal
    return points, score[np.arange(count), track]


def extract_centerline(
    image: np.ndarray,
    anchors,
    corridor: float = CORRIDOR,
    smoothing_px: float = 0.75,
    spacing_px: float = 4.0,
    curvature_penalty: float = CURVATURE_PENALTY,
) -> Centerline:
    """Extract the rod centerline from a side-view photograph."""
    anchors = validate_anchors(image, anchors)
    if not 0.02 <= corridor <= 0.45:
        raise ValueError("搜索宽度比例必须在 0.02 到 0.45 之间。")
    if not np.isfinite(smoothing_px) or smoothing_px < 0:
        raise ValueError("平滑参数必须是非负有限数。")
    if not np.isfinite(spacing_px) or spacing_px < 0.5:
        raise ValueError("采样间距至少为 0.5 像素。")

    segments = [
        trace_segment(image, a, b, corridor, curvature_penalty)
        for a, b in zip(anchors, anchors[1:])
    ]
    raw = np.concatenate([segments[0][0], *[points[1:] for points, _ in segments[1:]]])
    support = np.concatenate([segments[0][1], *[scores[1:] for _, scores in segments[1:]]])
    raw_arc = cumulative_length(raw)
    anchor_indices = np.r_[0, np.cumsum([len(points) - 1 for points, _ in segments])]

    # Anchors are the only ground truth in the photograph, so they outweigh any
    # number of ordinary traced points in the spline fit.
    weights = 0.35 + 0.65 * support
    weights[anchor_indices] = 100.0
    spline, _ = make_splprep(raw.T, u=raw_arc, w=weights, k=3, s=len(raw) * smoothing_px ** 2)

    # Sample densely before computing arc length to reduce pixel-grid bias.
    dense_t = np.unique(np.r_[np.linspace(0, raw_arc[-1], len(raw) * 4), raw_arc[anchor_indices]])
    dense = spline(dense_t).T
    dense[0], dense[-1] = anchors[0], anchors[-1]
    fitted_arc = cumulative_length(dense)
    length = fitted_arc[-1]
    sample_arc = np.r_[np.arange(0, length, spacing_px), length]
    pixels = np.column_stack([np.interp(sample_arc, fitted_arc, col) for col in dense.T])
    sample_t = np.interp(sample_arc, fitted_arc, dense_t)
    evidence = np.interp(sample_t, raw_arc, support)

    # Flag samples whose evidence is weak, and those where smoothing departs far
    # enough from the traced line that the two disagree about where the rod is.
    raw_at_samples = np.column_stack([np.interp(sample_t, raw_arc, col) for col in raw.T])
    fit_offset = np.linalg.norm(pixels - raw_at_samples, axis=1)
    review = (evidence < EVIDENCE_REVIEW_THRESHOLD) | (fit_offset > max(FIT_REVIEW_PX, 3 * smoothing_px))

    root_relative = pixels - pixels[0]
    root_relative[:, 0] *= -1 if anchors[-1, 0] < anchors[0, 0] else 1
    root_relative[:, 1] *= -1
    return Centerline(anchors, raw, dense, pixels, sample_arc, root_relative, evidence, review)
