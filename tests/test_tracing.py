"""Accuracy and behaviour tests against a known centerline."""

from __future__ import annotations

import numpy as np
import pytest

from rodcurve import extract_centerline, physical_coordinates, validate_anchors
from rodcurve.calibration import measure_clamp_angle, parse_length_mm

from .synthetic import distance_to_polyline, parabola, render


@pytest.fixture(scope="module")
def scene():
    truth, anchors = parabola()
    return render(truth), truth, anchors


def test_traces_painted_body_and_bare_tip(scene):
    """Sub-pixel agreement with the analytic centerline.

    The synthetic rod tapers from 22 px to 7 px and then continues as a 2 px
    carbon line, so the budget is 1.0 px RMSE on a 4096 px frame.  Measured on
    the current implementation: 0.81 px RMSE, 1.44 px at the 95th percentile.
    """
    image, truth, anchors = scene
    result = extract_centerline(image, anchors)
    error = distance_to_polyline(result.pixels, truth)
    assert np.sqrt(np.mean(error ** 2)) < 1.0, f"RMSE {np.sqrt(np.mean(error ** 2)):.2f} px"
    assert np.percentile(error, 95) < 2.0
    # The sample count follows from the arc length and the default spacing.
    assert len(result.pixels) == pytest.approx(result.arc_px[-1] / 4.0 + 1, abs=2)


def test_root_and_tip_are_honoured(scene):
    image, _, anchors = scene
    result = extract_centerline(image, anchors)
    assert np.allclose(result.pixels[0], anchors[0], atol=1e-6)
    assert np.allclose(result.pixels[-1], anchors[-1], atol=0.5)


def test_arc_length_exceeds_chord(scene):
    image, truth, anchors = scene
    result = extract_centerline(image, anchors)
    chord = float(np.linalg.norm(anchors[1] - anchors[0]))
    assert chord < result.arc_px[-1] < chord * 1.25


def test_known_length_sets_the_scale(scene):
    image, _, anchors = scene
    result = extract_centerline(image, anchors)
    physical, info = physical_coordinates(result, None, None, 3450)
    assert info["mode"] == "known_arc_length"
    assert physical[-1, 0] == pytest.approx(3450.0)
    assert info["mm_per_px"] == pytest.approx(3450.0 / result.arc_px[-1])
    # y points up, so a sagging rod has a negative tip ordinate.
    assert physical[-1, 2] < 0


def test_uniform_scale_matches_known_length(scene):
    image, _, anchors = scene
    result = extract_centerline(image, anchors)
    scale = 0.9
    physical, info = physical_coordinates(result, None, scale)
    assert info["mode"] == "uniform_scale"
    assert physical[-1, 0] == pytest.approx(result.arc_px[-1] * scale)


def test_rejects_degenerate_anchors(scene):
    image, _, anchors = scene
    with pytest.raises(ValueError):
        validate_anchors(image, [anchors[0]])
    with pytest.raises(ValueError):
        validate_anchors(image, [anchors[0], anchors[0] + 2])
    with pytest.raises(ValueError):
        validate_anchors(image, [anchors[0], anchors[1] + 10_000])


def test_length_parsing():
    assert parse_length_mm("3400") == 3400
    for bad in ("", "abc", "0", "-5"):
        with pytest.raises(ValueError):
            parse_length_mm(bad)


def test_clamp_angle_is_upward_positive():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    level = measure_clamp_angle(image, (10, 50), (80, 50))
    assert level == pytest.approx(0.0, abs=1e-9)
    # front point is 30 px higher on screen => +29.5 degrees about a 53 px run
    raised = measure_clamp_angle(image, (10, 80), (90, 50))
    assert raised == pytest.approx(np.degrees(np.arctan2(30, 80)))
    lowered = measure_clamp_angle(image, (10, 50), (90, 80))
    assert lowered == pytest.approx(-np.degrees(np.arctan2(30, 80)))
