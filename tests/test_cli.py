"""End-to-end checks: CLI output layout, batch manifest, CSV schema."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from rodcurve.cli import main, read_cases, write_summary

from .synthetic import parabola, render


@pytest.fixture(scope="module")
def photo(tmp_path_factory):
    truth, anchors = parabola()
    path = tmp_path_factory.mktemp("photos") / "sample.png"
    import cv2

    cv2.imwrite(str(path), cv2.cvtColor(render(truth), cv2.COLOR_RGB2BGR))
    points = ";".join(f"{x:.0f},{y:.0f}" for x, y in anchors)
    return path, points


def _point_flags(points: str) -> list[str]:
    flags: list[str] = []
    for pair in points.split(";"):
        flags += ["--point", pair]
    return flags


def test_single_case_writes_all_outputs(photo, tmp_path):
    path, points = photo
    code = main([str(path), *_point_flags(points), "--length-mm", "3450",
                 "--clamp-back", "3700,520", "--output", str(tmp_path)])
    assert code == 0
    for name in ("centerline.csv", "overlay.png", "curve.png", "measurement.json"):
        assert (tmp_path / name).exists(), name

    with (tmp_path / "centerline.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["s_px"] == "0.00000"
    assert float(rows[-1]["s_mm"]) == pytest.approx(3450.0, abs=1e-3)
    assert "clamp_angle_deg" in rows[0]

    details = json.loads((tmp_path / "measurement.json").read_text(encoding="utf-8"))
    assert details["length_mm"] == pytest.approx(3450.0)
    assert details["clamp_angle_deg"] == pytest.approx(0.0, abs=1e-6)
    assert details["sample_count"] == len(rows)


def test_batch_manifest_and_summary(photo, tmp_path):
    path, points = photo
    manifest = tmp_path / "cases.csv"
    manifest.write_text(
        "image,points,length_mm\n"
        f'{path.name},"{points}",3450\n',
        encoding="utf-8",
    )
    cases = read_cases(manifest, path.parent)
    assert len(cases) == 1 and cases[0]["length_mm"] == 3450

    records = [{"image": "a.jpg", "length_px": 1234.5, "length_mm": 3450.0,
                "tip_x_mm": 3100.0, "tip_y_mm": -800.25, "clamp_angle_deg": None,
                "sample_count": 900, "review_count": 12, "output": "out/a"}]
    summary = tmp_path / "summary.csv"
    write_summary(summary, records)
    with summary.open(encoding="utf-8-sig") as handle:
        row = next(csv.DictReader(handle))
    assert row["tip_y_mm"] == "-800.25000"
    assert row["clamp_angle_deg"] == ""


def test_manifest_rejects_bad_rows(tmp_path):
    manifest = tmp_path / "cases.csv"
    manifest.write_text("image,points,length_mm\nonly_name,,\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_cases(manifest, tmp_path)


def test_missing_point_is_reported(photo):
    path, _ = photo
    assert main([str(path), "--point", "100,100", "--output",
                 str(path.parent / "out")]) == 1
