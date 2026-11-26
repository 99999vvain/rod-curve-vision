"""Compare the tracer against a previously recorded centerline.

This is a regression check, not an accuracy check: the reference CSV was
produced by an earlier version of the same method, so agreement means the
behaviour did not drift.  Absolute accuracy comes from ``tests/test_tracing.py``,
which uses a synthetic photograph with an analytic centerline.

Usage::

    python tools/validate_against_reference.py examples

Every ``*-reference.csv`` in the directory is paired with the photograph of the
same name (``sample-rod-0g-reference.csv`` -> ``sample-rod-0g.jpg``).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rodcurve import extract_centerline, load_image  # noqa: E402


def read_reference(path: Path):
    with path.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    arc = np.array([float(r["s_px"]) for r in rows])
    u = np.array([float(r["image_u_px"]) for r in rows])
    v = np.array([float(r["image_v_px"]) for r in rows])
    length_mm = float(rows[-1]["s_mm"]) if rows[0].get("s_mm") else None
    return arc, u, v, length_mm


def compare(photo: Path, reference: Path) -> dict:
    arc, u, v, length_mm = read_reference(reference)
    # The recorded anchors are not stored in the CSV, so rebuild a comparable
    # three-point selection from the curve itself: clamp exit, mid-span, tip.
    middle = len(arc) // 2
    anchors = [[u[0], v[0]], [u[middle], v[middle]], [u[-1], v[-1]]]
    result = extract_centerline(load_image(photo), anchors)
    matched = np.minimum(result.arc_px, arc[-1])
    deviation = np.hypot(
        result.pixels[:, 0] - np.interp(matched, arc, u),
        result.pixels[:, 1] - np.interp(matched, arc, v),
    )
    return {
        "photo": photo.name,
        "reference_arc_px": float(arc[-1]),
        "arc_px": float(result.arc_px[-1]),
        "median": float(np.median(deviation)),
        "rms": float(np.sqrt(np.mean(deviation ** 2))),
        "p95": float(np.percentile(deviation, 95)),
        "max": float(deviation.max()),
        "review": int(result.review.sum()),
        "samples": len(result.pixels),
        "length_mm": length_mm,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path("examples"))
    args = parser.parse_args()

    references = sorted(args.directory.glob("*-reference.csv"))
    if not references:
        print(f"{args.directory} 下没有 *-reference.csv", file=sys.stderr)
        return 1
    rows = []
    for reference in references:
        photo = reference.with_name(reference.name.replace("-reference.csv", ".jpg"))
        if not photo.exists():
            print(f"跳过 {reference.name}：找不到 {photo.name}", file=sys.stderr)
            continue
        rows.append(compare(photo, reference))

    print("| 照片 | 参照弧长 / px | 本程序弧长 / px | 中位偏差 / px | RMS / px | P95 / px | 最大 / px | 复核点 |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['photo']} | {r['reference_arc_px']:.1f} | {r['arc_px']:.1f} | "
              f"{r['median']:.2f} | {r['rms']:.2f} | {r['p95']:.2f} | {r['max']:.2f} | "
              f"{r['review']}/{r['samples']} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
