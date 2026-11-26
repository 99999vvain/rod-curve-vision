"""Fishing-rod centerline extraction from side-view photographs.

The package deliberately keeps one public surface so scripts and the bundled
interfaces share a single implementation::

    from rodcurve import load_image, extract_centerline, physical_coordinates

The mechanical model that consumes the exported CSV lives in ``web/model.html``
and is driven entirely in the browser; see ``docs/theory.md``.
"""

from .calibration import measure_clamp_angle, parse_length_mm, physical_coordinates
from .evidence import image_evidence
from .export import render_overlay, save_curve_plot, write_csv
from .geometry import cumulative_length, pixel_pitch
from .imageio import decode_image, load_image, preview_bytes
from .tracing import Centerline, extract_centerline, trace_segment, validate_anchors

__version__ = "1.1.0"

__all__ = [
    "Centerline",
    "cumulative_length",
    "decode_image",
    "extract_centerline",
    "image_evidence",
    "load_image",
    "measure_clamp_angle",
    "parse_length_mm",
    "physical_coordinates",
    "pixel_pitch",
    "preview_bytes",
    "render_overlay",
    "save_curve_plot",
    "trace_segment",
    "validate_anchors",
    "write_csv",
]
