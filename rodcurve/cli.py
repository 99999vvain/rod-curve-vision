"""Command line entry point: single photograph or a whole batch of cases."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from .calibration import measure_clamp_angle, parse_length_mm, physical_coordinates
from .export import render_overlay, save_curve_plot, save_overlay_png, write_csv
from .imageio import load_image
from .tracing import CORRIDOR, CURVATURE_PENALTY, extract_centerline

DEFAULT_SMOOTHING_PX = 0.75
DEFAULT_SPACING_PX = 4.0


def parse_pair(value: str) -> tuple[float, float]:
    try:
        u, v = value.split(",")
        return float(u), float(v)
    except ValueError as error:
        raise argparse.ArgumentTypeError("点位格式为 U,V，例如 100,250。") from error


def parse_points(value: str) -> list[tuple[float, float]]:
    """``"3824,455;961,769;270,1270"`` -> anchor list."""
    return [parse_pair(item.strip()) for item in value.split(";") if item.strip()]


def choose_image():
    from tkinter import Tk, filedialog

    window = Tk()
    window.withdraw()
    try:
        chosen = filedialog.askopenfilename(
            title="选择鱼竿侧视照片",
            filetypes=[("照片", "*.jpg *.jpeg *.png *.webp *.bmp"), ("所有文件", "*.*")],
        )
    finally:
        window.destroy()
    return Path(chosen) if chosen else None


def pick_anchors(image, allow_length_entry=True):
    """Interactive anchor picking with matplotlib, including a length box."""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.widgets import Button, TextBox

    from .tracing import validate_anchors

    for family in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
        if family in font_manager.get_font_names():
            plt.rcParams["font.sans-serif"] = [family, *plt.rcParams["font.sans-serif"]]
            break
    figure, axes = plt.subplots(figsize=(13, 8))
    figure.canvas.manager.set_window_title("鱼竿曲线识别")
    figure.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.17)
    axes.imshow(image)
    axes.set_axis_off()
    title = axes.set_title("依次点选夹具出口、必要的引导点、实际竿梢", pad=10)
    marks, = axes.plot([], [], "o--", color="#13e7c1", markeredgecolor="#17334a",
                       markersize=6, linewidth=0.9)
    points: list[tuple[float, float]] = []
    state = {"accepted": False, "length": None}
    length_box = None
    if allow_length_entry:
        length_box = TextBox(figure.add_axes((0.20, 0.065, 0.17, 0.05)),
                             "曲线总长度 (mm)  ", initial="")
        figure.text(0.39, 0.081, "填夹具出口至竿梢的实际长度\n留空以像素输出", fontsize=10)

    def redraw():
        marks.set_data(*np.asarray(points).T if points else ([], []))
        title.set_text(f"已选 {len(points)} 点 · 夹具出口 → 引导点 → 竿梢")
        figure.canvas.draw_idle()

    def undo(_event=None):
        if points:
            points.pop()
            redraw()

    def accept(_event=None):
        try:
            validate_anchors(image, points)
            if length_box is not None and length_box.text.strip():
                state["length"] = parse_length_mm(length_box.text)
        except ValueError as error:
            title.set_text(str(error))
            figure.canvas.draw_idle()
            return
        state["accepted"] = True
        plt.close(figure)

    def select(event):
        toolbar = getattr(figure.canvas, "toolbar", None)
        if event.inaxes != axes or getattr(toolbar, "mode", ""):
            return
        if event.button == 1:
            points.append((event.xdata, event.ydata))
            redraw()
        elif event.button == 3:
            undo()

    def key(event):
        if length_box is not None and length_box.capturekeystrokes:
            return
        if event.key == "enter":
            accept()
        elif event.key in ("backspace", "delete"):
            undo()

    Button(figure.add_axes((0.68, 0.065, 0.11, 0.05)), "撤销").on_clicked(undo)
    Button(figure.add_axes((0.81, 0.065, 0.15, 0.05)), "提取曲线").on_clicked(accept)
    figure.text(0.035, 0.025, "可用工具栏的放大镜精确选点，放大后关闭放大模式再点选。", fontsize=10)
    figure.canvas.mpl_connect("button_press_event", select)
    figure.canvas.mpl_connect("key_press_event", key)
    plt.show()
    return (points, state["length"]) if state["accepted"] else None


def make_output_directory(requested: Path | None, image_path: Path) -> Path:
    if requested is not None:
        requested.mkdir(parents=True, exist_ok=True)
        return requested
    parent = Path(__file__).resolve().parent.parent / "outputs"
    parent.mkdir(exist_ok=True)
    index = 1
    while True:
        name = image_path.stem if index == 1 else f"{image_path.stem}_{index:02d}"
        candidate = parent / name
        try:
            candidate.mkdir()
        except FileExistsError:
            index += 1
        else:
            return candidate


def process_case(case: dict) -> dict:
    """Run one case end to end and return a summary record.

    Kept free of argparse state so it can run in a worker process for batch work.
    """
    image_path = Path(case["image"])
    output = make_output_directory(case.get("output"), image_path)
    image = load_image(image_path)
    anchors = np.asarray(case["anchors"], dtype=float)
    result = extract_centerline(
        image,
        anchors,
        corridor=case.get("corridor", CORRIDOR),
        smoothing_px=case.get("smoothing_px", DEFAULT_SMOOTHING_PX),
        spacing_px=case.get("spacing_px", DEFAULT_SPACING_PX),
        curvature_penalty=case.get("curvature_penalty", CURVATURE_PENALTY),
    )

    calibration = case.get("calibration")
    physical, calibration_info = physical_coordinates(
        result, calibration, case.get("mm_per_px"), case.get("length_mm")
    )
    clamp_angle = None
    clamp_points = None
    if case.get("clamp_back") is not None:
        homography = calibration_info["homography"] if calibration is not None else None
        clamp_points = (case["clamp_back"], case.get("clamp_front") or tuple(anchors[0]))
        clamp_angle = measure_clamp_angle(image, *clamp_points, homography=homography)

    write_csv(output / "centerline.csv", result, physical, clamp_angle)
    canvas = render_overlay(image, result, clamp_points)
    save_overlay_png(output / "overlay.png", canvas, case.get("overlay_compress", 6))
    if not case.get("no_plot", False):
        save_curve_plot(output / "curve.png", result, physical)

    details = {
        "image": image_path.name,
        "image_size_px": list(image.shape[1::-1]),
        "anchors_image_px": result.anchors.tolist(),
        "length_px": float(result.arc_px[-1]),
        "tip_relative_px": result.coordinates_px[-1].tolist(),
        "tip_vertical_drop_px": float(-result.coordinates_px[-1, 1]),
        "sample_count": len(result.pixels),
        "review_sample_count": int(result.review.sum()),
        "corridor": case.get("corridor", CORRIDOR),
        "smoothing_px": case.get("smoothing_px", DEFAULT_SMOOTHING_PX),
        "spacing_px": case.get("spacing_px", DEFAULT_SPACING_PX),
        "curvature_penalty": case.get("curvature_penalty", CURVATURE_PENALTY),
        "calibration": calibration_info,
    }
    if physical is not None:
        tip_s, tip_x, tip_y = physical[-1].tolist()
        details.update(
            length_mm=tip_s,
            tip_relative_mm=[tip_x, tip_y],
            tip_vertical_distance_mm=abs(tip_y),
            tip_vertical_drop_mm=-tip_y,
        )
    if clamp_angle is not None:
        details.update(
            clamp_points_image_px=clamp_points,
            clamp_angle_deg=clamp_angle,
            clamp_angle_reference="world_horizontal" if calibration is not None else "image_horizontal",
        )
    (output / "measurement.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "image": image_path.name,
        "output": str(output),
        "length_px": details["length_px"],
        "length_mm": details.get("length_mm"),
        "tip_x_mm": details.get("tip_relative_mm", [None, None])[0],
        "tip_y_mm": details.get("tip_relative_mm", [None, None])[1],
        "clamp_angle_deg": details.get("clamp_angle_deg"),
        "sample_count": details["sample_count"],
        "review_count": details["review_sample_count"],
    }


def read_cases(path: Path, base: Path) -> list[dict]:
    """Read a batch manifest.

    Columns: ``image, points, length_mm`` and optionally ``clamp_back``,
    ``clamp_front``, ``output``.  ``points`` uses ``u,v;u,v;...`` from clamp exit
    to tip, the same order as repeated ``--point`` flags.
    """
    cases = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for number, row in enumerate(csv.DictReader(handle), start=2):
            if None in row:
                raise ValueError(
                    f"第 {number} 行的字段数多于表头：points 列本身含逗号，"
                    "在 CSV 中需要用双引号包裹整格。"
                )
            image = (row.get("image") or "").strip()
            points = (row.get("points") or "").strip()
            if not image or not points:
                raise ValueError(f"第 {number} 行缺少 image 或 points 列。")
            image_path = Path(image)
            if not image_path.is_absolute():
                image_path = base / image_path
            case = {
                "image": image_path,
                "anchors": parse_points(points),
                "length_mm": (row.get("length_mm") or "").strip() or None,
            }
            if case["length_mm"] is not None:
                case["length_mm"] = parse_length_mm(case["length_mm"])
            for key in ("clamp_back", "clamp_front"):
                value = (row.get(key) or "").strip()
                case[key] = parse_pair(value) if value else None
            output = (row.get("output") or "").strip()
            if output:
                case["output"] = Path(output) if Path(output).is_absolute() else base / output
            cases.append(case)
    if not cases:
        raise ValueError("批量清单为空。")
    return cases


def write_summary(path: Path, records: list[dict]) -> None:
    headings = ["image", "length_px", "length_mm", "tip_x_mm", "tip_y_mm",
                "clamp_angle_deg", "sample_count", "review_count", "output"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(headings)
        for record in records:
            writer.writerow(["" if record.get(h) is None else
                             (f"{record[h]:.5f}" if isinstance(record.get(h), float) else record.get(h))
                             for h in headings])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rodcurve", description="从侧视照片提取鱼竿中心线，并换算到毫米坐标。")
    parser.add_argument("image", nargs="?", type=Path, help="图片路径，省略时打开文件选择窗口")
    parser.add_argument("--point", action="append", type=parse_pair,
                        help="按夹具出口至竿梢顺序给出 U,V。重复使用。省略时在图中点选。")
    parser.add_argument("--clamp-back", type=parse_pair, help="夹持段靠竿柄侧的点 U,V。")
    parser.add_argument("--clamp-front", type=parse_pair,
                        help="夹持段靠竿梢侧的点 U,V，搭配 --clamp-back，默认取夹具出口。")
    parser.add_argument("--output", type=Path, help="输出目录，默认在仓库根目录的 outputs 下")
    parser.add_argument("--corridor", default=CORRIDOR, type=float,
                        help=f"搜索半宽占相邻两点距离的比例，默认 {CORRIDOR}")
    parser.add_argument("--smoothing-px", default=DEFAULT_SMOOTHING_PX, type=float,
                        help="样条平滑参数")
    parser.add_argument("--spacing-px", default=DEFAULT_SPACING_PX, type=float,
                        help="沿弧长的输出间距")
    parser.add_argument("--curvature-penalty", default=CURVATURE_PENALTY, type=float,
                        help="路径二阶平滑惩罚，0 表示退化为原始一阶搜索")
    parser.add_argument("--no-plot", action="store_true", help="跳过 curve.png，省去绘图开销")
    parser.add_argument("--overlay-compress", type=int, default=6, choices=range(0, 10),
                        metavar="0-9",
                        help="叠加图 PNG 压缩级别，6 为默认；1 更快但文件更大")
    parser.add_argument("--cases", type=Path, help="批量清单 CSV，见 docs/usage.md")
    parser.add_argument("--jobs", type=int, default=1, help="批量模式下并行进程数")
    calibration_args = parser.add_mutually_exclusive_group()
    calibration_args.add_argument("--calibration", type=Path, help="同平面标定点 JSON")
    calibration_args.add_argument("--mm-per-px", type=float, help="已知的均匀毫米比例")
    calibration_args.add_argument("--length-mm", type=float, help="夹具出口至竿梢的实际曲线总长（毫米）")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.clamp_front is not None and args.clamp_back is None:
        parser.error("--clamp-front 必须搭配 --clamp-back。")
    if args.cases is not None and args.image is not None:
        parser.error("--cases 与单张图片不能同时使用。")

    try:
        if args.cases is not None:
            return _run_batch(args)
        return _run_single(args)
    except (ValueError, TypeError, OSError, cv2.error) as error:
        print(f"提取失败：{error}", file=sys.stderr)
        return 1


def _common_case(args, image_path: Path) -> dict:
    calibration = None
    if args.calibration:
        calibration = json.loads(args.calibration.read_text(encoding="utf-8-sig"))
        if not isinstance(calibration, dict):
            raise ValueError("标定文件需要包含 image_px 和 world_mm 的 JSON 对象。")
    return {
        "image": image_path,
        "output": args.output,
        "corridor": args.corridor,
        "smoothing_px": args.smoothing_px,
        "spacing_px": args.spacing_px,
        "curvature_penalty": args.curvature_penalty,
        "no_plot": args.no_plot,
        "overlay_compress": args.overlay_compress,
        "calibration": calibration,
        "mm_per_px": args.mm_per_px,
        "length_mm": args.length_mm,
        "clamp_back": args.clamp_back,
        "clamp_front": args.clamp_front,
    }


def _run_single(args) -> int:
    image_path = args.image or choose_image()
    if image_path is None:
        return 0
    case = _common_case(args, image_path)
    if args.point is None:
        image = load_image(image_path)
        allow_length_entry = (args.calibration is None and args.mm_per_px is None
                              and args.length_mm is None)
        selection = pick_anchors(image, allow_length_entry)
        if selection is None:
            return 0
        anchors, entered_length = selection
        case["anchors"] = anchors
        if entered_length is not None:
            case["length_mm"] = entered_length
    else:
        case["anchors"] = args.point

    record = process_case(case)
    print(f"弧长 {record['length_px']:.1f} px，采样 {record['sample_count']} 点，"
          f"需复核 {record['review_count']} 点。")
    if record["length_mm"] is not None:
        print(f"实际弧长 {record['length_mm']:.2f} mm，"
              f"竿梢坐标 ({record['tip_x_mm']:.2f}, {record['tip_y_mm']:.2f}) mm。")
        print(f"竿梢至夹具出口所在水平面的垂直距离 {abs(record['tip_y_mm']):.2f} mm。")
    if record["clamp_angle_deg"] is not None:
        reference = "标定坐标水平线" if args.calibration is not None else "照片水平线"
        print(f"夹持角 {record['clamp_angle_deg']:+.2f}°（以{reference}为基准，向竿梢侧上仰为正）。")
    print(f"结果目录 {Path(record['output']).resolve()}")
    if args.point is None and sys.platform == "win32":
        import os

        try:
            os.startfile(Path(record["output"]) / "overlay.png")
        except OSError:
            print(f"可手动打开 {Path(record['output']) / 'overlay.png'} 查看叠加图。")
    return 0


def _run_batch(args) -> int:
    base = args.cases.resolve().parent
    cases = read_cases(args.cases, base)
    for case in cases:
        for key in ("corridor", "smoothing_px", "spacing_px", "curvature_penalty",
                    "no_plot", "overlay_compress"):
            case[key] = getattr(args, key)
        case.setdefault("calibration", None)
        case["calibration"] = json.loads(args.calibration.read_text(encoding="utf-8-sig")) \
            if args.calibration else None
        case["mm_per_px"] = args.mm_per_px
        case["clamp_back"] = case.get("clamp_back") or args.clamp_back
        case["clamp_front"] = case.get("clamp_front") or args.clamp_front
        if not case.get("output"):
            case["output"] = (args.output / Path(case["image"]).stem) if args.output else None

    records: list[dict] = []
    failures: list[tuple[str, str]] = []
    if args.jobs > 1 and len(cases) > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(process_case, case): case for case in cases}
            for future, case in futures.items():
                try:
                    records.append(future.result())
                except Exception as error:  # noqa: BLE001 - report and continue
                    failures.append((Path(case["image"]).name, str(error)))
    else:
        for case in cases:
            try:
                records.append(process_case(case))
            except (ValueError, TypeError, OSError, cv2.error) as error:
                failures.append((Path(case["image"]).name, str(error)))

    summary = Path(args.output or (Path(__file__).resolve().parent.parent / "outputs")) / "summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    write_summary(summary, records)
    print(f"完成 {len(records)} 例，汇总 {summary.resolve()}")
    for name, message in failures:
        print(f"失败 {name}：{message}", file=sys.stderr)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
