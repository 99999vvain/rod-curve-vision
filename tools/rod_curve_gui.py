"""Optional desktop interface (Tkinter) for the rod centerline tool.

The browser interface (``python -m rodcurve.server``) is the primary one; this
window exists for machines where a browser is inconvenient and for measuring
without leaving the desktop.  It uses the same tracing code.
"""

"""Windows-friendly desktop interface for the fishing-rod measurement tool."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from matplotlib import font_manager, rcParams
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    from rodcurve import (
        extract_centerline, load_image, measure_clamp_angle, parse_length_mm, physical_coordinates,
        validate_anchors, write_csv,
    )
except ImportError as error:
    if __name__ != "__main__":
        raise
    window = tk.Tk()
    window.withdraw()
    messagebox.showerror(
        "无法启动鱼竿曲线识别",
        f"请在程序目录运行 python -m pip install -r requirements.txt。\n\n{error}",
        parent=window,
    )
    window.destroy()
    raise SystemExit(1)


BACKGROUND = "#f1f5f8"
PAPER = "#ffffff"
INK = "#172a3a"
MUTED = "#526575"
LINE = "#dce5ec"
TEAL = "#087366"
TEAL_SOFT = "#e2f4ef"
WARNING = "#965422"
CLAMP = "#c88fff"
FONT = "Microsoft YaHei UI"


class RodCurveApp:
    def __init__(self, root):
        self.root = root
        self.image = None
        self.image_path = None
        self.anchors = []
        self.clamp_points = None
        self.clamp_start = None
        self.clamp_angle = None
        self.picking_clamp = False
        self.result = None
        self.physical = None
        self.length_error = None
        self.future = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.view = "photo"
        self.panning = False
        self.drag_origin = None
        self.photo_limits = None

        self.file_text = tk.StringVar(value="尚未导入照片")
        self.point_text = tk.StringVar(value="至少选择夹具出口和竿梢两点。")
        self.clamp_text = tk.StringVar(value="以照片水平线为基准。")
        self.length_text = tk.StringVar()
        self.length_hint = tk.StringVar(value="填夹具出口至竿梢沿竿身的实际长度。")
        self.metric_text = tk.StringVar(value="—")
        self.tip_text = tk.StringVar(value="完成曲线提取后显示")
        self.status_text = tk.StringVar(value="导入一张鱼竿侧视照片开始。")

        root.title("鱼竿曲线识别")
        root.geometry("1260x805")
        root.minsize(910, 690)
        root.configure(background=BACKGROUND)
        self.configure_styles()
        self.build_layout()
        self.length_text.trace_add("write", self.update_scale)
        root.bind("<Control-o>", lambda _event: self.open_photo())
        root.bind("<Control-s>", lambda _event: self.export_csv())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.draw_view()
        self.update_controls()

    def configure_styles(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", font=(FONT, 10), foreground=INK)
        style.configure("TButton", background=PAPER, bordercolor=LINE, lightcolor=PAPER,
                        darkcolor=PAPER, foreground=INK, padding=(10, 6))
        style.map("TButton", background=[("disabled", BACKGROUND), ("active", TEAL_SOFT)],
                  foreground=[("disabled", "#84919a")])
        style.configure("Primary.TButton", background=TEAL, bordercolor=TEAL,
                        lightcolor=TEAL, darkcolor=TEAL, foreground=PAPER, font=(FONT, 10, "bold"))
        style.map("Primary.TButton",
                  background=[("disabled", "#dce8eb"), ("active", "#095a53")],
                  foreground=[("disabled", "#687d86")])
        style.configure("Selected.TButton", background=TEAL_SOFT, bordercolor="#aacec4",
                        lightcolor=TEAL_SOFT, darkcolor=TEAL_SOFT, foreground=TEAL)
        style.configure("TEntry", padding=(8, 6), fieldbackground=PAPER, bordercolor=LINE)

        families = set(font_manager.get_font_names())
        for family in (FONT, "Microsoft YaHei", "SimHei"):
            if family in families:
                rcParams["font.sans-serif"] = [family, *rcParams["font.sans-serif"]]
                break
        rcParams["axes.unicode_minus"] = False

    def build_layout(self):
        header = tk.Frame(self.root, background=PAPER, padx=23, pady=13)
        header.pack(fill="x")
        tk.Label(header, text="鱼竿曲线识别", background=PAPER, foreground=INK,
                 font=(FONT, 19, "bold")).pack(side="left")
        self.export_button = ttk.Button(header, text="导出 CSV…", style="Primary.TButton",
                                        command=self.export_csv)
        self.export_button.pack(side="right")

        footer = tk.Frame(self.root, background=BACKGROUND, padx=22, pady=7)
        footer.pack(side="bottom", fill="x")
        tk.Label(footer, textvariable=self.status_text, background=BACKGROUND,
                 foreground=MUTED, anchor="w", font=(FONT, 9)).pack(fill="x")

        content = tk.Frame(self.root, background=BACKGROUND)
        content.pack(fill="both", expand=True, padx=16, pady=(14, 0))
        content.rowconfigure(0, weight=1)
        content.columnconfigure(1, weight=1)
        sidebar = tk.Frame(content, background=PAPER, highlightthickness=1,
                           highlightbackground=LINE, padx=15, pady=11)
        sidebar.grid(row=0, column=0, sticky="ns", padx=(0, 12))
        self.build_sidebar(sidebar)

        work = tk.Frame(content, background=PAPER, highlightthickness=1, highlightbackground=LINE)
        work.grid(row=0, column=1, sticky="nsew")
        toolbar = tk.Frame(work, background=PAPER, padx=12, pady=9)
        toolbar.pack(fill="x")
        self.photo_button = ttk.Button(toolbar, text="照片标注", command=lambda: self.switch_view("photo"))
        self.photo_button.pack(side="left")
        self.curve_button = ttk.Button(toolbar, text="变形曲线", command=lambda: self.switch_view("curve"))
        self.curve_button.pack(side="left", padx=(5, 0))
        self.reset_button = ttk.Button(toolbar, text="重置视图", command=self.reset_view)
        self.reset_button.pack(side="right")
        self.pan_button = ttk.Button(toolbar, text="拖动平移", command=self.toggle_pan)
        self.pan_button.pack(side="right", padx=(0, 5))
        tk.Frame(work, height=1, background=LINE).pack(fill="x")

        self.figure = Figure(facecolor=PAPER)
        self.axes = self.figure.add_subplot()
        self.canvas = FigureCanvasTkAgg(self.figure, master=work)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self.canvas_press)
        self.canvas.mpl_connect("button_release_event", self.canvas_release)
        self.canvas.mpl_connect("motion_notify_event", self.canvas_move)
        self.canvas.mpl_connect("scroll_event", self.canvas_scroll)

    def build_sidebar(self, sidebar):
        def heading(text, top=0):
            tk.Label(sidebar, text=text, background=PAPER, foreground=INK,
                     font=(FONT, 11, "bold"), anchor="w").pack(fill="x", pady=(top, 4))

        def note(variable):
            tk.Label(sidebar, textvariable=variable, background=PAPER, foreground=MUTED,
                     font=(FONT, 9), wraplength=238, justify="left", anchor="w").pack(
                         fill="x", pady=(4, 0))

        heading("实验照片")
        self.import_button = ttk.Button(sidebar, text="导入图片…", command=self.open_photo)
        self.import_button.pack(fill="x")
        self.file_label = tk.Label(sidebar, textvariable=self.file_text, background=PAPER,
                                   foreground=MUTED, font=(FONT, 9), width=29, anchor="w")
        self.file_label.pack(fill="x", pady=(4, 0))

        heading("参照点", top=13)
        tk.Label(sidebar, text="夹具出口 → 可选引导点 → 实际竿梢", background=PAPER,
                 foreground=MUTED, font=(FONT, 9), anchor="w").pack(fill="x")
        note(self.point_text)
        row = tk.Frame(sidebar, background=PAPER)
        row.pack(fill="x", pady=(5, 0))
        self.undo_button = ttk.Button(row, text="撤销上一个", command=self.undo_point)
        self.undo_button.pack(side="left", fill="x", expand=True)
        self.clear_button = ttk.Button(row, text="清空", command=self.clear_points)
        self.clear_button.pack(side="left", fill="x", expand=True, padx=(5, 0))

        row = tk.Frame(sidebar, background=PAPER)
        row.pack(fill="x", pady=(8, 0))
        self.clamp_button = ttk.Button(row, text="测量夹持角", command=self.toggle_clamp_pick)
        self.clamp_button.pack(side="left", fill="x", expand=True)
        self.clear_clamp_button = ttk.Button(row, text="移除", command=self.clear_clamp)
        self.clear_clamp_button.pack(side="left", padx=(5, 0))
        note(self.clamp_text)

        heading("曲线总长度（mm）", top=13)
        self.length_entry = ttk.Entry(sidebar, textvariable=self.length_text, width=22)
        self.length_entry.pack(fill="x")
        note(self.length_hint)
        self.extract_button = ttk.Button(sidebar, text="提取曲线", style="Primary.TButton",
                                         command=self.start_extraction)
        self.extract_button.pack(fill="x", pady=(12, 0))

        metric = tk.Frame(sidebar, background=TEAL_SOFT, padx=11, pady=8)
        metric.pack(fill="x", pady=(14, 0))
        tk.Label(metric, text="竿梢至夹具水平面的垂直距离", background=TEAL_SOFT,
                 foreground=INK, font=(FONT, 9, "bold"), anchor="w").pack(fill="x")
        tk.Label(metric, textvariable=self.metric_text, background=TEAL_SOFT, foreground=TEAL,
                 font=(FONT, 21, "bold"), anchor="w").pack(fill="x")
        tk.Label(metric, textvariable=self.tip_text, background=TEAL_SOFT, foreground=MUTED,
                 font=(FONT, 9), anchor="w", justify="left", wraplength=216).pack(fill="x")

    def update_controls(self):
        idle = self.future is None
        loaded = self.image is not None
        ready = self.result is not None
        for widget, enabled in (
            (self.import_button, idle),
            (self.undo_button, idle and self.anchors),
            (self.clear_button, idle and self.anchors),
            (self.clamp_button, idle and loaded),
            (self.clear_clamp_button, idle and self.clamp_points is not None),
            (self.extract_button, idle and len(self.anchors) >= 2 and not self.length_error),
            (self.export_button, idle and ready and not self.length_error),
            (self.curve_button, ready),
            (self.pan_button, idle and loaded and self.view == "photo"),
            (self.reset_button, loaded and self.view == "photo"),
        ):
            widget.state(["!disabled"] if enabled else ["disabled"])
        self.extract_button.configure(text="正在提取…" if not idle else "提取曲线")
        self.clamp_button.configure(
            text="取消选点" if self.picking_clamp else "测量夹持角",
            style="Selected.TButton" if self.picking_clamp else "TButton",
        )
        if self.picking_clamp:
            self.clamp_text.set("点击靠竿柄侧的点。" if self.clamp_start is None else "再点击靠竿梢侧的点。")
        elif self.clamp_angle is None:
            self.clamp_text.set("以照片水平线为基准。")
        else:
            self.clamp_text.set(f"夹持角 {self.clamp_angle:+.2f}°（向梢端上仰为正）")
        self.photo_button.configure(style="Selected.TButton" if self.view == "photo" else "TButton")
        self.curve_button.configure(style="Selected.TButton" if self.view == "curve" else "TButton")
        self.pan_button.configure(style="Selected.TButton" if self.panning else "TButton")

    def open_photo(self):
        if self.future is not None:
            return
        name = filedialog.askopenfilename(
            parent=self.root, title="导入鱼竿侧视照片",
            filetypes=[("图片", "*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff"), ("所有文件", "*.*")],
        )
        if name:
            self.load_photo(Path(name))

    def load_photo(self, path):
        try:
            image = load_image(path)
        except (OSError, ValueError) as error:
            messagebox.showerror("无法打开图片", str(error), parent=self.root)
            return
        self.image = image
        self.image_path = path
        self.file_text.set(path.name if len(path.name) <= 30 else f"{path.stem[:24]}…{path.suffix}")
        self.anchors.clear()
        self.clamp_points = self.clamp_angle = None
        self.clamp_start = None
        self.picking_clamp = False
        self.photo_limits = None
        self.panning = False
        self.canvas.get_tk_widget().configure(cursor="")
        self.points_changed()

    def points_changed(self):
        self.result = None
        self.physical = None
        self.view = "photo"
        count = len(self.anchors)
        if count >= 2:
            self.point_text.set(f"已选 {count} 点，末点为竿梢。")
        elif count == 1:
            self.point_text.set("已选夹具出口，继续选择引导点或竿梢。")
        else:
            self.point_text.set("至少选择夹具出口和竿梢两点。")
        self.update_scale()
        self.status_text.set("照片中左键选点，右键撤销，滚轮以指针为中心缩放。")
        self.draw_view()

    def undo_point(self):
        if self.anchors and self.future is None:
            self.anchors.pop()
            self.points_changed()

    def clear_points(self):
        if self.anchors and self.future is None:
            self.anchors.clear()
            self.points_changed()

    def toggle_clamp_pick(self):
        if self.picking_clamp:
            self.switch_view("photo")
            self.status_text.set("已退出夹持角选点。")
            return
        self.switch_view("photo")
        self.picking_clamp = True
        self.canvas.get_tk_widget().configure(cursor="crosshair")
        self.status_text.set("从竿柄侧朝竿梢侧点选夹持段上的两点，也可用与夹持段平行的参照边。")
        self.update_controls()

    def clear_clamp(self):
        self.clamp_points = self.clamp_angle = None
        self.switch_view(self.view)
        self.status_text.set("已移除夹持角。")

    def update_scale(self, *_args):
        text = self.length_text.get().strip()
        self.length_error = None
        length = None
        if text:
            try:
                length = parse_length_mm(text)
            except ValueError as error:
                self.length_error = str(error)
        self.length_hint.set(self.length_error or "填自由段实测长度。留空时使用像素。")
        self.physical = None
        if self.result is None or self.length_error:
            self.metric_text.set("—")
            self.tip_text.set("请填写有效的长度。" if self.length_error else "完成曲线提取后显示")
        else:
            if length is not None:
                self.physical, _ = physical_coordinates(self.result, None, None, length)
            x, y, unit = self.tip_measurement()
            self.metric_text.set(f"{abs(y):.2f} {unit}")
            direction = "下方" if y < 0 else "上方" if y > 0 else "同一高度"
            self.tip_text.set(f"竿梢 ({x:.2f}, {y:.2f}) {unit}\n相对夹具水平面：{direction}")
        if self.view == "curve":
            self.draw_view()
        self.update_controls()

    def tip_measurement(self):
        if self.physical is not None:
            x, y = self.physical[-1, 1:]
            return x, y, "mm"
        x, y = self.result.coordinates_px[-1]
        return x, y, "px"

    def start_extraction(self):
        if self.future is not None or self.image is None or self.length_error:
            return
        try:
            anchors = validate_anchors(self.image, self.anchors)
        except ValueError as error:
            self.status_text.set(str(error))
            return
        self.panning = False
        self.picking_clamp = False
        self.clamp_start = None
        self.canvas.get_tk_widget().configure(cursor="")
        self.future = self.executor.submit(extract_centerline, self.image, anchors)
        self.status_text.set("正在追踪竿身和细梢…")
        self.update_controls()
        self.root.after(70, self.finish_extraction)

    def finish_extraction(self):
        if not self.future.done():
            self.root.after(70, self.finish_extraction)
            return
        future, self.future = self.future, None
        try:
            self.result = future.result()
        except Exception as error:
            self.status_text.set("曲线提取失败，请检查参照点。")
            messagebox.showerror("曲线提取失败", str(error), parent=self.root)
            self.update_controls()
            return
        self.update_scale()
        self.switch_view("photo")
        count = int(self.result.review.sum())
        self.status_text.set(f"已提取 {len(self.result.pixels)} 个点 · {count} 个橙色点待复核"
                             " · 可查看变形曲线或导出 CSV。")

    def export_csv(self):
        if self.result is None or self.future is not None or self.length_error:
            return
        filename = filedialog.asksaveasfilename(
            parent=self.root, title="导出曲线 CSV", defaultextension=".csv",
            filetypes=[("CSV 文件", "*.csv")],
            initialfile=f"{self.image_path.stem}_中心线.csv", initialdir=self.image_path.parent,
        )
        if filename:
            try:
                write_csv(Path(filename), self.result, self.physical, self.clamp_angle)
            except OSError as error:
                messagebox.showerror("CSV 保存失败", str(error), parent=self.root)
                return
            self.status_text.set(f"已保存 {Path(filename).name} · {'含毫米坐标' if self.physical is not None else '像素坐标'}")

    def switch_view(self, view):
        if view == "curve" and self.result is None:
            return
        self.view = view
        self.panning = False
        self.picking_clamp = False
        self.clamp_start = None
        self.canvas.get_tk_widget().configure(cursor="")
        self.draw_view()
        self.update_controls()

    def draw_view(self):
        self.axes.clear()
        if self.view == "curve" and self.result is not None:
            self.draw_curve()
        else:
            self.draw_photo()
        self.canvas.draw_idle()

    def draw_photo(self):
        axes = self.axes
        self.figure.subplots_adjust(left=0.015, bottom=0.019, right=0.985, top=0.981)
        axes.set_axis_off()
        if self.image is None:
            axes.text(0.5, 0.53, "尚未导入照片", transform=axes.transAxes,
                      ha="center", va="center", fontsize=18, color=INK)
            axes.text(0.5, 0.46, "选择左侧「导入图片」打开实验照片。",
                      transform=axes.transAxes, ha="center", va="center", fontsize=11, color=MUTED)
            return
        axes.imshow(self.image, interpolation="antialiased")
        if self.result is not None:
            axes.plot(*self.result.fitted.T, color="#19e6c3", linewidth=1.45)
            review = self.result.pixels[self.result.review]
            axes.scatter(*review.T, color="#ffa028", s=12, linewidths=0)
        if self.clamp_points is not None and not self.picking_clamp:
            (back_x, back_y), (front_x, front_y) = self.clamp_points
            axes.plot([back_x, front_x], [front_y, front_y], color="white", linestyle="--",
                      linewidth=1, zorder=3)
            axes.plot([back_x, front_x], [back_y, front_y], color=CLAMP, linewidth=1.7,
                      marker="o", zorder=3)
        if self.clamp_start is not None:
            axes.scatter(*self.clamp_start, color=CLAMP, edgecolors=PAPER, s=39, zorder=5)
        if self.anchors:
            axes.scatter(*zip(*self.anchors), color=TEAL, edgecolors=PAPER, s=39, zorder=4)
            for index, point in enumerate(self.anchors):
                if index == 0:
                    label = "夹具出口"
                elif index == len(self.anchors) - 1:
                    label = "竿梢"
                else:
                    label = f"引导点 {index}"
                axes.annotate(label, point, xytext=(0, 9), textcoords="offset points",
                              ha="center", color=PAPER, fontsize=9, annotation_clip=True,
                              bbox={"boxstyle": "round,pad=0.26", "fc": TEAL, "ec": "none", "alpha": 0.95})
        if self.photo_limits is not None:
            axes.set_xlim(self.photo_limits[0])
            axes.set_ylim(self.photo_limits[1])

    def draw_curve(self):
        axes = self.axes
        self.figure.subplots_adjust(left=0.12, right=0.96, bottom=0.12, top=0.91)
        _, _, unit = self.tip_measurement()
        xy = self.physical[:, 1:] if self.physical is not None else self.result.coordinates_px
        axes.axhline(0, color="#8898a4", linewidth=0.9, linestyle="--", label="夹具出口水平线")
        axes.plot(*xy.T, color=TEAL, linewidth=2, label="鱼竿中心线")
        review = xy[self.result.review]
        if review.size:
            axes.scatter(*review.T, color="#d88334", s=15, label="待复核", zorder=3)
        tip_x, tip_y = xy[-1]
        axes.vlines(tip_x, min(0, tip_y), max(0, tip_y), color="#758797",
                    linewidth=1, linestyle=":")
        axes.scatter(tip_x, tip_y, color=TEAL, s=21, zorder=4)
        axes.set(xlabel=f"横向坐标 x（{unit}）", ylabel=f"竖直坐标 y（{unit}）", aspect="equal")
        axes.grid(color=LINE, linewidth=0.65)
        axes.spines[["top", "right"]].set_visible(False)
        axes.legend(loc="upper right", frameon=False, fontsize=9)

    def toggle_pan(self):
        self.panning = not self.panning
        cursor = "fleur" if self.panning else "crosshair" if self.picking_clamp else ""
        self.canvas.get_tk_widget().configure(cursor=cursor)
        self.update_controls()

    def reset_view(self):
        self.photo_limits = None
        self.draw_view()

    def canvas_press(self, event):
        if (self.view != "photo" or self.image is None or self.future is not None
                or event.inaxes is not self.axes):
            return
        if event.button == 3:
            if self.picking_clamp:
                self.toggle_clamp_pick()
            else:
                self.undo_point()
        elif event.button == 1:
            if self.panning:
                self.drag_origin = (event.x, event.y)
            else:
                x, y = event.xdata, event.ydata
                height, width = self.image.shape[:2]
                if not (0 <= x <= width - 1 and 0 <= y <= height - 1):
                    return
                if self.picking_clamp:
                    point = (x, y)
                    if self.clamp_start is None:
                        self.clamp_start = point
                        self.draw_view()
                        self.update_controls()
                        return
                    try:
                        angle = measure_clamp_angle(self.image, self.clamp_start, point)
                    except ValueError as error:
                        self.status_text.set(str(error))
                        return
                    self.clamp_points = (self.clamp_start, point)
                    self.clamp_angle = angle
                    self.switch_view("photo")
                    self.status_text.set("已测量夹持角。自由段弧长从夹具出口起算。")
                else:
                    self.anchors.append((x, y))
                    self.points_changed()

    def canvas_release(self, _event):
        self.drag_origin = None

    def canvas_move(self, event):
        if self.drag_origin is None or not self.panning:
            return
        before = self.axes.transData.inverted().transform(self.drag_origin)
        after = self.axes.transData.inverted().transform((event.x, event.y))
        delta = before - after
        self.drag_origin = (event.x, event.y)
        self.set_photo_limits([value + delta[0] for value in self.axes.get_xlim()],
                              [value + delta[1] for value in self.axes.get_ylim()])

    def canvas_scroll(self, event):
        if self.view != "photo" or self.image is None or event.inaxes is not self.axes:
            return
        scale = 1.2 ** -event.step
        x0, x1 = self.axes.get_xlim()
        if not 12 <= (x1 - x0) * scale <= self.image.shape[1] * 2.5:
            return
        x, y = event.xdata, event.ydata
        self.set_photo_limits([x + (v - x) * scale for v in self.axes.get_xlim()],
                              [y + (v - y) * scale for v in self.axes.get_ylim()])

    def set_photo_limits(self, xlim, ylim):
        self.photo_limits = (xlim, ylim)
        self.axes.set_xlim(xlim)
        self.axes.set_ylim(ylim)
        self.canvas.draw_idle()

    def close(self):
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="打开鱼竿曲线识别窗口。")
    parser.add_argument("photo", nargs="?", type=Path)
    args = parser.parse_args()
    root = tk.Tk()
    app = RodCurveApp(root)
    if args.photo is not None:
        app.load_photo(args.photo)
    root.mainloop()


if __name__ == "__main__":
    main()
