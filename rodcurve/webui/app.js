"use strict";

const by = id => document.getElementById(id);
const canvas = by("drawing");
const ctx = canvas.getContext("2d");
const model = by("model");
const state = {
  photo: null,
  anchors: [],
  clamp: null,
  clampStart: null,
  result: null,
  busy: false,
  view: "photo",
  mode: "point",
  camera: null,
  drag: null,
};

function setStatus(text, error = false) {
  by("status").textContent = text;
  by("status").classList.toggle("error", error);
}

async function request(route, options) {
  const response = await fetch(route, options);
  if (response.ok) return response;
  const {error} = await response.json();
  throw new Error(error);
}

function lengthValue() {
  const input = by("length");
  if (input.validity.badInput) return NaN;
  if (!input.value) return null;
  const length = input.valueAsNumber;
  return Number.isFinite(length) && length > 0 ? length : NaN;
}

function measurement() {
  const length = lengthValue();
  const end = state.result?.samples.at(-1);
  if (!end || Number.isNaN(length)) return null;
  const scale = length === null ? 1 : length / end.s;
  if (!Number.isFinite(scale) || scale <= 0) return null;
  return {unit: length === null ? "px" : "mm", scale, end};
}

function refresh() {
  const {photo, anchors, clamp, clampStart, mode, view, result, busy} = state;
  const lengthError = Number.isNaN(lengthValue());
  const data = measurement();
  const hasMillimetres = data?.unit === "mm";
  const modelReady = typeof model.contentWindow?.importRecognizedCurve === "function";
  by("photo-name").textContent = photo?.file.name ?? "尚未导入照片";
  by("placeholder").hidden = Boolean(photo);
  by("point-count").textContent = anchors.length >= 2 ? "已选 " + anchors.length + " 点，末点为竿梢。"
    : anchors.length === 1 ? "已选夹具出口，继续选择引导点或竿梢。" : "至少选择夹具出口和竿梢两点。";
  by("angle-note").textContent = mode === "angle"
    ? (clampStart ? "再点击靠竿梢侧的点。" : "点击靠竿柄侧的点。")
    : clamp ? clamp.angle.toFixed(2) + "°（向竿梢侧上仰为正）" : "以照片水平线为基准。";
  by("length-note").textContent = lengthError
    ? "自由段弧长必须是以毫米为单位的正数。"
    : "填夹具出口至竿梢沿竿身的实测长度。留空时使用像素。";
  by("open-photo").disabled = busy;
  by("undo").disabled = by("clear").disabled = busy || !anchors.length;
  by("pick-angle").disabled = busy || !photo;
  by("pick-angle").textContent = mode === "angle" ? "取消选点" : "测量夹持角";
  by("pick-angle").setAttribute("aria-pressed", String(mode === "angle"));
  by("clear-angle").disabled = busy || !clamp;
  by("extract").disabled = busy || anchors.length < 2 || lengthError;
  by("curve-view").disabled = !result;
  by("pan").disabled = busy || !photo || view !== "photo";
  by("reset-view").disabled = !photo || view !== "photo";
  by("export-curve").disabled = busy || !data;
  by("send-curve").disabled = busy || !hasMillimetres || !modelReady;
  by("send-curve").title = !hasMillimetres ? "请先填写自由段实测弧长并提取曲线。"
    : modelReady ? "" : "正在准备力学模型。";
  for (const [id, selected] of [["photo-view", view === "photo"], ["curve-view", view === "curve"], ["pan", mode === "pan"]]) {
    by(id).setAttribute("aria-pressed", String(selected));
  }
  canvas.style.cursor = view !== "photo" ? "" : mode === "pan" ? (state.drag ? "grabbing" : "grab")
    : mode === "angle" ? "crosshair" : "";
  if (data) {
    const x = data.end.x * data.scale, y = data.end.y * data.scale;
    by("tip-distance").textContent = Math.abs(y).toFixed(2) + " " + data.unit;
    by("tip-coordinates").textContent = "竿梢 (" + x.toFixed(2) + ", " + y.toFixed(2) + ") " + data.unit
      + " · " + (y < 0 ? "水平面下方" : y > 0 ? "水平面上方" : "同一高度");
  } else {
    by("tip-distance").textContent = "—";
    by("tip-coordinates").textContent = lengthError ? "请填写有效的长度。" : "完成曲线提取后显示";
  }
  draw();
}

function switchPage(page) {
  by("recognition").hidden = page !== "recognition";
  model.hidden = page !== "model";
  by("recognition-page").setAttribute("aria-pressed", String(page === "recognition"));
  by("model-page").setAttribute("aria-pressed", String(page === "model"));
  if (page === "recognition") refresh();
}

function switchView(view, mode = "point") {
  state.view = view;
  state.mode = mode;
  state.clampStart = null;
  state.drag = null;
  refresh();
}

async function openPhoto(file) {
  if (!file || state.busy) return;
  state.busy = true;
  setStatus("正在读取照片…");
  refresh();
  try {
    const response = await request("preview", {method: "POST", body: file});
    const bitmap = await createImageBitmap(await response.blob());
    state.photo?.bitmap.close();
    state.photo = {file, bitmap};
    state.anchors = [];
    state.clamp = state.result = state.camera = null;
    switchView("photo");
    setStatus("左键选点，右键撤销，滚轮以指针为中心缩放。");
  } catch (error) {
    setStatus("无法读取照片：" + error.message, true);
  } finally {
    state.busy = false;
    refresh();
  }
}

async function extract() {
  if (by("extract").disabled) return;
  state.busy = true;
  switchView("photo");
  by("extract").textContent = "正在提取…";
  setStatus("正在追踪竿身和细梢…");
  try {
    const response = await request("extract", {
      method: "POST",
      headers: {"X-Rod-Anchors": JSON.stringify(state.anchors)},
      body: state.photo.file,
    });
    state.result = await response.json();
    const samples = state.result.samples;
    const review = samples.filter(point => point.review).length;
    setStatus("已提取 " + samples.length + " 个点 · " + review + " 个橙色点待复核");
  } catch (error) {
    setStatus("曲线提取失败：" + error.message, true);
  } finally {
    state.busy = false;
    by("extract").textContent = "提取曲线";
    refresh();
  }
}

function pointsChanged() {
  state.result = null;
  refresh();
  setStatus("左键选点，右键撤销，滚轮以指针为中心缩放。");
}

function undo() {
  if (by("undo").disabled) return;
  state.anchors.pop();
  pointsChanged();
}

function sendCurve() {
  if (by("send-curve").disabled) return;
  const {scale} = measurement();
  const {samples} = state.result;
  try {
    model.contentWindow.importRecognizedCurve({
      name: state.photo.file.name,
      points: samples.map(({s, x, y}) => [s * scale, x * scale, y * scale]),
      reviewCount: samples.filter(point => point.review).length,
      clampAngle: state.clamp?.angle ?? null,
    });
    switchPage("model");
    model.contentWindow.focus();
    setStatus("已将 " + state.photo.file.name + " 的实测曲线送入模型对照。");
  } catch (error) {
    setStatus("无法送入模型：" + error.message, true);
  }
}

async function saveRodCsv(filename, text) {
  const blob = new Blob(["\uFEFF", text], {type: "text/csv;charset=utf-8"});
  if (typeof window.showSaveFilePicker === "function") {
    try {
      const handle = await window.showSaveFilePicker({
        suggestedName: filename,
        types: [{description: "CSV 文件", accept: {"text/csv": [".csv"]}}],
      });
      const writer = await handle.createWritable();
      await writer.write(blob);
      await writer.close();
      return true;
    } catch (error) {
      if (error.name === "AbortError") return false;
      if (error.name !== "SecurityError") throw error;
    }
  }
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  return true;
}

async function exportCurve() {
  if (by("export-curve").disabled) return;
  const {scale, unit} = measurement();
  const headings = ["s_px", "x_px", "y_px", "image_u_px", "image_v_px", "evidence", "needs_review"];
  if (unit === "mm") headings.push("s_mm", "x_mm", "y_mm");
  if (state.clamp) headings.push("clamp_angle_deg");
  const rows = state.result.samples.map(({s, x, y, u, v, evidence, review}) => {
    const row = [s, x, y, u, v, evidence].map(value => value.toFixed(5));
    row.push(Number(review));
    if (unit === "mm") row.push(...[s, x, y].map(value => (value * scale).toFixed(5)));
    if (state.clamp) row.push(state.clamp.angle.toFixed(5));
    return row.join(",");
  });
  const name = state.photo.file.name.replace(/\.[^.]+$/, "") + "_中心线.csv";
  try {
    if (await saveRodCsv(name, [headings.join(","), ...rows].join("\r\n") + "\r\n")) {
      setStatus("已保存 " + name);
    }
  } catch (error) {
    setStatus("CSV 保存失败：" + error.message, true);
  }
}

function photoTransform(width, height) {
  const {bitmap} = state.photo;
  const camera = state.camera ?? {x: bitmap.width / 2, y: bitmap.height / 2, zoom: 1};
  const scale = Math.min((width - 18) / bitmap.width, (height - 18) / bitmap.height) * camera.zoom;
  return {...camera, scale, left: width / 2 - camera.x * scale, top: height / 2 - camera.y * scale};
}

function position(event) {
  const {left, top, width, height} = canvas.getBoundingClientRect();
  return {x: event.clientX - left, y: event.clientY - top, width, height};
}

function line(points, project, color, width, dash = []) {
  ctx.save();
  ctx.beginPath();
  ctx.setLineDash(dash);
  for (const point of points) ctx.lineTo(...project(point));
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
  ctx.restore();
}

function dot(x, y, color, radius, outline = false) {
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, 2 * Math.PI);
  ctx.fillStyle = color;
  ctx.fill();
  if (outline) {
    ctx.lineWidth = 1;
    ctx.strokeStyle = "#fff";
    ctx.stroke();
  }
}

function label(text, x, y) {
  ctx.font = '12px "Microsoft YaHei", sans-serif';
  const wide = ctx.measureText(text).width + 10;
  ctx.fillStyle = "#0d6f61";
  ctx.fillRect(x - wide / 2, y - 23, wide, 17);
  ctx.fillStyle = "#fff";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, x, y - 14);
}

function drawPhoto(width, height) {
  if (!state.photo) return;
  const {bitmap} = state.photo;
  const {left, top, scale} = photoTransform(width, height);
  const project = ([u, v]) => [left + (u + .5) * scale, top + (v + .5) * scale];
  ctx.drawImage(bitmap, left, top, bitmap.width * scale, bitmap.height * scale);
  if (state.result) {
    line(state.result.fitted, project, "#19e6c3", 1.7);
    for (const {u, v, review} of state.result.samples) {
      if (review) dot(...project([u, v]), "#ffa028", 2.35);
    }
  }
  if (state.clamp && state.mode !== "angle") {
    const {back, front} = state.clamp;
    line([[back[0], front[1]], front], project, "#fff", 1.4, [5, 4]);
    line([back, front], project, "#d194ff", 2);
    dot(...project(back), "#d194ff", 4, true);
    dot(...project(front), "#d194ff", 4, true);
  }
  if (state.clampStart) dot(...project(state.clampStart), "#d194ff", 4, true);
  state.anchors.forEach((point, index) => {
    const [x, y] = project(point);
    dot(x, y, "#0d6f61", 4.5, true);
    label(index === 0 ? "夹具出口" : index === state.anchors.length - 1 ? "竿梢" : "引导点 " + index, x, y);
  });
}

function ticks(min, max) {
  const step = (max - min) / 5;
  if (!Number.isFinite(step) || step <= 0) return [];
  const base = 10 ** Math.floor(Math.log10(step));
  if (!base) return [];
  const spacing = ([1, 2, 5, 10].find(value => value >= step / base)) * base;
  const values = [];
  for (let i = Math.ceil(min / spacing); i <= Math.floor(max / spacing); i++) values.push(i * spacing);
  return values;
}

function drawCurve(width, height) {
  const data = measurement();
  if (!data) return;
  const points = state.result.samples.map(({x, y}) => [x * data.scale, y * data.scale]);
  let xMin = 0, xMax = 0, yMin = 0, yMax = 0;
  for (const [x, y] of points) {
    xMin = Math.min(xMin, x); xMax = Math.max(xMax, x);
    yMin = Math.min(yMin, y); yMax = Math.max(yMax, y);
  }
  const marginX = 58, marginY = 49;
  if (width <= 2 * marginX || height <= 2 * marginY) return;
  const plotWidth = width - 2 * marginX, plotHeight = height - 2 * marginY;
  const scale = Math.min(plotWidth / ((xMax - xMin) || 1), plotHeight / ((yMax - yMin) || 1)) * .94;
  if (!Number.isFinite(scale) || scale <= 0) return;
  const midX = (xMin + xMax) / 2, midY = (yMin + yMax) / 2;
  const left = midX - plotWidth / (2 * scale), right = midX + plotWidth / (2 * scale);
  const bottom = midY - plotHeight / (2 * scale), top = midY + plotHeight / (2 * scale);
  const project = ([x, y]) => [width / 2 + (x - midX) * scale, height / 2 - (y - midY) * scale];
  ctx.font = '12px "Microsoft YaHei", sans-serif';
  ctx.fillStyle = "#46635f";
  ctx.textBaseline = "middle";
  ctx.textAlign = "center";
  for (const x of ticks(left, right)) {
    line([[x, bottom], [x, top]], project, "#dce8e3", 1);
    ctx.fillText(Number(x.toPrecision(5)), project([x, bottom])[0], height - marginY + 12);
  }
  ctx.textAlign = "right";
  for (const y of ticks(bottom, top)) {
    line([[left, y], [right, y]], project, "#dce8e3", 1);
    ctx.fillText(Number(y.toPrecision(5)), marginX - 6, project([left, y])[1]);
  }
  ctx.textAlign = "center";
  ctx.fillText("横向坐标 x（" + data.unit + "）", width / 2, height - 12);
  ctx.save();
  ctx.translate(11, height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("竖直坐标 y（" + data.unit + "）", 0, 0);
  ctx.restore();

  line([[left, 0], [right, 0]], project, "#829b94", 1, [5, 4]);
  line(points, project, "#0d6f61", 2);
  const tip = points.at(-1);
  line([[tip[0], 0], tip], project, "#829b94", 1, [2, 3]);
  state.result.samples.forEach(({review}, index) => {
    if (review) dot(...project(points[index]), "#d88334", 2);
  });
  dot(...project(tip), "#0d6f61", 3.5);
}

function draw() {
  const {width, height} = canvas.getBoundingClientRect();
  if (width <= 18 || height <= 18) return;
  const ratio = window.devicePixelRatio || 1;
  const pixelWidth = Math.round(width * ratio), pixelHeight = Math.round(height * ratio);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  if (state.view === "photo") drawPhoto(width, height);
  else drawCurve(width, height);
}

canvas.addEventListener("pointerdown", event => {
  if (event.button !== 0 || !state.photo || state.busy || state.view !== "photo") return;
  const {x, y, width, height} = position(event);
  const frame = photoTransform(width, height);
  if (state.mode === "pan") {
    state.drag = {x, y};
    canvas.setPointerCapture(event.pointerId);
    refresh();
    return;
  }
  const point = [(x - frame.left) / frame.scale - .5, (y - frame.top) / frame.scale - .5];
  const {bitmap} = state.photo;
  if (point[0] < 0 || point[1] < 0 || point[0] > bitmap.width - 1 || point[1] > bitmap.height - 1) return;
  if (state.mode !== "angle") {
    state.anchors.push(point);
    pointsChanged();
  } else if (!state.clampStart) {
    state.clampStart = point;
    refresh();
  } else {
    const back = state.clampStart;
    const dx = Math.abs(point[0] - back[0]), dy = back[1] - point[1];
    if (Math.hypot(dx, dy) < 8) {
      setStatus("夹持角的两个点至少相距 8 像素。", true);
      return;
    }
    state.clamp = {back, front: point, angle: Math.atan2(dy, dx) * 180 / Math.PI};
    switchView("photo");
    setStatus("已测量夹持角。自由段弧长从夹具出口起算。");
  }
});

canvas.addEventListener("pointermove", event => {
  if (!state.drag) return;
  const {x, y, width, height} = position(event);
  const frame = photoTransform(width, height);
  state.camera = {x: frame.x - (x - state.drag.x) / frame.scale,
    y: frame.y - (y - state.drag.y) / frame.scale, zoom: frame.zoom};
  state.drag = {x, y};
  draw();
});
canvas.addEventListener("lostpointercapture", () => {
  state.drag = null;
  refresh();
});

canvas.addEventListener("wheel", event => {
  if (!state.photo || state.view !== "photo") return;
  event.preventDefault();
  if (!event.deltaY) return;
  const {x, y, width, height} = position(event);
  const frame = photoTransform(width, height);
  const zoom = Math.min(80, Math.max(.4, frame.zoom * 1.2 ** -Math.sign(event.deltaY)));
  const nextScale = frame.scale * zoom / frame.zoom;
  state.camera = {x: (x - frame.left) / frame.scale - (x - width / 2) / nextScale,
    y: (y - frame.top) / frame.scale - (y - height / 2) / nextScale, zoom};
  draw();
}, {passive: false});

canvas.addEventListener("contextmenu", event => {
  if (!state.photo || state.view !== "photo") return;
  event.preventDefault();
  if (state.mode === "angle") by("pick-angle").click();
  else undo();
});

by("recognition-page").onclick = () => switchPage("recognition");
by("model-page").onclick = () => switchPage("model");
model.addEventListener("load", refresh);
by("open-photo").onclick = () => by("photo-file").click();
by("photo-file").onchange = event => {
  const input = event.currentTarget;
  const file = input.files[0];
  input.value = "";
  openPhoto(file);
};
by("undo").onclick = undo;
by("clear").onclick = () => { state.anchors = []; pointsChanged(); };
by("pick-angle").onclick = () => {
  const selecting = state.mode !== "angle";
  switchView("photo", selecting ? "angle" : "point");
  setStatus(selecting ? "从竿柄侧朝竿梢侧依次选两点，可沿与夹持段平行的参照边选点。"
    : "已退出夹持角选点。");
};
by("clear-angle").onclick = () => {
  state.clamp = null;
  switchView(state.view);
  setStatus("已移除夹持角。");
};
by("photo-view").onclick = () => switchView("photo");
by("curve-view").onclick = () => switchView("curve");
by("pan").onclick = () => switchView("photo", state.mode === "pan" ? "point" : "pan");
by("reset-view").onclick = () => { state.camera = null; draw(); };
by("length").oninput = refresh;
by("extract").onclick = extract;
by("send-curve").onclick = sendCurve;
by("export-curve").onclick = exportCurve;
new ResizeObserver(draw).observe(canvas);

document.addEventListener("keydown", event => {
  if (by("recognition").hidden || !(event.ctrlKey || event.metaKey)) return;
  const key = event.key.toLowerCase();
  if (key === "o" || key === "s") {
    event.preventDefault();
    by(key === "o" ? "open-photo" : "export-curve").click();
  }
});

async function openStartupPhoto() {
  try {
    const {photo} = await (await request("startup")).json();
    if (!photo) return;
    const original = await (await request("initial-photo")).blob();
    if (!state.photo && !state.busy) await openPhoto(new File([original], photo));
  } catch (error) {
    setStatus("无法读取启动时指定的照片：" + error.message, true);
  }
}

refresh();
openStartupPhoto();
