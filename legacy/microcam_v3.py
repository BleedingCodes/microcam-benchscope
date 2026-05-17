# MicroCam Ultra Inspection Workstation

Single-file Python/OpenCV electronics microscope workstation.

```python
#!/usr/bin/env python3
"""
MicroCam Ultra Inspection Workstation

A single-file OpenCV/V4L2 microscope viewer for Linux USB capture cards.
Designed for electronics inspection, PCB repair, solder inspection, measurement,
snapshots, recording, and visual analysis.

Tested conceptually with MacroSilicon USB Video capture devices on Linux.
"""

import cv2
import time
import os
import json
import math
import csv
import shutil
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Tuple, List

import numpy as np

# ============================================================
# USER CONFIG
# ============================================================

DEVICE = "/dev/video0"
WINDOW_NAME = "MicroCam Ultra"

DEFAULT_WIDTH = 1440
DEFAULT_HEIGHT = 1180
DEFAULT_FPS = 35
DEFAULT_FOURCC = "MJPG"

CAPTURE_DIR = "captures"
CONFIG_FILE = "microcam_ultra_config.json"
MEASUREMENTS_CSV = os.path.join(CAPTURE_DIR, "measurements.csv")

# Known calibration default: DIP/header pitch = 2.54 mm = 100 mil
DEFAULT_PIXELS_PER_MM = 15.08
DEFAULT_KNOWN_MM = 2.54

MAX_ZOOM = 12.0
ZOOM_STEP = 0.20
PAN_STEP = 50

# ============================================================
# STATE
# ============================================================

@dataclass
class AppState:
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    fps: int = DEFAULT_FPS
    fourcc: str = DEFAULT_FOURCC

    zoom: float = 1.0
    pan_x: int = 0
    pan_y: int = 0

    brightness_beta: int = 20
    contrast_alpha: float = 1.0
    gamma: float = 1.0

    pixels_per_mm: float = DEFAULT_PIXELS_PER_MM
    known_mm: float = DEFAULT_KNOWN_MM

    paused: bool = False
    fullscreen: bool = False
    recording: bool = False

    show_help: bool = False
    show_hud: bool = True
    clean_snapshot: bool = False
    record_clean: bool = False

    crosshair: bool = True
    grid: bool = False
    ruler: bool = True
    magnifier: bool = False
    focus_meter: bool = True
    histogram: bool = False

    grayscale: bool = False
    invert: bool = False
    edge: bool = False
    threshold: bool = False
    adaptive_threshold: bool = False
    contrast_enhance: bool = False
    sharpen: bool = False
    denoise: bool = False
    false_color: bool = False

    threshold_value: int = 120

    measure_mode: bool = False
    dragging_measure: bool = False

    last_snapshot_path: str = ""
    last_recording_path: str = ""


state = AppState()

# Runtime objects
cap = None
writer = None
last_frame = None
last_record_toggle = 0.0
prev_time = time.time()
fps_smoothed = 0.0

# Mouse/measurement state
point1: Optional[Tuple[int, int]] = None
point2: Optional[Tuple[int, int]] = None
mouse_pos: Tuple[int, int] = (0, 0)
locked_measurements: List[Tuple[Tuple[int, int], Tuple[int, int], str]] = []

# Clean frame caches for saving/recording
raw_frame_cache = None
processed_clean_cache = None
display_frame_cache = None

# ============================================================
# FILE/CONFIG HELPERS
# ============================================================

def ensure_dirs():
    os.makedirs(CAPTURE_DIR, exist_ok=True)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_config():
    data = asdict(state)
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Config saved: {CONFIG_FILE}")
    except Exception as e:
        print(f"Config save failed: {e}")


def load_config():
    if not os.path.exists(CONFIG_FILE):
        return

    try:
        with open(CONFIG_FILE, "r") as f:
            data = json.load(f)

        for key, value in data.items():
            if hasattr(state, key):
                setattr(state, key, value)

        print(f"Loaded config: {CONFIG_FILE}")
    except Exception as e:
        print(f"Config load failed: {e}")


def append_measurement_csv(label: str, pixels: float, mm: float, mils: float):
    exists = os.path.exists(MEASUREMENTS_CSV)
    with open(MEASUREMENTS_CSV, "a", newline="") as f:
        writer_csv = csv.writer(f)
        if not exists:
            writer_csv.writerow(["timestamp", "label", "pixels", "mm", "mils", "px_per_mm", "zoom"])
        writer_csv.writerow([
            datetime.now().isoformat(timespec="seconds"),
            label,
            f"{pixels:.3f}",
            f"{mm:.6f}",
            f"{mils:.3f}",
            f"{state.pixels_per_mm:.6f}",
            f"{state.zoom:.3f}",
        ])
    print(f"Measurement logged: {MEASUREMENTS_CSV}")

# ============================================================
# CAMERA HELPERS
# ============================================================

def open_camera():
    global cap

    if cap is not None:
        cap.release()

    cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*state.fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, state.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, state.height)
    cap.set(cv2.CAP_PROP_FPS, state.fps)

    if not cap.isOpened():
        print("Failed to open camera")
        raise SystemExit(1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Camera opened: {DEVICE}")
    print(f"Requested: {state.width}x{state.height}@{state.fps} {state.fourcc}")
    print(f"Actual:    {actual_w}x{actual_h}@{actual_fps:.1f}")


def set_v4l2_control(name: str, value):
    cmd = ["v4l2-ctl", f"--device={DEVICE}", f"--set-ctrl={name}={value}"]
    try:
        subprocess.run(cmd, check=False)
    except FileNotFoundError:
        print("v4l2-ctl not found. Install with: sudo apt install v4l-utils")


def query_v4l2_controls():
    try:
        subprocess.run(["v4l2-ctl", f"--device={DEVICE}", "--list-ctrls"])
    except FileNotFoundError:
        print("v4l2-ctl not found. Install with: sudo apt install v4l-utils")

# ============================================================
# IMAGE PROCESSING
# ============================================================

def apply_gamma(frame, gamma: float):
    if abs(gamma - 1.0) < 0.001:
        return frame
    inv = 1.0 / max(0.01, gamma)
    table = np.array([(i / 255.0) ** inv * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(frame, table)


def focus_score(frame) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def apply_zoom_pan(frame):
    if state.zoom <= 1.0:
        return frame

    h, w = frame.shape[:2]
    new_w = max(1, int(w / state.zoom))
    new_h = max(1, int(h / state.zoom))

    max_x = w - new_w
    max_y = h - new_h

    x1 = max(0, min((w - new_w) // 2 + state.pan_x, max_x))
    y1 = max(0, min((h - new_h) // 2 + state.pan_y, max_y))

    cropped = frame[y1:y1 + new_h, x1:x1 + new_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def process_frame(raw):
    frame = raw.copy()

    # Basic software brightness/contrast/gamma
    frame = cv2.convertScaleAbs(frame, alpha=state.contrast_alpha, beta=state.brightness_beta)
    frame = apply_gamma(frame, state.gamma)

    if state.denoise:
        frame = cv2.fastNlMeansDenoisingColored(frame, None, 4, 4, 7, 15)

    if state.contrast_enhance:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        lab2 = cv2.merge((l2, a, b))
        frame = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    if state.grayscale:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if state.invert:
        frame = cv2.bitwise_not(frame)

    if state.sharpen:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        frame = cv2.filter2D(frame, -1, kernel)

    if state.edge:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blur, 40, 100)
        frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    if state.threshold:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, state.threshold_value, 255, cv2.THRESH_BINARY)
        frame = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    if state.adaptive_threshold:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            5,
        )
        frame = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    if state.false_color:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)

    frame = apply_zoom_pan(frame)
    return frame

# ============================================================
# DRAWING HELPERS
# ============================================================

def draw_text(frame, text, x, y, scale=0.55, color=(255, 255, 255), thickness=1):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def draw_hud(frame, fps_actual, focus):
    if not state.show_hud:
        return

    mode_bits = []
    if state.recording: mode_bits.append("REC")
    if state.paused: mode_bits.append("PAUSED")
    if state.clean_snapshot: mode_bits.append("CLEAN-SNAP")
    if state.record_clean: mode_bits.append("CLEAN-REC")
    if state.measure_mode: mode_bits.append("MEASURE")
    if state.edge: mode_bits.append("EDGE")
    if state.threshold: mode_bits.append(f"THR:{state.threshold_value}")
    if state.adaptive_threshold: mode_bits.append("ADAPTIVE")
    if state.grayscale: mode_bits.append("GRAY")
    if state.invert: mode_bits.append("INV")
    if state.false_color: mode_bits.append("COLOR")

    draw_text(frame, f"{state.width}x{state.height}  FPS:{fps_actual:.1f}  Zoom:{state.zoom:.2f}x  Focus:{focus:.0f}", 20, 35, 0.65, (100, 255, 10), 2)
    draw_text(frame, f"Cal:{state.pixels_per_mm:.2f} px/mm  Known:{state.known_mm:.2f}mm  { ' | '.join(mode_bits) }", 20, 65, 0.55, (255, 255, 255), 1)

    if state.recording:
        cv2.circle(frame, (32, 105), 10, (0, 0, 255), -1)
        draw_text(frame, "REC", 50, 112, 0.75, (0, 0, 255), 2)


def draw_crosshair(frame):
    if not state.crosshair:
        return

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx - 45, cy), (cx + 45, cy), (0, 255, 255), 1)
    cv2.line(frame, (cx, cy - 45), (cx, cy + 45), (0, 255, 255), 1)
    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)


def draw_grid(frame):
    if not state.grid:
        return

    h, w = frame.shape[:2]
    spacing = 100
    for x in range(0, w, spacing):
        cv2.line(frame, (x, 0), (x, h), (75, 75, 75), 1)
        draw_text(frame, str(x), x + 3, 16, 0.35, (160, 160, 160), 1)
    for y in range(0, h, spacing):
        cv2.line(frame, (0, y), (w, y), (75, 75, 75), 1)
        draw_text(frame, str(y), 3, max(16, y - 3), 0.35, (160, 160, 160), 1)


def draw_ruler(frame):
    if not state.ruler:
        return

    h, w = frame.shape[:2]
    px_len = int(state.pixels_per_mm * 5)  # 5 mm ruler
    px_len = max(10, min(px_len, w - 80))

    x0 = 40
    y0 = h - 45
    cv2.line(frame, (x0, y0), (x0 + px_len, y0), (255, 255, 255), 2)
    cv2.line(frame, (x0, y0 - 8), (x0, y0 + 8), (255, 255, 255), 2)
    cv2.line(frame, (x0 + px_len, y0 - 8), (x0 + px_len, y0 + 8), (255, 255, 255), 2)
    draw_text(frame, "5 mm", x0, y0 - 14, 0.5, (255, 255, 255), 1)


def draw_magnifier(frame):
    if not state.magnifier:
        return

    h, w = frame.shape[:2]
    mag_size = 140
    zoom_factor = 3
    cx, cy = w // 2, h // 2

    x1 = max(0, cx - mag_size // 2)
    y1 = max(0, cy - mag_size // 2)
    x2 = min(w, x1 + mag_size)
    y2 = min(h, y1 + mag_size)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return

    magnified = cv2.resize(roi, (mag_size * zoom_factor, mag_size * zoom_factor), interpolation=cv2.INTER_NEAREST)
    mh, mw = magnified.shape[:2]

    dst_x1 = w - mw - 15
    dst_y1 = 15
    dst_x2 = w - 15
    dst_y2 = 15 + mh

    if dst_x1 < 0 or dst_y2 > h:
        return

    frame[dst_y1:dst_y2, dst_x1:dst_x2] = magnified
    cv2.rectangle(frame, (dst_x1, dst_y1), (dst_x2, dst_y2), (0, 255, 255), 2)
    draw_text(frame, "MAG", dst_x1 + 8, dst_y1 + 24, 0.6, (0, 255, 255), 2)


def draw_histogram(frame):
    if not state.histogram:
        return

    h, w = frame.shape[:2]
    hist_w, hist_h = 256, 100
    x0, y0 = w - hist_w - 20, h - hist_h - 20

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist, 0, hist_h, cv2.NORM_MINMAX)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + hist_w, y0 + hist_h), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    for i in range(1, 256):
        cv2.line(
            frame,
            (x0 + i - 1, y0 + hist_h - int(hist[i - 1][0])),
            (x0 + i, y0 + hist_h - int(hist[i][0])),
            (200, 200, 200),
            1,
        )

    cv2.rectangle(frame, (x0, y0), (x0 + hist_w, y0 + hist_h), (255, 255, 255), 1)


def measurement_values(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    pixels = math.hypot(dx, dy)
    mm = pixels / state.pixels_per_mm if state.pixels_per_mm else 0.0
    mils = mm * 39.3701
    return pixels, mm, mils


def draw_measurements(frame):
    # Locked measurements
    for p1, p2, label in locked_measurements:
        draw_measurement_line(frame, p1, p2, label, color=(255, 0, 255))

    # Active measurement
    if point1 is not None:
        cv2.circle(frame, point1, 5, (0, 255, 0), -1)

    if point1 is not None and point2 is not None:
        draw_measurement_line(frame, point1, point2, "", color=(255, 0, 0))


def draw_measurement_line(frame, p1, p2, label="", color=(255, 0, 0)):
    cv2.circle(frame, p1, 5, (0, 255, 0), -1)
    cv2.circle(frame, p2, 5, (0, 255, 0), -1)
    cv2.line(frame, p1, p2, color, 2)

    pixels, mm, mils = measurement_values(p1, p2)
    text = f"{label} {pixels:.1f}px | {mm:.3f} mm | {mils:.1f} mil".strip()

    tx = min(max(10, p1[0]), frame.shape[1] - 500)
    ty = max(25, p1[1] - 10)
    draw_text(frame, text, tx, ty, 0.55, (255, 255, 0), 2)


def draw_focus_meter(frame, focus):
    if not state.focus_meter:
        return

    h, w = frame.shape[:2]
    x0, y0 = 20, h - 80
    bar_w, bar_h = 260, 18

    normalized = max(0.0, min(focus / 2000.0, 1.0))
    fill = int(bar_w * normalized)

    cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (80, 80, 80), 1)
    cv2.rectangle(frame, (x0, y0), (x0 + fill, y0 + bar_h), (0, 220, 255), -1)
    draw_text(frame, f"Focus {focus:.0f}", x0, y0 - 8, 0.5, (0, 220, 255), 1)


def draw_help(frame):
    if not state.show_help:
        return

    help_lines = [
        "q quit | h help | H HUD",
        "p snapshot | v clean snapshot | r record | R clean record",
        "f fullscreen | space freeze",
        "+/- or wheel zoom | WASD pan | 0 reset view",
        "c crosshair | g grid | M magnifier | U ruler | O histogram",
        "b grayscale | i invert | e edge | y threshold | Y adaptive threshold",
        ",/. threshold adjust | n denoise | C contrast enhance | S sharpen | F false color",
        "t measure mode | drag mouse to measure | x clear | k calibrate",
        "l lock measurement | u undo locked | L log current measurement",
        "[/] known distance adjust | { } big adjust",
        "1 640x480@60 | 2 800x600@60 | 3 1280x720@60 | 4 1920x1080@30",
        "B/V software brightness -/+ | A/D contrast -/+ | G/T gamma -/+",
        "z print v4l2 controls | save config auto on quit",
    ]

    x, y = 20, 130
    line_h = 25
    box_w = 860
    box_h = len(help_lines) * line_h + 20

    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 10, y - 25), (x + box_w, y + box_h), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    for line in help_lines:
        draw_text(frame, line, x, y, 0.55, (255, 255, 255), 1)
        y += line_h

# ============================================================
# OUTPUT HELPERS
# ============================================================

def start_recording(frame):
    global writer

    filename = os.path.join(CAPTURE_DIR, f"recording_{timestamp()}.avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(filename, fourcc, state.fps, (frame.shape[1], frame.shape[0]))

    if writer.isOpened():
        state.recording = True
        state.last_recording_path = filename
        print(f"Recording started: {filename}")
    else:
        writer = None
        state.recording = False
        print("Failed to start recording")


def stop_recording():
    global writer
    if writer is not None:
        writer.release()
        writer = None
    state.recording = False
    print("Recording stopped")


def save_snapshot():
    if state.clean_snapshot and processed_clean_cache is not None:
        img = processed_clean_cache.copy()
    elif display_frame_cache is not None:
        img = display_frame_cache.copy()
    else:
        print("No frame to save")
        return

    filename = os.path.join(CAPTURE_DIR, f"snapshot_{timestamp()}.png")
    cv2.imwrite(filename, img)
    state.last_snapshot_path = filename
    print(f"Snapshot saved: {filename}")

# ============================================================
# INPUT HANDLERS
# ============================================================

def reset_view():
    state.zoom = 1.0
    state.pan_x = 0
    state.pan_y = 0
    print("View reset")


def set_resolution(width, height, fps):
    state.width = width
    state.height = height
    state.fps = fps
    reset_view()
    open_camera()


def mouse_callback(event, x, y, flags, param):
    global point1, point2, mouse_pos

    mouse_pos = (x, y)

    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            state.zoom = min(state.zoom + ZOOM_STEP, MAX_ZOOM)
        else:
            state.zoom = max(1.0, state.zoom - ZOOM_STEP)
        return

    if state.measure_mode:
        if event == cv2.EVENT_LBUTTONDOWN:
            point1 = (x, y)
            point2 = (x, y)
            state.dragging_measure = True

        elif event == cv2.EVENT_MOUSEMOVE and state.dragging_measure:
            point2 = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            point2 = (x, y)
            state.dragging_measure = False


def handle_key(key):
    global point1, point2, last_record_toggle

    if key == 255:
        return True

    if key == ord('q'):
        return False

    elif key == ord('h'):
        state.show_help = not state.show_help

    elif key == ord('H'):
        state.show_hud = not state.show_hud

    elif key == ord('p'):
        save_snapshot()

    elif key == ord('v'):
        state.clean_snapshot = not state.clean_snapshot
        print("Clean snapshot ON" if state.clean_snapshot else "Clean snapshot OFF")

    elif key == ord('R'):
        state.record_clean = not state.record_clean
        print("Clean recording ON" if state.record_clean else "Clean recording OFF")

    elif key == ord('r'):
        now = time.time()
        if now - last_record_toggle > 0.4:
            last_record_toggle = now
            if state.recording:
                stop_recording()
            else:
                frame_for_size = processed_clean_cache if state.record_clean else display_frame_cache
                if frame_for_size is not None:
                    start_recording(frame_for_size)

    elif key == ord('f'):
        state.fullscreen = not state.fullscreen
        cv2.setWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if state.fullscreen else cv2.WINDOW_NORMAL,
        )

    elif key == ord(' '):
        state.paused = not state.paused
        print("Paused" if state.paused else "Live")

    elif key in (ord('+'), ord('=')):
        state.zoom = min(state.zoom + ZOOM_STEP, MAX_ZOOM)

    elif key == ord('-'):
        state.zoom = max(1.0, state.zoom - ZOOM_STEP)

    elif key == ord('0'):
        reset_view()

    elif key == ord('w'):
        state.pan_y -= int(PAN_STEP / max(1.0, state.zoom))
    elif key == ord('s'):
        state.pan_y += int(PAN_STEP / max(1.0, state.zoom))
    elif key == ord('a'):
        state.pan_x -= int(PAN_STEP / max(1.0, state.zoom))
    elif key == ord('d'):
        state.pan_x += int(PAN_STEP / max(1.0, state.zoom))

    elif key == ord('c'):
        state.crosshair = not state.crosshair
    elif key == ord('g'):
        state.grid = not state.grid
    elif key == ord('M'):
        state.magnifier = not state.magnifier
    elif key == ord('U'):
        state.ruler = not state.ruler
    elif key == ord('O'):
        state.histogram = not state.histogram

    elif key == ord('b'):
        state.grayscale = not state.grayscale
    elif key == ord('i'):
        state.invert = not state.invert
    elif key == ord('e'):
        state.edge = not state.edge
    elif key == ord('y'):
        state.threshold = not state.threshold
    elif key == ord('Y'):
        state.adaptive_threshold = not state.adaptive_threshold
    elif key == ord('n'):
        state.denoise = not state.denoise
    elif key == ord('C'):
        state.contrast_enhance = not state.contrast_enhance
    elif key == ord('S'):
        state.sharpen = not state.sharpen
    elif key == ord('F'):
        state.false_color = not state.false_color

    elif key == ord(','):
        state.threshold_value = max(0, state.threshold_value - 5)
        print(f"Threshold: {state.threshold_value}")
    elif key == ord('.'):
        state.threshold_value = min(255, state.threshold_value + 5)
        print(f"Threshold: {state.threshold_value}")

    elif key == ord('t'):
        state.measure_mode = not state.measure_mode
        point1 = None
        point2 = None
        print("Measure mode ON" if state.measure_mode else "Measure mode OFF")

    elif key == ord('x'):
        point1 = None
        point2 = None
        print("Measurement cleared")

    elif key == ord('k'):
        if point1 is not None and point2 is not None:
            pixels, mm, mils = measurement_values(point1, point2)
            if pixels > 0:
                state.pixels_per_mm = pixels / state.known_mm
                print(f"Calibrated: {state.pixels_per_mm:.4f} px/mm using {state.known_mm:.3f} mm")
                save_config()
        else:
            print("Draw a measurement first")

    elif key == ord('l'):
        if point1 is not None and point2 is not None:
            label = f"M{len(locked_measurements)+1}"
            locked_measurements.append((point1, point2, label))
            print(f"Locked measurement {label}")

    elif key == ord('u'):
        if locked_measurements:
            locked_measurements.pop()
            print("Removed last locked measurement")

    elif key == ord('L'):
        if point1 is not None and point2 is not None:
            pixels, mm, mils = measurement_values(point1, point2)
            append_measurement_csv("active", pixels, mm, mils)

    elif key == ord('['):
        state.known_mm = max(0.01, state.known_mm - 0.01)
        print(f"Known distance: {state.known_mm:.3f} mm")
    elif key == ord(']'):
        state.known_mm += 0.01
        print(f"Known distance: {state.known_mm:.3f} mm")
    elif key == ord('{'):
        state.known_mm = max(0.01, state.known_mm - 0.10)
        print(f"Known distance: {state.known_mm:.3f} mm")
    elif key == ord('}'):
        state.known_mm += 0.10
        print(f"Known distance: {state.known_mm:.3f} mm")

    elif key == ord('B'):
        state.brightness_beta -= 5
        print(f"Software brightness: {state.brightness_beta}")
    elif key == ord('V'):
        state.brightness_beta += 5
        print(f"Software brightness: {state.brightness_beta}")
    elif key == ord('A'):
        state.contrast_alpha = max(0.1, state.contrast_alpha - 0.05)
        print(f"Software contrast: {state.contrast_alpha:.2f}")
    elif key == ord('D'):
        state.contrast_alpha += 0.05
        print(f"Software contrast: {state.contrast_alpha:.2f}")
    elif key == ord('G'):
        state.gamma = max(0.1, state.gamma - 0.05)
        print(f"Gamma: {state.gamma:.2f}")
    elif key == ord('T'):
        state.gamma += 0.05
        print(f"Gamma: {state.gamma:.2f}")

    elif key == ord('1'):
        set_resolution(640, 480, 60)
    elif key == ord('2'):
        set_resolution(800, 600, 60)
    elif key == ord('3'):
        set_resolution(1280, 720, 60)
    elif key == ord('4'):
        set_resolution(1920, 1080, 30)

    elif key == ord('z'):
        query_v4l2_controls()

    return True

# ============================================================
# MAIN
# ============================================================

def main():
    global last_frame, raw_frame_cache, processed_clean_cache, display_frame_cache
    global prev_time, fps_smoothed

    ensure_dirs()
    load_config()
    open_camera()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    print("""
==================== MICROCAM ULTRA ====================
q quit | h help | p snapshot | r record | f fullscreen
+/- or wheel zoom | WASD pan | t measure | k calibrate
Press h in the viewer for the full control list.
=========================================================
""")

    while True:
        if not state.paused:
            ok, raw = cap.read()
            if not ok:
                print("Frame read failed")
                break
            last_frame = raw.copy()
        else:
            if last_frame is None:
                continue
            raw = last_frame.copy()

        raw_frame_cache = raw.copy()

        processed = process_frame(raw)
        processed_clean_cache = processed.copy()

        focus = focus_score(processed)

        now = time.time()
        instant_fps = 1.0 / max(0.0001, now - prev_time)
        prev_time = now
        fps_smoothed = instant_fps if fps_smoothed == 0 else (fps_smoothed * 0.9 + instant_fps * 0.1)

        display = processed.copy()

        draw_grid(display)
        draw_crosshair(display)
        draw_ruler(display)
        draw_magnifier(display)
        draw_measurements(display)
        draw_focus_meter(display, focus)
        draw_histogram(display)
        draw_hud(display, fps_smoothed, focus)
        draw_help(display)

        display_frame_cache = display.copy()

        if state.recording and writer is not None:
            rec_frame = processed_clean_cache if state.record_clean else display_frame_cache
            writer.write(rec_frame)

        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF
        if not handle_key(key):
            break

    save_config()

    if state.recording:
        stop_recording()

    if cap is not None:
        cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
'''
```

## Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install opencv-python numpy
sudo apt install v4l-utils ffmpeg
```

## Run

```bash
python3 microcam_ultra.py
```

## Core Controls

| Key         | Action                 |
| ----------- | ---------------------- |
| q           | Quit                   |
| h           | Help overlay           |
| H           | Toggle HUD             |
| p           | Snapshot               |
| v           | Toggle clean snapshot  |
| r           | Record start/stop      |
| R           | Toggle clean recording |
| f           | Fullscreen             |
| space       | Freeze frame           |
| +/-         | Zoom                   |
| Mouse wheel | Zoom                   |
| WASD        | Pan                    |
| 0           | Reset view             |

## Overlay / Inspection Controls

| Key | Action      |
| --- | ----------- |
| c   | Crosshair   |
| g   | Grid        |
| M   | Magnifier   |
| U   | Scale ruler |
| O   | Histogram   |

## Image Modes

| Key | Action                 |
| --- | ---------------------- |
| b   | Grayscale              |
| i   | Invert                 |
| e   | Edge detect            |
| y   | Threshold              |
| Y   | Adaptive threshold     |
| ,/. | Threshold adjust       |
| n   | Denoise                |
| C   | CLAHE contrast enhance |
| S   | Sharpen                |
| F   | False color            |

## Measurement / Calibration

| Key        | Action                             |
| ---------- | ---------------------------------- |
| t          | Measure mode                       |
| Drag mouse | Draw measurement                   |
| x          | Clear active measurement           |
| k          | Calibrate using active measurement |
| l          | Lock active measurement            |
| u          | Undo locked measurement            |
| L          | Log measurement to CSV             |
| [/]        | Adjust known distance by 0.01 mm   |
| {/}        | Adjust known distance by 0.10 mm   |

## Resolution Presets

| Key | Mode               |
| --- | ------------------ |
| 1   | 640x480 @ 60 FPS   |
| 2   | 800x600 @ 60 FPS   |
| 3   | 1280x720 @ 60 FPS  |
| 4   | 1920x1080 @ 30 FPS |

## Software Image Tuning

| Key | Action                       |
| --- | ---------------------------- |
| B/V | Brightness down/up           |
| A/D | Contrast down/up             |
| G/T | Gamma down/up                |
| z   | Print V4L2 hardware controls |

## Notes

* Calibration persists in `microcam_ultra_config.json`.
* Snapshots and recordings go into `captures/`.
* Measurement logs go into `captures/measurements.csv`.
* Clean snapshots/recordings save the processed image without overlays.
* Normal snapshots/recordings include overlays.

## Best Use Workflow

1. Tune hardware brightness/contrast with `v4l2-ctl` first.
2. Start MicroCam Ultra.
3. Calibrate with a 2.54 mm DIP/header pitch.
4. Use clean snapshots for documentation.
5. Use overlay snapshots for repair notes.
6. Use edge/threshold modes only when they help; do not leave them on by default.

## Future Pro-Level Upgrades

* Qt/PySide GUI with sliders
* Hardware V4L2 control panel inside app
* MP4/H.264 recording backend using ffmpeg
* OCR chip marking reader
* Focus stacking
* HDR capture
* Auto white-balance card calibration
* Solder bridge detection model
* Trace width auto-measurement
* Component/pad segmentation
* Multi-camera support
* Macro stitching / panorama mode
* Automatic report generation
'''