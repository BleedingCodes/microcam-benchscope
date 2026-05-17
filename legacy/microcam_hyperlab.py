"""
# MicroCam HyperLab Pro

A full PySide6/OpenCV electronics inspection workstation for Linux USB microscopes/capture cards.

This version includes working implementations or practical first-pass versions of:

* Qt/PySide GUI
* Live camera viewer
* Multi-camera selector
* V4L2 hardware controls
* Software sliders
* MP4/H.264 ffmpeg recording
* Snapshots
* OCR chip-marking reader
* Focus stacking
* HDR capture
* Auto white-balance calibration
* Solder bridge detection heuristic
* Trace width auto-measurement heuristic
* Pad/component segmentation heuristic
* Panorama/macro stitching
* Measurement calibration
* Automatic HTML report generation

Some advanced computer-vision tools are heuristic, not trained AI models. They are useful starting points and can later be replaced with trained models.

---

## Install

```bash
sudo apt install v4l-utils ffmpeg tesseract-ocr python3-tk
python3 -m venv venv
source venv/bin/activate
pip install opencv-python numpy PySide6 pytesseract pillow reportlab
```

Optional stitching support is included in most `opencv-python` builds. If stitching fails, try:

```bash
pip install opencv-contrib-python
```

---

## Run

```bash
python3 microcam_hyperlab.py
```

---

## Full Single-File Application

```python
#!/usr/bin/env python3

MicroCam HyperLab Pro
Linux USB microscope / capture-card inspection workstation.

Designed for electronics repair, PCB inspection, measurement, recording,
OCR, focus stacking, HDR capture, segmentation, solder bridge detection,
trace width estimation, panorama stitching, and report generation.

This is intentionally single-file for easy hacking.
"""

import os
import sys
import cv2
import csv
import json
import time
import math
import glob
import shutil
import signal
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False

from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtGui import QImage, QPixmap, QAction
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QCheckBox,
    QFileDialog,
    QTextEdit,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QGroupBox,
    QTabWidget,
    QMessageBox,
    QLineEdit,
    QSplitter,
)

# ============================================================
# PATHS
# ============================================================

APP_NAME = "MicroCam HyperLab Pro"
CAPTURE_DIR = Path("captures")
STACK_DIR = CAPTURE_DIR / "focus_stack"
HDR_DIR = CAPTURE_DIR / "hdr"
PANORAMA_DIR = CAPTURE_DIR / "panorama"
REPORT_DIR = CAPTURE_DIR / "reports"
CONFIG_FILE = Path("microcam_hyperlab_config.json")
MEASUREMENTS_CSV = CAPTURE_DIR / "measurements.csv"

for d in [CAPTURE_DIR, STACK_DIR, HDR_DIR, PANORAMA_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONFIG / STATE
# ============================================================

@dataclass
class Config:
    device: str = "/dev/video0"
    width: int = 1280
    height: int = 720
    fps: int = 30
    fourcc: str = "MJPG"

    pixels_per_mm: float = 15.08
    known_mm: float = 2.54

    brightness: int = 0
    contrast: float = 1.0
    gamma: float = 1.0
    threshold_value: int = 120

    zoom: float = 1.0
    pan_x: int = 0
    pan_y: int = 0

    show_crosshair: bool = True
    show_grid: bool = False
    show_ruler: bool = True
    show_histogram: bool = False
    show_focus_meter: bool = True

    grayscale: bool = False
    invert: bool = False
    edge: bool = False
    threshold: bool = False
    adaptive_threshold: bool = False
    denoise: bool = False
    sharpen: bool = False
    clahe: bool = False
    false_color: bool = False

    record_clean: bool = False
    snapshot_clean: bool = False

config = Config()

# Runtime state
cap = None
current_raw = None
current_processed = None
current_display = None
last_frame = None
is_paused = False
is_recording = False
record_proc = None
record_file = None
record_start_time = None
measurement_start = None
measurement_end = None
locked_measurements = []
mouse_down = False
panning = False
last_pan_pos = (0, 0)
focus_stack_frames: List[np.ndarray] = []
hdr_frames: List[np.ndarray] = []
panorama_frames: List[np.ndarray] = []
last_analysis: Dict[str, str] = {}

# ============================================================
# UTILS
# ============================================================

def now_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def log(msg: str):
    print(msg)
    if MainWindow.instance is not None:
        MainWindow.instance.append_log(msg)


def save_config():
    with open(CONFIG_FILE, "w") as f:
        json.dump(asdict(config), f, indent=4)
    log(f"Saved config: {CONFIG_FILE}")


def load_config():
    global config
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text())
            for k, v in data.items():
                if hasattr(config, k):
                    setattr(config, k, v)
            log(f"Loaded config: {CONFIG_FILE}")
        except Exception as e:
            log(f"Config load failed: {e}")


def run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return p.returncode, p.stdout, p.stderr
    except Exception as e:
        return 1, "", str(e)


def list_video_devices() -> List[str]:
    devices = sorted(glob.glob("/dev/video*"))
    return devices


def frame_to_qimage(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()


def apply_gamma(frame, gamma):
    if abs(gamma - 1.0) < 0.001:
        return frame
    inv = 1.0 / max(0.05, gamma)
    table = np.array([(i / 255.0) ** inv * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(frame, table)


def focus_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def measurement_values(p1, p2):
    px = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    mm = px / config.pixels_per_mm if config.pixels_per_mm else 0
    mils = mm * 39.3701
    return px, mm, mils

# ============================================================
# CAMERA
# ============================================================

def open_camera():
    global cap
    if cap is not None:
        cap.release()
    cap = cv2.VideoCapture(config.device)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config.fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    cap.set(cv2.CAP_PROP_FPS, config.fps)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {config.device}")
    log(f"Opened {config.device}: {config.width}x{config.height}@{config.fps} {config.fourcc}")


def set_v4l2_control(name, value):
    code, out, err = run_cmd(["v4l2-ctl", f"--device={config.device}", f"--set-ctrl={name}={value}"])
    if code != 0:
        log(f"V4L2 control failed: {err.strip()}")


def get_v4l2_controls_text():
    code, out, err = run_cmd(["v4l2-ctl", f"--device={config.device}", "--list-ctrls"])
    return out if out else err

# ============================================================
# PROCESSING PIPELINE
# ============================================================

def process_frame(raw):
    frame = raw.copy()
    frame = cv2.convertScaleAbs(frame, alpha=config.contrast, beta=config.brightness)
    frame = apply_gamma(frame, config.gamma)

    if config.denoise:
        frame = cv2.fastNlMeansDenoisingColored(frame, None, 4, 4, 7, 15)

    if config.clahe:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        frame = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    if config.grayscale:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if config.invert:
        frame = cv2.bitwise_not(frame)

    if config.sharpen:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        frame = cv2.filter2D(frame, -1, kernel)

    if config.edge:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        edge = cv2.Canny(blur, 40, 100)
        frame = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

    if config.threshold:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, config.threshold_value, 255, cv2.THRESH_BINARY)
        frame = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

    if config.adaptive_threshold:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
        frame = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

    if config.false_color:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)

    frame = apply_zoom_pan(frame)
    return frame


def apply_zoom_pan(frame):
    if config.zoom <= 1.0:
        return frame
    h, w = frame.shape[:2]
    new_w = max(1, int(w / config.zoom))
    new_h = max(1, int(h / config.zoom))
    max_x = w - new_w
    max_y = h - new_h
    x1 = max(0, min((w - new_w) // 2 + config.pan_x, max_x))
    y1 = max(0, min((h - new_h) // 2 + config.pan_y, max_y))
    crop = frame[y1:y1 + new_h, x1:x1 + new_w]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

# ============================================================
# DRAW OVERLAYS
# ============================================================

def draw_text(frame, text, x, y, scale=0.55, color=(255,255,255), thick=1):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thick+2)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def draw_overlays(frame, fps):
    h, w = frame.shape[:2]
    fscore = focus_score(frame)
    draw_text(frame, f"{config.device} {config.width}x{config.height}@{config.fps} FPS:{fps:.1f} Zoom:{config.zoom:.2f} Focus:{fscore:.0f}", 20, 35, 0.6, (100,255,10), 2)
    draw_text(frame, f"Cal {config.pixels_per_mm:.2f}px/mm Known {config.known_mm:.3f}mm", 20, 65, 0.55, (255,255,255), 1)

    if is_recording:
        cv2.circle(frame, (35, 100), 10, (0,0,255), -1)
        draw_text(frame, "REC", 55, 108, 0.8, (0,0,255), 2)

    if config.show_grid:
        for x in range(0, w, 100):
            cv2.line(frame, (x,0), (x,h), (70,70,70), 1)
        for y in range(0, h, 100):
            cv2.line(frame, (0,y), (w,y), (70,70,70), 1)

    if config.show_crosshair:
        cx, cy = w//2, h//2
        cv2.line(frame, (cx-45, cy), (cx+45, cy), (0,255,255), 1)
        cv2.line(frame, (cx, cy-45), (cx, cy+45), (0,255,255), 1)
        cv2.circle(frame, (cx,cy), 3, (0,0,255), -1)

    if config.show_ruler:
        px_len = int(config.pixels_per_mm * 5)
        px_len = max(20, min(px_len, w - 100))
        x0, y0 = 40, h - 45
        cv2.line(frame, (x0,y0), (x0+px_len,y0), (255,255,255), 2)
        cv2.line(frame, (x0,y0-8), (x0,y0+8), (255,255,255), 2)
        cv2.line(frame, (x0+px_len,y0-8), (x0+px_len,y0+8), (255,255,255), 2)
        draw_text(frame, "5 mm", x0, y0-12, 0.5)

    if config.show_focus_meter:
        bar_w = 250
        val = int(min(1.0, fscore / 2000.0) * bar_w)
        x0, y0 = 40, h - 80
        cv2.rectangle(frame, (x0,y0), (x0+bar_w,y0+15), (80,80,80), 1)
        cv2.rectangle(frame, (x0,y0), (x0+val,y0+15), (0,220,255), -1)

    if config.show_histogram:
        draw_histogram(frame)

    draw_measurements(frame)


def draw_histogram(frame):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0,256])
    cv2.normalize(hist, hist, 0, 100, cv2.NORM_MINMAX)
    x0, y0 = w - 280, h - 130
    cv2.rectangle(frame, (x0,y0), (x0+260,y0+110), (0,0,0), -1)
    for i in range(1, 256):
        cv2.line(frame, (x0+i-1,y0+100-int(hist[i-1][0])), (x0+i,y0+100-int(hist[i][0])), (220,220,220), 1)


def draw_measurements(frame):
    for p1, p2, label in locked_measurements:
        draw_measurement(frame, p1, p2, label, (255,0,255))
    if measurement_start is not None and measurement_end is not None:
        draw_measurement(frame, measurement_start, measurement_end, "active", (255,0,0))


def draw_measurement(frame, p1, p2, label, color):
    cv2.circle(frame, p1, 5, (0,255,0), -1)
    cv2.circle(frame, p2, 5, (0,255,0), -1)
    cv2.line(frame, p1, p2, color, 2)
    px, mm, mils = measurement_values(p1, p2)
    draw_text(frame, f"{label}: {px:.1f}px {mm:.3f}mm {mils:.1f}mil", p1[0], max(25, p1[1]-10), 0.55, (255,255,0), 2)

# ============================================================
# RECORDING / SNAPSHOTS
# ============================================================

def start_recording():
    global is_recording, record_proc, record_file, record_start_time
    if current_processed is None:
        return
    h, w = current_processed.shape[:2]
    record_file = CAPTURE_DIR / f"recording_{now_stamp()}.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}",
        "-r", str(config.fps),
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(record_file),
    ]
    try:
        record_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        is_recording = True
        record_start_time = time.time()
        log(f"MP4 recording started: {record_file}")
    except Exception as e:
        log(f"Recording failed: {e}")
        record_proc = None
        is_recording = False


def stop_recording():
    global is_recording, record_proc
    if record_proc is not None:
        try:
            record_proc.stdin.close()
            record_proc.wait(timeout=5)
        except Exception:
            try:
                record_proc.kill()
            except Exception:
                pass
    record_proc = None
    is_recording = False
    log("Recording stopped")


def write_record_frame(frame):
    if record_proc is None or record_proc.stdin is None:
        return
    try:
        record_proc.stdin.write(frame.tobytes())
    except Exception as e:
        log(f"Recording write failed: {e}")
        stop_recording()


def save_snapshot():
    if current_processed is None:
        return
    img = current_processed if config.snapshot_clean else current_display
    file = CAPTURE_DIR / f"snapshot_{now_stamp()}.png"
    cv2.imwrite(str(file), img)
    log(f"Snapshot: {file}")

# ============================================================
# ANALYSIS TOOLS
# ============================================================

def run_ocr():
    if current_processed is None:
        return
    if not TESSERACT_AVAILABLE:
        log("pytesseract not installed or unavailable")
        return
    gray = cv2.cvtColor(current_processed, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(th, config="--psm 6")
    last_analysis["OCR"] = text.strip()
    log("OCR result:\n" + text.strip())


def detect_solder_bridges():
    """Heuristic: detect suspicious bright blobs connecting nearby pin-like regions."""
    if current_processed is None:
        return
    img = current_processed.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for c in contours:
        area = cv2.contourArea(c)
        if 30 < area < 5000:
            x,y,w,h = cv2.boundingRect(c)
            aspect = max(w,h) / max(1, min(w,h))
            if aspect > 2.2 and min(w,h) < 35:
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
                count += 1
    file = CAPTURE_DIR / f"solder_bridge_candidates_{now_stamp()}.png"
    cv2.imwrite(str(file), img)
    last_analysis["Solder bridge candidates"] = f"{count} candidates saved to {file}"
    log(f"Solder bridge candidates: {count}; saved {file}")


def segment_components_pads():
    """Heuristic segmentation using edges + morphology."""
    if current_processed is None:
        return
    img = current_processed.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 130)
    kernel = np.ones((3,3), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for c in contours:
        area = cv2.contourArea(c)
        if 80 < area < 50000:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 1)
            count += 1
    file = CAPTURE_DIR / f"segmentation_{now_stamp()}.png"
    cv2.imwrite(str(file), img)
    last_analysis["Segmentation"] = f"{count} regions saved to {file}"
    log(f"Segmented regions: {count}; saved {file}")


def trace_width_measure():
    """First-pass trace-width estimator from active measurement line.
    User draws across a trace; the tool samples brightness profile along the line.
    """
    if current_processed is None or measurement_start is None or measurement_end is None:
        log("Draw a measurement line across a trace first")
        return
    gray = cv2.cvtColor(current_processed, cv2.COLOR_BGR2GRAY)
    p1, p2 = measurement_start, measurement_end
    n = int(math.hypot(p2[0]-p1[0], p2[1]-p1[1]))
    if n < 5:
        return
    xs = np.linspace(p1[0], p2[0], n).astype(int)
    ys = np.linspace(p1[1], p2[1], n).astype(int)
    xs = np.clip(xs, 0, gray.shape[1]-1)
    ys = np.clip(ys, 0, gray.shape[0]-1)
    vals = gray[ys, xs]
    thresh = (vals.max() + vals.min()) / 2
    mask = vals > thresh
    runs = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i-1))
            start = None
    if start is not None:
        runs.append((start, len(mask)-1))
    if not runs:
        log("No bright trace-like run detected")
        return
    longest = max(runs, key=lambda r: r[1]-r[0])
    px = longest[1] - longest[0] + 1
    mm = px / config.pixels_per_mm
    mils = mm * 39.3701
    last_analysis["Trace width"] = f"Estimated {px}px = {mm:.4f} mm = {mils:.2f} mil"
    log(last_analysis["Trace width"])


def add_focus_stack_frame():
    if current_processed is not None:
        focus_stack_frames.append(current_processed.copy())
        file = STACK_DIR / f"stack_{len(focus_stack_frames):03d}_{now_stamp()}.png"
        cv2.imwrite(str(file), current_processed)
        log(f"Focus stack frame added: {len(focus_stack_frames)}")


def build_focus_stack():
    if len(focus_stack_frames) < 2:
        log("Need at least 2 focus stack frames")
        return
    # Laplacian-per-pixel best-focus fusion
    frames = focus_stack_frames
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    laps = [cv2.Laplacian(g, cv2.CV_64F) for g in grays]
    abs_laps = [np.abs(l) for l in laps]
    stack = np.stack(abs_laps, axis=0)
    best = np.argmax(stack, axis=0)
    result = np.zeros_like(frames[0])
    for i, f in enumerate(frames):
        mask = best == i
        result[mask] = f[mask]
    file = STACK_DIR / f"focus_stacked_{now_stamp()}.png"
    cv2.imwrite(str(file), result)
    last_analysis["Focus stack"] = f"Saved {file} from {len(frames)} frames"
    log(last_analysis["Focus stack"])


def clear_focus_stack():
    focus_stack_frames.clear()
    log("Focus stack cleared")


def add_hdr_frame():
    if current_raw is not None:
        hdr_frames.append(current_raw.copy())
        log(f"HDR frame added: {len(hdr_frames)}")


def build_hdr():
    if len(hdr_frames) < 2:
        log("Need at least 2 HDR frames")
        return
    # Exposure fusion fallback: median/weighted average style
    arr = np.stack([f.astype(np.float32) for f in hdr_frames], axis=0)
    result = np.mean(arr, axis=0)
    result = np.clip(result, 0, 255).astype(np.uint8)
    # Local contrast after merge
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
    result = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2BGR)
    file = HDR_DIR / f"hdr_fusion_{now_stamp()}.png"
    cv2.imwrite(str(file), result)
    last_analysis["HDR"] = f"Saved {file} from {len(hdr_frames)} frames"
    log(last_analysis["HDR"])


def clear_hdr():
    hdr_frames.clear()
    log("HDR frames cleared")


def auto_white_balance():
    """Gray-world white balance calibration on current frame."""
    if current_raw is None:
        return
    img = current_raw.astype(np.float32)
    avg_b, avg_g, avg_r = img.reshape(-1,3).mean(axis=0)
    gray = (avg_b + avg_g + avg_r) / 3.0
    gains = np.array([gray/avg_b, gray/avg_g, gray/avg_r])
    corrected = np.clip(img * gains, 0, 255).astype(np.uint8)
    file = CAPTURE_DIR / f"white_balance_preview_{now_stamp()}.png"
    cv2.imwrite(str(file), corrected)
    last_analysis["White balance"] = f"Preview saved {file}; gains BGR={gains}"
    log(last_analysis["White balance"])


def add_panorama_frame():
    if current_processed is not None:
        panorama_frames.append(current_processed.copy())
        log(f"Panorama frame added: {len(panorama_frames)}")


def build_panorama():
    if len(panorama_frames) < 2:
        log("Need at least 2 panorama frames")
        return
    try:
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        status, pano = stitcher.stitch(panorama_frames)
        if status == cv2.Stitcher_OK:
            file = PANORAMA_DIR / f"panorama_{now_stamp()}.png"
            cv2.imwrite(str(file), pano)
            last_analysis["Panorama"] = f"Saved {file}"
            log(last_analysis["Panorama"])
        else:
            log(f"Panorama stitching failed with status {status}")
    except Exception as e:
        log(f"Panorama stitching error: {e}")


def clear_panorama():
    panorama_frames.clear()
    log("Panorama frames cleared")


def generate_report():
    file = REPORT_DIR / f"inspection_report_{now_stamp()}.html"
    snaps = sorted(CAPTURE_DIR.glob("snapshot_*.png"))[-8:]
    recs = sorted(CAPTURE_DIR.glob("recording_*.mp4"))[-5:]
    html = [
        "<html><head><title>MicroCam Inspection Report</title>",
        "<style>body{font-family:Arial;margin:24px;} img{max-width:420px;border:1px solid #ccc;margin:8px;} code{background:#eee;padding:2px 4px;}</style>",
        "</head><body>",
        f"<h1>Inspection Report</h1><p>Generated: {datetime.now()}</p>",
        "<h2>Configuration</h2><pre>" + json.dumps(asdict(config), indent=2) + "</pre>",
        "<h2>Latest Analysis</h2><ul>",
    ]
    for k,v in last_analysis.items():
        html.append(f"<li><b>{k}</b>: {v}</li>")
    html.append("</ul>")
    html.append("<h2>Recent Snapshots</h2>")
    for s in snaps:
        rel = os.path.relpath(s, REPORT_DIR)
        html.append(f"<div><img src='{rel}'><p>{s.name}</p></div>")
    html.append("<h2>Recent Recordings</h2><ul>")
    for r in recs:
        html.append(f"<li>{r}</li>")
    html.append("</ul></body></html>")
    file.write_text("\n".join(html))
    log(f"Report generated: {file}")

# ============================================================
# GUI
# ============================================================

class MainWindow(QMainWindow):
    instance = None

    def __init__(self):
        super().__init__()
        MainWindow.instance = self
        self.setWindowTitle(APP_NAME)
        self.resize(1500, 950)

        self.video_label = VideoLabel()
        self.video_label.setMinimumSize(900, 650)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background:#111; color:#ddd;")

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(170)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.camera_tab(), "Camera")
        self.tabs.addTab(self.processing_tab(), "Processing")
        self.tabs.addTab(self.tools_tab(), "Tools")
        self.tabs.addTab(self.analysis_tab(), "Analysis")
        self.tabs.addTab(self.output_tab(), "Output")

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self.tabs)
        right_layout.addWidget(QLabel("Log"))
        right_layout.addWidget(self.log_box)

        splitter = QSplitter()
        splitter.addWidget(self.video_label)
        splitter.addWidget(right)
        splitter.setSizes([1050, 450])
        self.setCentralWidget(splitter)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)

        self.last_tick = time.time()
        self.fps = 0.0

        self.create_menu()
        self.refresh_devices()

    def create_menu(self):
        save_action = QAction("Save Config", self)
        save_action.triggered.connect(save_config)
        self.menuBar().addAction(save_action)

    def append_log(self, text):
        self.log_box.append(text)

    def camera_tab(self):
        w = QWidget(); layout = QGridLayout(w)
        self.device_combo = QComboBox()
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh_devices)
        open_btn = QPushButton("Open Camera")
        open_btn.clicked.connect(self.apply_camera_settings)

        self.width_spin = QSpinBox(); self.width_spin.setRange(160, 7680); self.width_spin.setValue(config.width)
        self.height_spin = QSpinBox(); self.height_spin.setRange(120, 4320); self.height_spin.setValue(config.height)
        self.fps_spin = QSpinBox(); self.fps_spin.setRange(1, 240); self.fps_spin.setValue(config.fps)
        self.fourcc_box = QComboBox(); self.fourcc_box.addItems(["MJPG", "YUYV"]); self.fourcc_box.setCurrentText(config.fourcc)

        layout.addWidget(QLabel("Device"),0,0); layout.addWidget(self.device_combo,0,1); layout.addWidget(refresh,0,2)
        layout.addWidget(QLabel("Width"),1,0); layout.addWidget(self.width_spin,1,1)
        layout.addWidget(QLabel("Height"),2,0); layout.addWidget(self.height_spin,2,1)
        layout.addWidget(QLabel("FPS"),3,0); layout.addWidget(self.fps_spin,3,1)
        layout.addWidget(QLabel("FOURCC"),4,0); layout.addWidget(self.fourcc_box,4,1)
        layout.addWidget(open_btn,5,0,1,3)

        controls_btn = QPushButton("Read V4L2 Controls")
        controls_btn.clicked.connect(lambda: log(get_v4l2_controls_text()))
        layout.addWidget(controls_btn,6,0,1,3)

        self.hw_name = QLineEdit("brightness")
        self.hw_val = QSpinBox(); self.hw_val.setRange(-9999,9999); self.hw_val.setValue(0)
        hw_btn = QPushButton("Set V4L2 Control")
        hw_btn.clicked.connect(lambda: set_v4l2_control(self.hw_name.text(), self.hw_val.value()))
        layout.addWidget(QLabel("Ctrl name"),7,0); layout.addWidget(self.hw_name,7,1)
        layout.addWidget(QLabel("Value"),8,0); layout.addWidget(self.hw_val,8,1)
        layout.addWidget(hw_btn,9,0,1,3)
        return w

    def processing_tab(self):
        w = QWidget(); layout = QGridLayout(w)
        self.brightness_slider = self.slider(-100,100,config.brightness, lambda v: setattr(config,'brightness',v))
        self.contrast_spin = QDoubleSpinBox(); self.contrast_spin.setRange(0.1,5.0); self.contrast_spin.setSingleStep(0.05); self.contrast_spin.setValue(config.contrast); self.contrast_spin.valueChanged.connect(lambda v: setattr(config,'contrast',v))
        self.gamma_spin = QDoubleSpinBox(); self.gamma_spin.setRange(0.1,5.0); self.gamma_spin.setSingleStep(0.05); self.gamma_spin.setValue(config.gamma); self.gamma_spin.valueChanged.connect(lambda v: setattr(config,'gamma',v))
        self.threshold_slider = self.slider(0,255,config.threshold_value, lambda v: setattr(config,'threshold_value',v))

        layout.addWidget(QLabel("Brightness"),0,0); layout.addWidget(self.brightness_slider,0,1)
        layout.addWidget(QLabel("Contrast"),1,0); layout.addWidget(self.contrast_spin,1,1)
        layout.addWidget(QLabel("Gamma"),2,0); layout.addWidget(self.gamma_spin,2,1)
        layout.addWidget(QLabel("Threshold"),3,0); layout.addWidget(self.threshold_slider,3,1)

        toggles = [
            ("Grayscale", 'grayscale'), ("Invert", 'invert'), ("Edge", 'edge'), ("Threshold", 'threshold'),
            ("Adaptive Threshold", 'adaptive_threshold'), ("Denoise", 'denoise'), ("Sharpen", 'sharpen'),
            ("CLAHE", 'clahe'), ("False Color", 'false_color')
        ]
        row = 4
        for text, attr in toggles:
            cb = QCheckBox(text); cb.setChecked(getattr(config, attr)); cb.toggled.connect(lambda checked, a=attr: setattr(config,a,checked))
            layout.addWidget(cb,row,0,1,2); row += 1
        return w

    def tools_tab(self):
        w = QWidget(); layout = QGridLayout(w)
        items = [("Crosshair",'show_crosshair'),("Grid",'show_grid'),("Ruler",'show_ruler'),("Histogram",'show_histogram'),("Focus Meter",'show_focus_meter')]
        row=0
        for text, attr in items:
            cb=QCheckBox(text); cb.setChecked(getattr(config,attr)); cb.toggled.connect(lambda checked,a=attr:setattr(config,a,checked))
            layout.addWidget(cb,row,0); row+=1
        self.zoom_spin=QDoubleSpinBox(); self.zoom_spin.setRange(1.0,20.0); self.zoom_spin.setSingleStep(0.1); self.zoom_spin.setValue(config.zoom); self.zoom_spin.valueChanged.connect(lambda v:setattr(config,'zoom',v))
        layout.addWidget(QLabel("Zoom"),row,0); layout.addWidget(self.zoom_spin,row,1); row+=1
        reset=QPushButton("Reset View"); reset.clicked.connect(self.reset_view); layout.addWidget(reset,row,0,1,2); row+=1
        self.known_spin=QDoubleSpinBox(); self.known_spin.setRange(0.01,1000); self.known_spin.setDecimals(4); self.known_spin.setValue(config.known_mm); self.known_spin.valueChanged.connect(lambda v:setattr(config,'known_mm',v))
        layout.addWidget(QLabel("Known mm"),row,0); layout.addWidget(self.known_spin,row,1); row+=1
        cal=QPushButton("Calibrate from Measurement"); cal.clicked.connect(self.calibrate_from_measurement); layout.addWidget(cal,row,0,1,2); row+=1
        return w

    def analysis_tab(self):
        w=QWidget(); layout=QGridLayout(w)
        buttons = [
            ("OCR Chip Marking", run_ocr),
            ("Detect Solder Bridges", detect_solder_bridges),
            ("Segment Pads/Components", segment_components_pads),
            ("Estimate Trace Width", trace_width_measure),
            ("Auto White Balance Preview", auto_white_balance),
            ("Add Focus Stack Frame", add_focus_stack_frame),
            ("Build Focus Stack", build_focus_stack),
            ("Clear Focus Stack", clear_focus_stack),
            ("Add HDR Frame", add_hdr_frame),
            ("Build HDR", build_hdr),
            ("Clear HDR", clear_hdr),
            ("Add Panorama Frame", add_panorama_frame),
            ("Build Panorama", build_panorama),
            ("Clear Panorama", clear_panorama),
            ("Generate HTML Report", generate_report),
        ]
        for i,(text,fn) in enumerate(buttons):
            btn=QPushButton(text); btn.clicked.connect(fn); layout.addWidget(btn,i,0)
        return w

    def output_tab(self):
        w=QWidget(); layout=QGridLayout(w)
        snap=QPushButton("Snapshot"); snap.clicked.connect(save_snapshot)
        rec=QPushButton("Start/Stop MP4 Recording"); rec.clicked.connect(self.toggle_record)
        self.clean_snap=QCheckBox("Clean snapshots"); self.clean_snap.setChecked(config.snapshot_clean); self.clean_snap.toggled.connect(lambda v:setattr(config,'snapshot_clean',v))
        self.clean_rec=QCheckBox("Clean recording"); self.clean_rec.setChecked(config.record_clean); self.clean_rec.toggled.connect(lambda v:setattr(config,'record_clean',v))
        layout.addWidget(snap,0,0); layout.addWidget(rec,1,0); layout.addWidget(self.clean_snap,2,0); layout.addWidget(self.clean_rec,3,0)
        return w

    def slider(self, mn, mx, val, callback):
        s=QSlider(Qt.Horizontal); s.setRange(mn,mx); s.setValue(val); s.valueChanged.connect(callback); return s

    def refresh_devices(self):
        self.device_combo.clear(); self.device_combo.addItems(list_video_devices()); self.device_combo.setCurrentText(config.device)

    def apply_camera_settings(self):
        config.device=self.device_combo.currentText(); config.width=self.width_spin.value(); config.height=self.height_spin.value(); config.fps=self.fps_spin.value(); config.fourcc=self.fourcc_box.currentText(); open_camera()

    def reset_view(self):
        config.zoom=1.0; config.pan_x=0; config.pan_y=0; self.zoom_spin.setValue(1.0)

    def toggle_record(self):
        if is_recording: stop_recording()
        else: start_recording()

    def calibrate_from_measurement(self):
        global measurement_start, measurement_end
        if measurement_start is None or measurement_end is None:
            log("Draw measurement first")
            return
        px,_,_=measurement_values(measurement_start,measurement_end)
        if px>0:
            config.pixels_per_mm=px/config.known_mm
            log(f"Calibrated: {config.pixels_per_mm:.4f} px/mm")
            save_config()

    def update_frame(self):
        global current_raw,current_processed,current_display,last_frame
        global self_fps_time
        if cap is None: return
        if is_paused and last_frame is not None:
            raw=last_frame.copy()
        else:
            ok, raw=cap.read()
            if not ok: return
            last_frame=raw.copy()
        current_raw=raw.copy()
        processed=process_frame(raw)
        current_processed=processed.copy()
        display=processed.copy()
        now=time.time(); dt=max(0.0001,now-self.last_tick); self.last_tick=now; self.fps=0.9*self.fps+0.1*(1/dt) if self.fps else 1/dt
        draw_overlays(display,self.fps)
        current_display=display.copy()
        if is_recording:
            write_record_frame(current_processed if config.record_clean else current_display)
        qimg=frame_to_qimage(display)
        pix=QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pix.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def closeEvent(self,event):
        save_config()
        if is_recording: stop_recording()
        if cap is not None: cap.release()
        event.accept()

class VideoLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setFocusPolicy(Qt.StrongFocus)

    def mousePressEvent(self, event):
        global measurement_start, measurement_end, mouse_down
        global panning, last_pan_pos

        self.setFocus()

        if event.button() == Qt.LeftButton:
            measurement_start = self.map_to_frame(event.position().x(), event.position().y())
            measurement_end = measurement_start
            mouse_down = True

        elif event.button() == Qt.MiddleButton or event.button() == Qt.RightButton:
            panning = True
            last_pan_pos = (event.position().x(), event.position().y())

    def mouseMoveEvent(self, event):
        global measurement_end, mouse_down
        global panning, last_pan_pos

        if mouse_down:
            measurement_end = self.map_to_frame(event.position().x(), event.position().y())

        if panning:
            dx = event.position().x() - last_pan_pos[0]
            dy = event.position().y() - last_pan_pos[1]

            config.pan_x -= int(dx / max(1.0, config.zoom))
            config.pan_y -= int(dy / max(1.0, config.zoom))

            last_pan_pos = (event.position().x(), event.position().y())

    def mouseReleaseEvent(self, event):
        global measurement_end, mouse_down
        global panning

        if event.button() == Qt.LeftButton:
            measurement_end = self.map_to_frame(event.position().x(), event.position().y())
            mouse_down = False

        elif event.button() == Qt.MiddleButton or event.button() == Qt.RightButton:
            panning = False

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            config.zoom = min(config.zoom + 0.2, 20)
        else:
            config.zoom = max(config.zoom - 0.2, 1)

        if MainWindow.instance:
            MainWindow.instance.zoom_spin.setValue(config.zoom)

    def keyPressEvent(self, event):
        global is_paused

        key = event.key()

        if key == Qt.Key_W:
            config.pan_y -= 50
        elif key == Qt.Key_S:
            config.pan_y += 50
        elif key == Qt.Key_A:
            config.pan_x -= 50
        elif key == Qt.Key_D:
            config.pan_x += 50
        elif key == Qt.Key_Space:
            is_paused = not is_paused
        elif key == Qt.Key_P:
            save_snapshot()
        elif key == Qt.Key_R:
            if is_recording:
                stop_recording()
            else:
                start_recording()

    def map_to_frame(self, x, y):
        if current_display is None or self.pixmap() is None:
            return (int(x), int(y))

        frame_h, frame_w = current_display.shape[:2]
        label_w, label_h = self.width(), self.height()

        scale = min(label_w / frame_w, label_h / frame_h)

        disp_w = frame_w * scale
        disp_h = frame_h * scale

        off_x = (label_w - disp_w) / 2
        off_y = (label_h - disp_h) / 2

        fx = int((x - off_x) / scale)
        fy = int((y - off_y) / scale)

        return (
            max(0, min(frame_w - 1, fx)),
            max(0, min(frame_h - 1, fy))
        )

# ============================================================
# MAIN
# ============================================================

def main():
    load_config()
    open_camera()
    app=QApplication(sys.argv)
    win=MainWindow()
    win.show()
    rc=app.exec()
    if is_recording: stop_recording()
    if cap is not None: cap.release()
    sys.exit(rc)

if __name__=="__main__":
    main()
'''

---

## Important Reality Check

This includes working first-pass versions of the big features. The following are heuristic, not magic:

* solder bridge detection
* trace width detection
* component/pad segmentation
* OCR accuracy
* panorama stitching
* HDR quality
* focus stacking quality

They work best when lighting is good, focus is sharp, and the PCB region is clear. Real production-grade solder defect detection needs labeled training data, but this gives you a powerful starting workstation right now.
'''