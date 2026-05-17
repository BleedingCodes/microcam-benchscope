
#!/usr/bin/env python3
"""
MicroCam BenchScope Pro - PyAV Edition

Linux HDMI microscope / USB capture-card inspection workstation.

Designed for MacroSilicon-style HDMI USB capture cards, USB microscopes,
electronics repair, PCB inspection, snapshots, calibrated measurements,
focus stacking, OCR, solder-bridge candidate detection, trace-width estimates,
and H.264 MP4 recording through PyAV.

Install:

sudo apt install v4l-utils tesseract-ocr
python3 -m venv venv
source venv/bin/activate
pip install PySide6 opencv-python numpy av pytesseract

Optional:
pip install opencv-contrib-python

Run:
python3 microcam_benchscope_pyav.py
"""

import sys
import cv2
import json
import time
import math
import glob
import queue
import shutil
import threading
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple, Dict

import av
import numpy as np

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False

from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtGui import QImage, QPixmap, QAction, QKeySequence
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
    QTextEdit,
    QVBoxLayout,
    QGridLayout,
    QTabWidget,
    QLineEdit,
    QSplitter,
    QSizePolicy,
)


APP_NAME = "MicroCam BenchScope Pro - PyAV Edition"

CONFIG_FILE = Path("microcam_benchscope_pyav_config.json")
CAPTURE_DIR = Path("captures")
SNAPSHOT_DIR = CAPTURE_DIR / "snapshots"
VIDEO_DIR = CAPTURE_DIR / "videos"
ANALYSIS_DIR = CAPTURE_DIR / "analysis"
STACK_DIR = CAPTURE_DIR / "focus_stack"
HDR_DIR = CAPTURE_DIR / "hdr"
REPORT_DIR = CAPTURE_DIR / "reports"

for folder in [
    CAPTURE_DIR,
    SNAPSHOT_DIR,
    VIDEO_DIR,
    ANALYSIS_DIR,
    STACK_DIR,
    HDR_DIR,
    REPORT_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)


@dataclass
class AppConfig:
    device: str = "/dev/video0"
    width: int = 1920
    height: int = 1080
    fps: int = 30
    fourcc: str = "MJPG"

    brightness: int = 0
    contrast: float = 1.0
    gamma: float = 1.0
    saturation: float = 1.0
    threshold_value: int = 128

    zoom: float = 1.0
    pan_x: int = 0
    pan_y: int = 0

    pixels_per_mm: float = 100.0
    known_mm: float = 10.0

    show_crosshair: bool = True
    show_grid: bool = False
    show_ruler: bool = True
    show_focus_meter: bool = True
    show_histogram: bool = False
    show_measurements: bool = True
    show_status_text: bool = True

    grayscale: bool = False
    invert: bool = False
    sharpen: bool = False
    denoise: bool = False
    clahe: bool = False
    edges: bool = False
    threshold: bool = False
    adaptive_threshold: bool = False
    false_color: bool = False

    snapshot_clean: bool = False
    recording_clean: bool = False

    record_codec: str = "libx264"
    record_crf: int = 18
    record_preset: str = "veryfast"
    record_queue_size: int = 90


config = AppConfig()


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_config() -> None:
    global config

    if not CONFIG_FILE.exists():
        return

    try:
        data = json.loads(CONFIG_FILE.read_text())

        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)

    except Exception as exc:
        print(f"Config load failed: {exc}")


def save_config() -> None:
    try:
        CONFIG_FILE.write_text(json.dumps(asdict(config), indent=4))
    except Exception as exc:
        print(f"Config save failed: {exc}")


def run_command(cmd: List[str], timeout: float = 5.0) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr

    except Exception as exc:
        return 1, "", str(exc)


def list_video_devices() -> List[str]:
    devices = sorted(glob.glob("/dev/video*"))
    usable = []

    for dev in devices:
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)

        if cap.isOpened():
            usable.append(dev)

        cap.release()

    return usable or devices


def get_device_label(device: str) -> str:
    code, out, err = run_command(["v4l2-ctl", f"--device={device}", "--info"])
    text = out or err

    for line in text.splitlines():
        if "Card type" in line:
            return line.split(":", 1)[-1].strip()

    return device


def get_supported_formats(device: str) -> str:
    code, out, err = run_command(
        ["v4l2-ctl", f"--device={device}", "--list-formats-ext"],
        timeout=7.0,
    )
    return out if out else err


def get_v4l2_controls(device: str) -> str:
    code, out, err = run_command(
        ["v4l2-ctl", f"--device={device}", "--list-ctrls"],
        timeout=7.0,
    )
    return out if out else err


def set_v4l2_control(device: str, name: str, value: int) -> Tuple[bool, str]:
    code, out, err = run_command(
        ["v4l2-ctl", f"--device={device}", f"--set-ctrl={name}={value}"],
        timeout=5.0,
    )
    return code == 0, out if out else err


def frame_to_qimage(frame_bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()


def apply_gamma(frame: np.ndarray, gamma: float) -> np.ndarray:
    if abs(gamma - 1.0) < 0.001:
        return frame

    gamma = max(0.05, gamma)
    inv = 1.0 / gamma
    lut = np.array([(i / 255.0) ** inv * 255 for i in range(256)]).astype(np.uint8)

    return cv2.LUT(frame, lut)


def focus_score(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def measurement_values(
    p1: Tuple[int, int],
    p2: Tuple[int, int],
) -> Tuple[float, float, float]:
    px = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    mm = px / config.pixels_per_mm if config.pixels_per_mm > 0 else 0.0
    mil = mm * 39.3701
    return px, mm, mil


def draw_text(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    scale: float = 0.55,
    color: Tuple[int, int, int] = (255, 255, 255),
    thick: int = 1,
) -> None:
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thick + 3,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thick,
    )


class FrameProcessor:
    @staticmethod
    def process(raw: np.ndarray) -> np.ndarray:
        frame = raw.copy()

        frame = cv2.convertScaleAbs(
            frame,
            alpha=config.contrast,
            beta=config.brightness,
        )

        frame = apply_gamma(frame, config.gamma)

        if abs(config.saturation - 1.0) > 0.001:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] *= config.saturation
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

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
            kernel = np.array(
                [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
                dtype=np.float32,
            )
            frame = cv2.filter2D(frame, -1, kernel)

        if config.edges:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 45, 120)
            frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        if config.threshold:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(
                gray,
                config.threshold_value,
                255,
                cv2.THRESH_BINARY,
            )
            frame = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

        if config.adaptive_threshold:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            th = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                5,
            )
            frame = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

        if config.false_color:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)

        return FrameProcessor.apply_zoom_pan(frame)

    @staticmethod
    def apply_zoom_pan(frame: np.ndarray) -> np.ndarray:
        if config.zoom <= 1.0:
            return frame

        h, w = frame.shape[:2]

        crop_w = max(1, int(w / config.zoom))
        crop_h = max(1, int(h / config.zoom))

        max_x = w - crop_w
        max_y = h - crop_h

        x1 = max(0, min((w - crop_w) // 2 + config.pan_x, max_x))
        y1 = max(0, min((h - crop_h) // 2 + config.pan_y, max_y))

        crop = frame[y1:y1 + crop_h, x1:x1 + crop_w]

        return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


class CameraWorker(QObject):
    frame_ready = Signal(object)
    error = Signal(str)
    status = Signal(str)

    def __init__(self):
        super().__init__()
        self.cap: Optional[cv2.VideoCapture] = None
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.paused = False
        self.last_frame: Optional[np.ndarray] = None

    def start(self) -> None:
        self.stop()

        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

        self.thread = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def open_capture(self) -> bool:
        self.cap = cv2.VideoCapture(config.device, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            self.error.emit(f"Could not open {config.device}")
            return False

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config.fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
        self.cap.set(cv2.CAP_PROP_FPS, config.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = float(self.cap.get(cv2.CAP_PROP_FPS))

        self.status.emit(
            f"Opened {config.device}: requested "
            f"{config.width}x{config.height}@{config.fps} {config.fourcc}; "
            f"actual {actual_w}x{actual_h}@{actual_fps:.1f}"
        )

        return True

    def _loop(self) -> None:
        if not self.open_capture():
            return

        frame_interval = 1.0 / max(1, config.fps)

        while self.running:
            start = time.time()

            if self.paused and self.last_frame is not None:
                self.frame_ready.emit(self.last_frame.copy())

            else:
                if self.cap is None:
                    break

                ok, frame = self.cap.read()

                if not ok or frame is None:
                    self.error.emit("Camera read failed.")
                    time.sleep(0.1)
                    continue

                self.last_frame = frame.copy()
                self.frame_ready.emit(frame)

            elapsed = time.time() - start
            time.sleep(max(0.001, frame_interval - elapsed))


class PyAVRecorder:
    """
    Threaded PyAV H.264 recorder.

    Difference from ffmpeg subprocess:
    - No rawvideo pipe.
    - No shell command for recording.
    - Frames are encoded through PyAV directly.
    - Encoding happens on a worker thread so the GUI stays smoother.
    """

    def __init__(self):
        self.file: Optional[Path] = None
        self.width = 0
        self.height = 0
        self.fps = 30

        self.frame_queue: Optional[queue.Queue] = None
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.dropped_frames = 0
        self.error: Optional[str] = None

    @property
    def active(self) -> bool:
        return self.running

    def start(self, width: int, height: int, fps: int) -> Path:
        self.stop()

        self.file = VIDEO_DIR / f"recording_{now_stamp()}.mp4"
        self.width = int(width)
        self.height = int(height)
        self.fps = int(max(1, fps))
        self.dropped_frames = 0
        self.error = None

        self.frame_queue = queue.Queue(maxsize=max(5, config.record_queue_size))
        self.running = True

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

        return self.file

    def write(self, frame_bgr: np.ndarray) -> None:
        if not self.running or self.frame_queue is None:
            return

        try:
            self.frame_queue.put_nowait(frame_bgr.copy())
        except queue.Full:
            self.dropped_frames += 1

    def stop(self) -> Optional[Path]:
        if not self.running:
            return self.file

        self.running = False

        if self.frame_queue is not None:
            try:
                self.frame_queue.put_nowait(None)
            except queue.Full:
                pass

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=8.0)

        out = self.file

        self.thread = None
        self.frame_queue = None

        return out

    def _worker(self) -> None:
        container = None
        stream = None

        try:
            container = av.open(str(self.file), mode="w")

            try:
                stream = container.add_stream(config.record_codec, rate=self.fps)
            except Exception:
                stream = container.add_stream("mpeg4", rate=self.fps)

            stream.width = self.width
            stream.height = self.height
            stream.pix_fmt = "yuv420p"

            try:
                stream.options = {
                    "crf": str(config.record_crf),
                    "preset": str(config.record_preset),
                }
            except Exception:
                pass

            while True:
                if self.frame_queue is None:
                    break

                try:
                    item = self.frame_queue.get(timeout=0.25)
                except queue.Empty:
                    if not self.running:
                        break
                    continue

                if item is None:
                    break

                frame_bgr = item

                if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
                    frame_bgr = cv2.resize(
                        frame_bgr,
                        (self.width, self.height),
                        interpolation=cv2.INTER_AREA,
                    )

                video_frame = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")

                for packet in stream.encode(video_frame):
                    container.mux(packet)

            for packet in stream.encode():
                container.mux(packet)

        except Exception as exc:
            self.error = str(exc)

        finally:
            try:
                if container is not None:
                    container.close()
            except Exception:
                pass

            self.running = False


class VideoLabel(QLabel):
    request_snapshot = Signal()
    request_record_toggle = Signal()
    request_pause_toggle = Signal()

    def __init__(self):
        super().__init__()

        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(900, 650)
        self.setStyleSheet("background:#111; color:#ccc;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFocusPolicy(Qt.StrongFocus)

        self.frame_shape: Optional[Tuple[int, int]] = None

        self.measurement_start: Optional[Tuple[int, int]] = None
        self.measurement_end: Optional[Tuple[int, int]] = None
        self.locked_measurements: List[
            Tuple[Tuple[int, int], Tuple[int, int], str]
        ] = []

        self.mouse_down = False
        self.panning = False
        self.last_pan_pos = (0.0, 0.0)

    def set_frame_shape(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        self.frame_shape = (h, w)

    def map_to_frame(self, x: float, y: float) -> Tuple[int, int]:
        if self.frame_shape is None:
            return int(x), int(y)

        frame_h, frame_w = self.frame_shape
        label_w, label_h = self.width(), self.height()

        scale = min(label_w / frame_w, label_h / frame_h)

        display_w = frame_w * scale
        display_h = frame_h * scale

        off_x = (label_w - display_w) / 2
        off_y = (label_h - display_h) / 2

        fx = int((x - off_x) / scale)
        fy = int((y - off_y) / scale)

        return (
            max(0, min(frame_w - 1, fx)),
            max(0, min(frame_h - 1, fy)),
        )

    def mousePressEvent(self, event):
        self.setFocus()

        if event.button() == Qt.LeftButton:
            self.measurement_start = self.map_to_frame(
                event.position().x(),
                event.position().y(),
            )
            self.measurement_end = self.measurement_start
            self.mouse_down = True

        elif event.button() in (Qt.MiddleButton, Qt.RightButton):
            self.panning = True
            self.last_pan_pos = (
                event.position().x(),
                event.position().y(),
            )

    def mouseMoveEvent(self, event):
        if self.mouse_down:
            self.measurement_end = self.map_to_frame(
                event.position().x(),
                event.position().y(),
            )

        if self.panning:
            x = event.position().x()
            y = event.position().y()

            dx = x - self.last_pan_pos[0]
            dy = y - self.last_pan_pos[1]

            config.pan_x -= int(dx / max(1.0, config.zoom))
            config.pan_y -= int(dy / max(1.0, config.zoom))

            self.last_pan_pos = (x, y)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.measurement_end = self.map_to_frame(
                event.position().x(),
                event.position().y(),
            )
            self.mouse_down = False

        elif event.button() in (Qt.MiddleButton, Qt.RightButton):
            self.panning = False

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            config.zoom = min(30.0, config.zoom + 0.25)
        else:
            config.zoom = max(1.0, config.zoom - 0.25)

    def keyPressEvent(self, event):
        key = event.key()

        if key == Qt.Key_W:
            config.pan_y -= 50

        elif key == Qt.Key_S:
            config.pan_y += 50

        elif key == Qt.Key_A:
            config.pan_x -= 50

        elif key == Qt.Key_D:
            config.pan_x += 50

        elif key == Qt.Key_P:
            self.request_snapshot.emit()

        elif key == Qt.Key_R:
            self.request_record_toggle.emit()

        elif key == Qt.Key_Space:
            self.request_pause_toggle.emit()

        elif key == Qt.Key_L:
            self.lock_current_measurement()

        elif key == Qt.Key_C:
            self.clear_measurements()

    def lock_current_measurement(self):
        if self.measurement_start is None or self.measurement_end is None:
            return

        label = f"M{len(self.locked_measurements) + 1}"

        self.locked_measurements.append(
            (
                self.measurement_start,
                self.measurement_end,
                label,
            )
        )

    def clear_measurements(self):
        self.locked_measurements.clear()
        self.measurement_start = None
        self.measurement_end = None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(APP_NAME)
        self.resize(1550, 950)

        self.camera = CameraWorker()
        self.camera.frame_ready.connect(self.on_raw_frame)
        self.camera.error.connect(self.log)
        self.camera.status.connect(self.log)

        self.recorder = PyAVRecorder()

        self.raw_frame: Optional[np.ndarray] = None
        self.processed_frame: Optional[np.ndarray] = None
        self.display_frame: Optional[np.ndarray] = None

        self.focus_stack_frames: List[np.ndarray] = []
        self.hdr_frames: List[np.ndarray] = []
        self.last_analysis: Dict[str, str] = {}

        self.last_tick = time.time()
        self.ui_fps = 0.0

        self.video = VideoLabel()
        self.video.request_snapshot.connect(self.save_snapshot)
        self.video.request_record_toggle.connect(self.toggle_recording)
        self.video.request_pause_toggle.connect(self.toggle_pause)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(170)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.make_camera_tab(), "Camera")
        self.tabs.addTab(self.make_image_tab(), "Image")
        self.tabs.addTab(self.make_overlay_tab(), "Overlays")
        self.tabs.addTab(self.make_analysis_tab(), "Analysis")
        self.tabs.addTab(self.make_output_tab(), "Output")

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self.tabs)
        right_layout.addWidget(QLabel("Log"))
        right_layout.addWidget(self.log_box)

        splitter = QSplitter()
        splitter.addWidget(self.video)
        splitter.addWidget(right)
        splitter.setSizes([1070, 480])

        self.setCentralWidget(splitter)

        self.display_timer = QTimer()
        self.display_timer.timeout.connect(self.update_display)
        self.display_timer.start(max(15, int(1000 / max(1, config.fps))))

        self.create_menu()
        self.refresh_devices()
        self.sync_widgets_from_config()

        self.camera.start()

    def create_menu(self):
        save_action = QAction("Save Config", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.save_config_clicked)

        self.menuBar().addAction(save_action)

    def log(self, msg: str):
        print(msg)
        self.log_box.append(str(msg))

    def make_camera_tab(self):
        tab = QWidget()
        layout = QGridLayout(tab)

        self.device_box = QComboBox()

        refresh_btn = QPushButton("Refresh Devices")
        refresh_btn.clicked.connect(self.refresh_devices)

        open_btn = QPushButton("Open / Restart Camera")
        open_btn.clicked.connect(self.restart_camera)

        self.width_spin = QSpinBox()
        self.width_spin.setRange(160, 7680)

        self.height_spin = QSpinBox()
        self.height_spin.setRange(120, 4320)

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 240)

        self.fourcc_box = QComboBox()
        self.fourcc_box.addItems(["MJPG", "YUYV"])

        formats_btn = QPushButton("Show Supported Formats")
        formats_btn.clicked.connect(self.show_formats)

        controls_btn = QPushButton("Show V4L2 Controls")
        controls_btn.clicked.connect(lambda: self.log(get_v4l2_controls(config.device)))

        self.ctrl_name = QLineEdit("brightness")

        self.ctrl_value = QSpinBox()
        self.ctrl_value.setRange(-99999, 99999)

        set_ctrl_btn = QPushButton("Set V4L2 Control")
        set_ctrl_btn.clicked.connect(self.set_hardware_control)

        row = 0

        layout.addWidget(QLabel("Device"), row, 0)
        layout.addWidget(self.device_box, row, 1)
        layout.addWidget(refresh_btn, row, 2)
        row += 1

        layout.addWidget(QLabel("Width"), row, 0)
        layout.addWidget(self.width_spin, row, 1)
        row += 1

        layout.addWidget(QLabel("Height"), row, 0)
        layout.addWidget(self.height_spin, row, 1)
        row += 1

        layout.addWidget(QLabel("FPS"), row, 0)
        layout.addWidget(self.fps_spin, row, 1)
        row += 1

        layout.addWidget(QLabel("FOURCC"), row, 0)
        layout.addWidget(self.fourcc_box, row, 1)
        row += 1

        layout.addWidget(open_btn, row, 0, 1, 3)
        row += 1

        layout.addWidget(formats_btn, row, 0, 1, 3)
        row += 1

        layout.addWidget(controls_btn, row, 0, 1, 3)
        row += 1

        layout.addWidget(QLabel("Control"), row, 0)
        layout.addWidget(self.ctrl_name, row, 1)
        row += 1

        layout.addWidget(QLabel("Value"), row, 0)
        layout.addWidget(self.ctrl_value, row, 1)
        row += 1

        layout.addWidget(set_ctrl_btn, row, 0, 1, 3)

        return tab

    def make_image_tab(self):
        tab = QWidget()
        layout = QGridLayout(tab)

        self.brightness_slider = self.make_slider(
            -100,
            100,
            config.brightness,
            lambda v: setattr(config, "brightness", v),
        )

        self.contrast_spin = self.make_double_spin(
            0.1,
            5.0,
            0.05,
            config.contrast,
            lambda v: setattr(config, "contrast", v),
        )

        self.gamma_spin = self.make_double_spin(
            0.1,
            5.0,
            0.05,
            config.gamma,
            lambda v: setattr(config, "gamma", v),
        )

        self.saturation_spin = self.make_double_spin(
            0.0,
            3.0,
            0.05,
            config.saturation,
            lambda v: setattr(config, "saturation", v),
        )

        self.threshold_slider = self.make_slider(
            0,
            255,
            config.threshold_value,
            lambda v: setattr(config, "threshold_value", v),
        )

        row = 0

        layout.addWidget(QLabel("Brightness"), row, 0)
        layout.addWidget(self.brightness_slider, row, 1)
        row += 1

        layout.addWidget(QLabel("Contrast"), row, 0)
        layout.addWidget(self.contrast_spin, row, 1)
        row += 1

        layout.addWidget(QLabel("Gamma"), row, 0)
        layout.addWidget(self.gamma_spin, row, 1)
        row += 1

        layout.addWidget(QLabel("Saturation"), row, 0)
        layout.addWidget(self.saturation_spin, row, 1)
        row += 1

        layout.addWidget(QLabel("Threshold"), row, 0)
        layout.addWidget(self.threshold_slider, row, 1)
        row += 1

        toggles = [
            ("Grayscale", "grayscale"),
            ("Invert", "invert"),
            ("Sharpen", "sharpen"),
            ("Denoise", "denoise"),
            ("CLAHE Local Contrast", "clahe"),
            ("Edges", "edges"),
            ("Binary Threshold", "threshold"),
            ("Adaptive Threshold", "adaptive_threshold"),
            ("False Color", "false_color"),
        ]

        for text, attr in toggles:
            cb = QCheckBox(text)
            cb.setChecked(getattr(config, attr))
            cb.toggled.connect(lambda checked, a=attr: setattr(config, a, checked))
            layout.addWidget(cb, row, 0, 1, 2)
            row += 1

        return tab

    def make_overlay_tab(self):
        tab = QWidget()
        layout = QGridLayout(tab)

        toggles = [
            ("Crosshair", "show_crosshair"),
            ("Grid", "show_grid"),
            ("Ruler", "show_ruler"),
            ("Focus Meter", "show_focus_meter"),
            ("Histogram", "show_histogram"),
            ("Measurements", "show_measurements"),
            ("Status Text", "show_status_text"),
        ]

        row = 0

        for text, attr in toggles:
            cb = QCheckBox(text)
            cb.setChecked(getattr(config, attr))
            cb.toggled.connect(lambda checked, a=attr: setattr(config, a, checked))
            layout.addWidget(cb, row, 0, 1, 2)
            row += 1

        self.zoom_spin = self.make_double_spin(
            1.0,
            30.0,
            0.25,
            config.zoom,
            lambda v: setattr(config, "zoom", v),
        )

        layout.addWidget(QLabel("Zoom"), row, 0)
        layout.addWidget(self.zoom_spin, row, 1)
        row += 1

        self.known_mm_spin = self.make_double_spin(
            0.001,
            10000.0,
            0.1,
            config.known_mm,
            lambda v: setattr(config, "known_mm", v),
        )
        self.known_mm_spin.setDecimals(4)

        layout.addWidget(QLabel("Known Distance mm"), row, 0)
        layout.addWidget(self.known_mm_spin, row, 1)
        row += 1

        calibrate_btn = QPushButton("Calibrate From Active Measurement")
        calibrate_btn.clicked.connect(self.calibrate_from_measurement)
        layout.addWidget(calibrate_btn, row, 0, 1, 2)
        row += 1

        lock_btn = QPushButton("Lock Measurement  L")
        lock_btn.clicked.connect(self.video.lock_current_measurement)
        layout.addWidget(lock_btn, row, 0, 1, 2)
        row += 1

        clear_btn = QPushButton("Clear Measurements  C")
        clear_btn.clicked.connect(self.video.clear_measurements)
        layout.addWidget(clear_btn, row, 0, 1, 2)
        row += 1

        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self.reset_view)
        layout.addWidget(reset_btn, row, 0, 1, 2)

        return tab

    def make_analysis_tab(self):
        tab = QWidget()
        layout = QGridLayout(tab)

        buttons = [
            ("OCR Chip Marking", self.run_ocr),
            ("Detect Solder Bridge Candidates", self.detect_solder_bridges),
            ("Segment Pads / Components", self.segment_pads_components),
            ("Estimate Trace Width From Measurement", self.estimate_trace_width),
            ("Auto White Balance Preview", self.auto_white_balance_preview),
            ("Add Focus Stack Frame", self.add_focus_stack_frame),
            ("Build Focus Stack", self.build_focus_stack),
            ("Clear Focus Stack", self.clear_focus_stack),
            ("Add HDR Frame", self.add_hdr_frame),
            ("Build HDR Fusion", self.build_hdr),
            ("Clear HDR Frames", self.clear_hdr),
            ("Generate HTML Report", self.generate_report),
        ]

        for row, (text, callback) in enumerate(buttons):
            btn = QPushButton(text)
            btn.clicked.connect(callback)
            layout.addWidget(btn, row, 0)

        return tab

    def make_output_tab(self):
        tab = QWidget()
        layout = QGridLayout(tab)

        snap_btn = QPushButton("Snapshot  P")
        snap_btn.clicked.connect(self.save_snapshot)

        rec_btn = QPushButton("Start / Stop PyAV Recording  R")
        rec_btn.clicked.connect(self.toggle_recording)

        pause_btn = QPushButton("Pause / Resume Camera  Space")
        pause_btn.clicked.connect(self.toggle_pause)

        self.snapshot_clean_cb = QCheckBox("Clean Snapshots")
        self.snapshot_clean_cb.setChecked(config.snapshot_clean)
        self.snapshot_clean_cb.toggled.connect(
            lambda v: setattr(config, "snapshot_clean", v)
        )

        self.recording_clean_cb = QCheckBox("Clean Recording")
        self.recording_clean_cb.setChecked(config.recording_clean)
        self.recording_clean_cb.toggled.connect(
            lambda v: setattr(config, "recording_clean", v)
        )

        self.codec_box = QComboBox()
        self.codec_box.addItems(["libx264", "mpeg4"])
        self.codec_box.setCurrentText(config.record_codec)
        self.codec_box.currentTextChanged.connect(
            lambda v: setattr(config, "record_codec", v)
        )

        self.crf_spin = QSpinBox()
        self.crf_spin.setRange(0, 35)
        self.crf_spin.setValue(config.record_crf)
        self.crf_spin.valueChanged.connect(lambda v: setattr(config, "record_crf", v))

        self.preset_box = QComboBox()
        self.preset_box.addItems(
            [
                "ultrafast",
                "superfast",
                "veryfast",
                "faster",
                "fast",
                "medium",
                "slow",
            ]
        )
        self.preset_box.setCurrentText(config.record_preset)
        self.preset_box.currentTextChanged.connect(
            lambda v: setattr(config, "record_preset", v)
        )

        row = 0

        layout.addWidget(snap_btn, row, 0, 1, 2)
        row += 1

        layout.addWidget(rec_btn, row, 0, 1, 2)
        row += 1

        layout.addWidget(pause_btn, row, 0, 1, 2)
        row += 1

        layout.addWidget(self.snapshot_clean_cb, row, 0, 1, 2)
        row += 1

        layout.addWidget(self.recording_clean_cb, row, 0, 1, 2)
        row += 1

        layout.addWidget(QLabel("Codec"), row, 0)
        layout.addWidget(self.codec_box, row, 1)
        row += 1

        layout.addWidget(QLabel("CRF"), row, 0)
        layout.addWidget(self.crf_spin, row, 1)
        row += 1

        layout.addWidget(QLabel("Preset"), row, 0)
        layout.addWidget(self.preset_box, row, 1)

        return tab

    def make_slider(self, mn, mx, value, callback):
        slider = QSlider(Qt.Horizontal)
        slider.setRange(mn, mx)
        slider.setValue(value)
        slider.valueChanged.connect(callback)
        return slider

    def make_double_spin(self, mn, mx, step, value, callback):
        spin = QDoubleSpinBox()
        spin.setRange(mn, mx)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    def sync_widgets_from_config(self):
        self.width_spin.setValue(config.width)
        self.height_spin.setValue(config.height)
        self.fps_spin.setValue(config.fps)
        self.fourcc_box.setCurrentText(config.fourcc)

    def refresh_devices(self):
        self.device_box.clear()

        devices = list_video_devices()

        for dev in devices:
            label = get_device_label(dev)
            self.device_box.addItem(f"{dev} — {label}", dev)

        index = self.device_box.findData(config.device)

        if index >= 0:
            self.device_box.setCurrentIndex(index)
        elif devices:
            self.device_box.setCurrentIndex(0)

    def restart_camera(self):
        device = self.device_box.currentData()

        if device:
            config.device = device

        config.width = self.width_spin.value()
        config.height = self.height_spin.value()
        config.fps = self.fps_spin.value()
        config.fourcc = self.fourcc_box.currentText()

        self.display_timer.setInterval(max(15, int(1000 / max(1, config.fps))))

        self.camera.start()

    def show_formats(self):
        self.log(get_supported_formats(config.device))

    def set_hardware_control(self):
        ok, msg = set_v4l2_control(
            config.device,
            self.ctrl_name.text().strip(),
            self.ctrl_value.value(),
        )

        if ok:
            self.log("V4L2 control set.")
        else:
            self.log(f"V4L2 control failed: {msg}")

    def save_config_clicked(self):
        save_config()
        self.log(f"Saved config: {CONFIG_FILE}")

    def reset_view(self):
        config.zoom = 1.0
        config.pan_x = 0
        config.pan_y = 0

        self.zoom_spin.blockSignals(True)
        self.zoom_spin.setValue(1.0)
        self.zoom_spin.blockSignals(False)

    def toggle_pause(self):
        self.camera.paused = not self.camera.paused
        self.log("Paused." if self.camera.paused else "Resumed.")

    def on_raw_frame(self, frame: np.ndarray):
        self.raw_frame = frame

    def update_display(self):
        if self.raw_frame is None:
            return

        raw = self.raw_frame.copy()
        processed = FrameProcessor.process(raw)

        self.processed_frame = processed.copy()
        display = processed.copy()

        now = time.time()
        dt = max(0.0001, now - self.last_tick)
        self.last_tick = now

        instant_fps = 1.0 / dt
        self.ui_fps = (
            instant_fps
            if self.ui_fps <= 0
            else 0.9 * self.ui_fps + 0.1 * instant_fps
        )

        self.draw_overlays(display)

        self.display_frame = display.copy()
        self.video.set_frame_shape(display)

        if self.recorder.active:
            frame_to_record = (
                self.processed_frame
                if config.recording_clean
                else self.display_frame
            )
            self.recorder.write(frame_to_record)

        qimg = frame_to_qimage(display)
        pix = QPixmap.fromImage(qimg)

        self.video.setPixmap(
            pix.scaled(
                self.video.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

        if abs(self.zoom_spin.value() - config.zoom) > 0.001:
            self.zoom_spin.blockSignals(True)
            self.zoom_spin.setValue(config.zoom)
            self.zoom_spin.blockSignals(False)

    def draw_overlays(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        fscore = focus_score(frame)

        if config.show_status_text:
            rec_text = ""

            if self.recorder.active:
                rec_text = f" REC drop:{self.recorder.dropped_frames}"

            draw_text(
                frame,
                f"{config.device} {config.width}x{config.height}@{config.fps} "
                f"UI:{self.ui_fps:.1f} Focus:{fscore:.0f} Zoom:{config.zoom:.2f}{rec_text}",
                20,
                35,
                0.6,
                (100, 255, 80),
                2,
            )

            draw_text(
                frame,
                f"Cal: {config.pixels_per_mm:.3f} px/mm | Known: {config.known_mm:.4f} mm",
                20,
                65,
                0.55,
                (255, 255, 255),
                1,
            )

        if self.recorder.active:
            cv2.circle(frame, (36, 100), 11, (0, 0, 255), -1)
            draw_text(frame, "REC", 56, 108, 0.75, (0, 0, 255), 2)

        if config.show_grid:
            step = max(50, int(config.pixels_per_mm))

            for x in range(0, w, step):
                cv2.line(frame, (x, 0), (x, h), (70, 70, 70), 1)

            for y in range(0, h, step):
                cv2.line(frame, (0, y), (w, y), (70, 70, 70), 1)

        if config.show_crosshair:
            cx, cy = w // 2, h // 2
            cv2.line(frame, (cx - 60, cy), (cx + 60, cy), (0, 255, 255), 1)
            cv2.line(frame, (cx, cy - 60), (cx, cy + 60), (0, 255, 255), 1)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        if config.show_ruler:
            ruler_mm = 5.0
            px_len = int(config.pixels_per_mm * ruler_mm)
            px_len = max(20, min(px_len, w - 120))

            x0 = 45
            y0 = h - 45

            cv2.line(frame, (x0, y0), (x0 + px_len, y0), (255, 255, 255), 2)
            cv2.line(frame, (x0, y0 - 10), (x0, y0 + 10), (255, 255, 255), 2)
            cv2.line(
                frame,
                (x0 + px_len, y0 - 10),
                (x0 + px_len, y0 + 10),
                (255, 255, 255),
                2,
            )
            draw_text(frame, f"{ruler_mm:g} mm", x0, y0 - 15, 0.55)

        if config.show_focus_meter:
            bar_w = 260
            val = int(min(1.0, fscore / 2500.0) * bar_w)

            x0 = 45
            y0 = h - 85

            cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + 16), (90, 90, 90), 1)
            cv2.rectangle(frame, (x0, y0), (x0 + val, y0 + 16), (0, 220, 255), -1)

        if config.show_histogram:
            self.draw_histogram(frame)

        if config.show_measurements:
            self.draw_measurements(frame)

    def draw_histogram(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist, 0, 100, cv2.NORM_MINMAX)

        x0 = w - 285
        y0 = h - 135

        cv2.rectangle(frame, (x0, y0), (x0 + 265, y0 + 115), (0, 0, 0), -1)

        for i in range(1, 256):
            cv2.line(
                frame,
                (x0 + i - 1, y0 + 105 - int(hist[i - 1][0])),
                (x0 + i, y0 + 105 - int(hist[i][0])),
                (220, 220, 220),
                1,
            )

    def draw_measurements(self, frame: np.ndarray):
        for p1, p2, label in self.video.locked_measurements:
            self.draw_measurement(frame, p1, p2, label, (255, 0, 255))

        if self.video.measurement_start is not None and self.video.measurement_end is not None:
            self.draw_measurement(
                frame,
                self.video.measurement_start,
                self.video.measurement_end,
                "active",
                (255, 0, 0),
            )

    def draw_measurement(
        self,
        frame: np.ndarray,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        label: str,
        color: Tuple[int, int, int],
    ):
        cv2.circle(frame, p1, 5, (0, 255, 0), -1)
        cv2.circle(frame, p2, 5, (0, 255, 0), -1)
        cv2.line(frame, p1, p2, color, 2)

        px, mm, mil = measurement_values(p1, p2)

        x = max(10, min(p1[0], frame.shape[1] - 360))
        y = max(25, p1[1] - 12)

        draw_text(
            frame,
            f"{label}: {px:.1f}px  {mm:.4f}mm  {mil:.2f}mil",
            x,
            y,
            0.55,
            (255, 255, 0),
            2,
        )

    def save_snapshot(self):
        if self.processed_frame is None or self.display_frame is None:
            return

        img = self.processed_frame if config.snapshot_clean else self.display_frame
        file = SNAPSHOT_DIR / f"snapshot_{now_stamp()}.png"

        cv2.imwrite(str(file), img)
        self.log(f"Snapshot saved: {file}")

    def toggle_recording(self):
        if self.recorder.active:
            file = self.recorder.stop()

            msg = f"PyAV recording stopped: {file}"

            if self.recorder.dropped_frames:
                msg += f" | dropped frames: {self.recorder.dropped_frames}"

            if self.recorder.error:
                msg += f" | error: {self.recorder.error}"

            self.log(msg)
            return

        if self.processed_frame is None:
            self.log("No frame available for recording.")
            return

        h, w = self.processed_frame.shape[:2]

        file = self.recorder.start(w, h, config.fps)
        self.log(f"PyAV recording started: {file}")

    def calibrate_from_measurement(self):
        p1 = self.video.measurement_start
        p2 = self.video.measurement_end

        if p1 is None or p2 is None:
            self.log("Draw a measurement over a known distance first.")
            return

        px, _, _ = measurement_values(p1, p2)

        if px <= 0:
            self.log("Measurement length is zero.")
            return

        config.pixels_per_mm = px / config.known_mm

        save_config()
        self.log(f"Calibrated: {config.pixels_per_mm:.6f} px/mm")

    def run_ocr(self):
        if self.processed_frame is None:
            return

        if not TESSERACT_AVAILABLE:
            self.log("pytesseract is unavailable. Install pytesseract and tesseract-ocr.")
            return

        gray = cv2.cvtColor(self.processed_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        th = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            7,
        )

        text = pytesseract.image_to_string(th, config="--psm 6").strip()

        file = ANALYSIS_DIR / f"ocr_input_{now_stamp()}.png"
        cv2.imwrite(str(file), th)

        self.last_analysis["OCR"] = text
        self.log(f"OCR result:\n{text}\nSaved OCR input: {file}")

    def detect_solder_bridges(self):
        if self.processed_frame is None:
            return

        img = self.processed_frame.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        _, th = cv2.threshold(
            blur,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        kernel = np.ones((3, 3), np.uint8)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(
            th,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        count = 0

        for c in contours:
            area = cv2.contourArea(c)

            if not (25 <= area <= 7000):
                continue

            x, y, w, h = cv2.boundingRect(c)
            aspect = max(w, h) / max(1, min(w, h))

            if aspect >= 2.0 and min(w, h) <= 45:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                count += 1

        file = ANALYSIS_DIR / f"solder_bridge_candidates_{now_stamp()}.png"
        cv2.imwrite(str(file), img)

        self.last_analysis["Solder bridge candidates"] = f"{count} candidates; {file}"
        self.log(f"Solder bridge candidates: {count}; saved: {file}")

    def segment_pads_components(self):
        if self.processed_frame is None:
            return

        img = self.processed_frame.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 45, 130)

        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(
            closed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        count = 0

        for c in contours:
            area = cv2.contourArea(c)

            if not (80 <= area <= 80000):
                continue

            x, y, w, h = cv2.boundingRect(c)

            if w < 4 or h < 4:
                continue

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            count += 1

        file = ANALYSIS_DIR / f"segmentation_{now_stamp()}.png"
        cv2.imwrite(str(file), img)

        self.last_analysis["Segmentation"] = f"{count} regions; {file}"
        self.log(f"Segmented regions: {count}; saved: {file}")

    def estimate_trace_width(self):
        if self.processed_frame is None:
            return

        p1 = self.video.measurement_start
        p2 = self.video.measurement_end

        if p1 is None or p2 is None:
            self.log("Draw a line across the trace first.")
            return

        gray = cv2.cvtColor(self.processed_frame, cv2.COLOR_BGR2GRAY)

        n = int(math.hypot(p2[0] - p1[0], p2[1] - p1[1]))

        if n < 5:
            self.log("Measurement line is too short.")
            return

        xs = np.linspace(p1[0], p2[0], n).astype(np.int32)
        ys = np.linspace(p1[1], p2[1], n).astype(np.int32)

        xs = np.clip(xs, 0, gray.shape[1] - 1)
        ys = np.clip(ys, 0, gray.shape[0] - 1)

        vals = gray[ys, xs].astype(np.float32)

        low = np.percentile(vals, 20)
        high = np.percentile(vals, 80)
        thresh = (low + high) / 2.0

        bright_mask = vals > thresh

        runs = []
        start = None

        for i, value in enumerate(bright_mask):
            if value and start is None:
                start = i

            elif not value and start is not None:
                runs.append((start, i - 1))
                start = None

        if start is not None:
            runs.append((start, len(bright_mask) - 1))

        if not runs:
            self.log("No trace-like bright run detected.")
            return

        longest = max(runs, key=lambda r: r[1] - r[0])

        px = longest[1] - longest[0] + 1
        mm = px / config.pixels_per_mm if config.pixels_per_mm > 0 else 0
        mil = mm * 39.3701

        result = f"Trace width estimate: {px}px = {mm:.5f} mm = {mil:.3f} mil"

        self.last_analysis["Trace width"] = result
        self.log(result)

    def auto_white_balance_preview(self):
        if self.raw_frame is None:
            return

        img = self.raw_frame.astype(np.float32)

        avg_b, avg_g, avg_r = img.reshape(-1, 3).mean(axis=0)

        avg_b = max(avg_b, 1.0)
        avg_g = max(avg_g, 1.0)
        avg_r = max(avg_r, 1.0)

        gray = (avg_b + avg_g + avg_r) / 3.0

        gains = np.array(
            [
                gray / avg_b,
                gray / avg_g,
                gray / avg_r,
            ],
            dtype=np.float32,
        )

        corrected = np.clip(img * gains, 0, 255).astype(np.uint8)

        file = ANALYSIS_DIR / f"white_balance_preview_{now_stamp()}.png"
        cv2.imwrite(str(file), corrected)

        self.last_analysis["White balance"] = f"BGR gains {gains.tolist()}; {file}"
        self.log(f"White balance preview saved: {file}; BGR gains={gains}")

    def add_focus_stack_frame(self):
        if self.processed_frame is None:
            return

        self.focus_stack_frames.append(self.processed_frame.copy())

        file = STACK_DIR / f"stack_frame_{len(self.focus_stack_frames):03d}_{now_stamp()}.png"
        cv2.imwrite(str(file), self.processed_frame)

        self.log(f"Focus stack frame added: {len(self.focus_stack_frames)}; {file}")

    def build_focus_stack(self):
        if len(self.focus_stack_frames) < 2:
            self.log("Need at least 2 focus stack frames.")
            return

        frames = self.focus_stack_frames
        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

        focus_maps = [
            np.abs(cv2.Laplacian(g, cv2.CV_64F))
            for g in grays
        ]

        stack = np.stack(focus_maps, axis=0)
        best = np.argmax(stack, axis=0)

        result = np.zeros_like(frames[0])

        for i, frame in enumerate(frames):
            mask = best == i
            result[mask] = frame[mask]

        result = cv2.medianBlur(result, 3)

        file = STACK_DIR / f"focus_stacked_{now_stamp()}.png"
        cv2.imwrite(str(file), result)

        self.last_analysis["Focus stack"] = f"{len(frames)} frames; {file}"
        self.log(f"Focus stack saved: {file}")

    def clear_focus_stack(self):
        self.focus_stack_frames.clear()
        self.log("Focus stack frames cleared.")

    def add_hdr_frame(self):
        if self.raw_frame is None:
            return

        self.hdr_frames.append(self.raw_frame.copy())
        self.log(f"HDR frame added: {len(self.hdr_frames)}")

    def build_hdr(self):
        if len(self.hdr_frames) < 2:
            self.log("Need at least 2 HDR frames.")
            return

        frames = [f.astype(np.float32) for f in self.hdr_frames]
        arr = np.stack(frames, axis=0)

        median = np.median(arr, axis=0)
        result = np.clip(median, 0, 255).astype(np.uint8)

        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)

        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)

        result = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

        file = HDR_DIR / f"hdr_fusion_{now_stamp()}.png"
        cv2.imwrite(str(file), result)

        self.last_analysis["HDR"] = f"{len(self.hdr_frames)} frames; {file}"
        self.log(f"HDR fusion saved: {file}")

    def clear_hdr(self):
        self.hdr_frames.clear()
        self.log("HDR frames cleared.")

    def generate_report(self):
        file = REPORT_DIR / f"inspection_report_{now_stamp()}.html"

        snapshots = sorted(SNAPSHOT_DIR.glob("*.png"))[-12:]
        videos = sorted(VIDEO_DIR.glob("*.mp4"))[-8:]
        analysis_images = sorted(ANALYSIS_DIR.glob("*.png"))[-12:]

        html = [
            "<!doctype html>",
            "<html>",
            "<head>",
            "<meta charset='utf-8'>",
            "<title>MicroCam Inspection Report</title>",
            "<style>",
            "body{font-family:Arial,sans-serif;margin:24px;background:#fafafa;color:#222;}",
            "img{max-width:460px;border:1px solid #bbb;margin:8px;background:white;}",
            "pre{background:#eee;padding:12px;border-radius:8px;overflow:auto;}",
            ".grid{display:flex;flex-wrap:wrap;gap:12px;}",
            ".card{background:white;border:1px solid #ddd;padding:10px;border-radius:8px;}",
            "</style>",
            "</head>",
            "<body>",
            "<h1>MicroCam Inspection Report</h1>",
            f"<p>Generated: {datetime.now()}</p>",
            "<h2>Configuration</h2>",
            "<pre>",
            json.dumps(asdict(config), indent=2),
            "</pre>",
            "<h2>Latest Analysis</h2>",
            "<ul>",
        ]

        for key, value in self.last_analysis.items():
            html.append(f"<li><b>{key}</b>: {value}</li>")

        html += [
            "</ul>",
            "<h2>Recent Snapshots</h2>",
            "<div class='grid'>",
        ]

        for img in snapshots:
            rel = img.relative_to(REPORT_DIR.parent)
            html.append(f"<div class='card'><img src='../{rel}'><p>{img.name}</p></div>")

        html += [
            "</div>",
            "<h2>Recent Analysis Images</h2>",
            "<div class='grid'>",
        ]

        for img in analysis_images:
            rel = img.relative_to(REPORT_DIR.parent)
            html.append(f"<div class='card'><img src='../{rel}'><p>{img.name}</p></div>")

        html += [
            "</div>",
            "<h2>Recent Videos</h2>",
            "<ul>",
        ]

        for vid in videos:
            rel = vid.relative_to(REPORT_DIR.parent)
            html.append(f"<li><a href='../{rel}'>{vid.name}</a></li>")

        html += [
            "</ul>",
            "</body>",
            "</html>",
        ]

        file.write_text("\n".join(html))
        self.log(f"Report generated: {file}")

    def closeEvent(self, event):
        save_config()

        self.camera.stop()

        if self.recorder.active:
            self.recorder.stop()

        event.accept()


def main():
    load_config()

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()

    rc = app.exec()
    sys.exit(rc)


if __name__ == "__main__":
    main()