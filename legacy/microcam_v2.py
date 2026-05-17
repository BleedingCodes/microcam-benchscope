
# MicroCam Pro Inspection Suite (Single-File Version)

```python
import cv2
import time
import os
import json
import numpy as np

# ============================================================
# CONFIG
# ============================================================

DEVICE = "/dev/video0"
WIDTH = 1440
HEIGHT = 1180
FPS = 35

CAPTURE_DIR = "captures"
CONFIG_FILE = "microcam_config.json"

os.makedirs(CAPTURE_DIR, exist_ok=True)

# ============================================================
# GLOBAL STATE
# ============================================================

zoom = 1.0
pan_x = 0
pan_y = 0

paused = False
recording = False
fullscreen = False
show_help = False

crosshair = True
edge_mode = False
invert_mode = False
threshold_mode = False
threshold_value = 120
grayscale_mode = False
grid = False
magnifier = False
clean_snapshot = False

measure_mode = False
dragging = False
point1 = None
point2 = None

pixels_per_mm = 15.08
known_mm = 2.54

writer = None
last_frame = None
last_record_toggle = 0

# ============================================================
# CONFIG SAVE/LOAD
# ============================================================


def load_config():
    global pixels_per_mm
    global known_mm

    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)

            pixels_per_mm = data.get("pixels_per_mm", pixels_per_mm)
            known_mm = data.get("known_mm", known_mm)

            print(f"Loaded calibration: {pixels_per_mm:.3f} px/mm")

        except Exception as e:
            print("Config load failed:", e)



def save_config():
    data = {
        "pixels_per_mm": pixels_per_mm,
        "known_mm": known_mm,
    }

    with open(CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=4)

    print("Calibration saved")


load_config()

# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)

if not cap.isOpened():
    print("Failed to open camera")
    exit()

# ============================================================
# HELP
# ============================================================

print("""
================ MICROCAM PRO =================

q = quit
h = help overlay

p = snapshot
v = clean snapshot toggle
r = record
f = fullscreen
space = freeze

+/- or mouse wheel = zoom
WASD = pan

c = crosshair
m = magnifier

b = grayscale
i = invert

e = edge detect

y = threshold mode
,/. = threshold adjust

g = grid

Measure:
    t = measure mode
    drag mouse = measure
    x = clear measurement
    k = calibrate using measurement

[/] = adjust known calibration distance

================================================
""")

# ============================================================
# UTILITIES
# ============================================================


def focus_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# ============================================================
# MOUSE
# ============================================================


def mouse_callback(event, x, y, flags, param):
    global zoom
    global point1, point2
    global measure_mode
    global dragging

    # Mouse wheel zoom
    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            zoom = min(zoom + 0.2, 8.0)
        else:
            zoom = max(1.0, zoom - 0.2)

        print(f"Zoom: {zoom:.2f}x")

    # Measurement mode
    if measure_mode:

        if event == cv2.EVENT_LBUTTONDOWN:
            point1 = (x, y)
            point2 = (x, y)
            dragging = True

        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            point2 = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            point2 = (x, y)
            dragging = False


cv2.namedWindow("MicroCam", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("MicroCam", mouse_callback)

# ============================================================
# MAIN LOOP
# ============================================================

prev_time = time.time()

while True:

    # ========================================================
    # FRAME ACQUISITION
    # ========================================================

    if not paused:
        ok, raw_frame = cap.read()

        if not ok:
            print("Frame read failed")
            break

        last_frame = raw_frame.copy()

    else:
        raw_frame = last_frame.copy()

    frame = raw_frame.copy()

    # ========================================================
    # IMAGE ADJUSTMENTS
    # ========================================================

    alpha = 1.0
    beta = 20

    frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    # ========================================================
    # ANALYSIS MODES
    # ========================================================

    if grayscale_mode:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if invert_mode:
        frame = cv2.bitwise_not(frame)

    if edge_mode:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blur, 40, 100)
        frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    if threshold_mode:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(
            gray,
            threshold_value,
            255,
            cv2.THRESH_BINARY
        )

        frame = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    # ========================================================
    # DIGITAL ZOOM + PAN
    # ========================================================

    if zoom > 1.0:

        h, w = frame.shape[:2]

        new_w = int(w / zoom)
        new_h = int(h / zoom)

        max_x = w - new_w
        max_y = h - new_h

        x1 = max(0, min((w - new_w) // 2 + pan_x, max_x))
        y1 = max(0, min((h - new_h) // 2 + pan_y, max_y))

        cropped = frame[y1:y1 + new_h, x1:x1 + new_w]

        frame = cv2.resize(cropped, (w, h))

    # ========================================================
    # RECORDING
    # ========================================================

    if recording and writer is not None:
        writer.write(frame)

    # ========================================================
    # STATS
    # ========================================================

    current_time = time.time()
    fps_actual = 1 / (current_time - prev_time)
    prev_time = current_time

    focus = focus_score(frame)

    # ========================================================
    # HUD
    # ========================================================

    cv2.putText(
        frame,
        f"{WIDTH}x{HEIGHT} FPS:{fps_actual:.1f} Zoom:{zoom:.2f}x Focus:{focus:.0f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (100, 255, 10),
        2
    )

    cv2.putText(
        frame,
        f"Calibration: {pixels_per_mm:.2f} px/mm | Known:{known_mm:.2f} mm",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    if recording:
        cv2.putText(
            frame,
            "REC",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    # ========================================================
    # CROSSHAIR
    # ========================================================

    h, w = frame.shape[:2]

    center_x = w // 2
    center_y = h // 2

    if crosshair:

        cv2.line(
            frame,
            (center_x - 40, center_y),
            (center_x + 40, center_y),
            (0, 255, 255),
            1
        )

        cv2.line(
            frame,
            (center_x, center_y - 40),
            (center_x, center_y + 40),
            (0, 255, 255),
            1
        )

        cv2.circle(
            frame,
            (center_x, center_y),
            3,
            (0, 0, 255),
            -1
        )

    # ========================================================
    # GRID
    # ========================================================

    if grid:

        spacing = 100

        for x in range(0, w, spacing):
            cv2.line(frame, (x, 0), (x, h), (80, 80, 80), 1)

        for y in range(0, h, spacing):
            cv2.line(frame, (0, y), (w, y), (80, 80, 80), 1)

    # ========================================================
    # MAGNIFIER
    # ========================================================

    if magnifier:

        mag_size = 120
        zoom_factor = 3

        x1 = max(0, center_x - mag_size // 2)
        y1 = max(0, center_y - mag_size // 2)

        x2 = min(w, x1 + mag_size)
        y2 = min(h, y1 + mag_size)

        roi = frame[y1:y2, x1:x2]

        magnified = cv2.resize(
            roi,
            (mag_size * zoom_factor, mag_size * zoom_factor)
        )

        mh, mw = magnified.shape[:2]

        frame[10:10+mh, w-mw-10:w-10] = magnified

        cv2.rectangle(
            frame,
            (w-mw-10, 10),
            (w-10, 10+mh),
            (0, 255, 255),
            2
        )

    # ========================================================
    # MEASUREMENTS
    # ========================================================

    if point1 is not None:
        cv2.circle(frame, point1, 5, (0, 255, 0), -1)

    if point1 is not None and point2 is not None:

        cv2.circle(frame, point2, 5, (0, 255, 0), -1)

        cv2.line(frame, point1, point2, (255, 0, 0), 2)

        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]

        pixel_distance = (dx**2 + dy**2) ** 0.5

        text = f"{pixel_distance:.1f}px"

        if pixels_per_mm:
            mm = pixel_distance / pixels_per_mm
            mils = mm * 39.3701

            text += f" | {mm:.3f} mm | {mils:.1f} mil"

        cv2.putText(
            frame,
            text,
            (point1[0], point1[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

    # ========================================================
    # HELP
    # ========================================================

    if show_help:

        help_lines = [
            "q: quit",
            "p: snapshot",
            "r: record",
            "f: fullscreen",
            "space: freeze",
            "+/- or wheel: zoom",
            "WASD: pan",
            "c: crosshair",
            "g: grid",
            "m: magnifier",
            "e: edge detect",
            "b: grayscale",
            "i: invert",
            "y: threshold",
            ",/. : threshold adjust",
            "t: measure mode",
            "x: clear measurement",
            "k: calibrate",
            "h: hide help",
        ]

        y = 150

        for line in help_lines:
            cv2.putText(
                frame,
                line,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            y += 24

    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow("MicroCam", frame)

    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    # ========================================================
    # SNAPSHOT
    # ========================================================

    elif key == ord('p'):

        save_frame = raw_frame.copy() if clean_snapshot else frame.copy()

        filename = os.path.join(
            CAPTURE_DIR,
            time.strftime("snapshot_%Y%m%d_%H%M%S.png")
        )

        cv2.imwrite(filename, save_frame)

        print(f"Saved: {filename}")

    # ========================================================
    # CLEAN SNAPSHOT TOGGLE
    # ========================================================

    elif key == ord('v'):
        clean_snapshot = not clean_snapshot
        print("Clean snapshot ON" if clean_snapshot else "Clean snapshot OFF")

    # ========================================================
    # FULLSCREEN
    # ========================================================

    elif key == ord('f'):

        fullscreen = not fullscreen

        if fullscreen:
            cv2.setWindowProperty(
                "MicroCam",
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN
            )
        else:
            cv2.setWindowProperty(
                "MicroCam",
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_NORMAL
            )

    # ========================================================
    # ZOOM
    # ========================================================

    elif key == ord('+') or key == ord('='):
        zoom = min(zoom + 0.2, 8.0)

    elif key == ord('-'):
        zoom = max(1.0, zoom - 0.2)

    # ========================================================
    # PAN
    # ========================================================

    elif key == ord('w'):
        pan_y -= int(40 / zoom)

    elif key == ord('s'):
        pan_y += int(40 / zoom)

    elif key == ord('a'):
        pan_x -= int(40 / zoom)

    elif key == ord('d'):
        pan_x += int(40 / zoom)

    # ========================================================
    # FREEZE
    # ========================================================

    elif key == ord(' '):
        paused = not paused

    # ========================================================
    # RECORDING
    # ========================================================

    elif key == ord('r'):

        now = time.time()

        if now - last_record_toggle > 0.5:

            last_record_toggle = now
            recording = not recording

            if recording:

                filename = os.path.join(
                    CAPTURE_DIR,
                    time.strftime("recording_%Y%m%d_%H%M%S.avi")
                )

                fourcc = cv2.VideoWriter_fourcc(*'MJPG')

                writer = cv2.VideoWriter(
                    filename,
                    fourcc,
                    FPS,
                    (frame.shape[1], frame.shape[0])
                )

                if writer.isOpened():
                    print(f"Recording: {filename}")
                else:
                    print("Failed to start recording")
                    recording = False
                    writer = None

            else:

                if writer is not None:
                    writer.release()
                    writer = None

                print("Recording stopped")

    # ========================================================
    # TOGGLES
    # ========================================================

    elif key == ord('c'):
        crosshair = not crosshair

    elif key == ord('g'):
        grid = not grid

    elif key == ord('m'):
        magnifier = not magnifier

    elif key == ord('e'):
        edge_mode = not edge_mode

    elif key == ord('i'):
        invert_mode = not invert_mode

    elif key == ord('b'):
        grayscale_mode = not grayscale_mode

    elif key == ord('y'):
        threshold_mode = not threshold_mode

    elif key == ord(','):
        threshold_value = max(0, threshold_value - 5)
        print(f"Threshold: {threshold_value}")

    elif key == ord('.'):
        threshold_value = min(255, threshold_value + 5)
        print(f"Threshold: {threshold_value}")

    # ========================================================
    # MEASUREMENT
    # ========================================================

    elif key == ord('t'):

        measure_mode = not measure_mode

        point1 = None
        point2 = None

        print("Measure mode ON" if measure_mode else "Measure mode OFF")

    elif key == ord('x'):
        point1 = None
        point2 = None

    elif key == ord('k'):

        if point1 is not None and point2 is not None:

            dx = point2[0] - point1[0]
            dy = point2[1] - point1[1]

            pixel_distance = (dx**2 + dy**2) ** 0.5

            if pixel_distance > 0:

                pixels_per_mm = pixel_distance / known_mm

                print(
                    f"Calibrated: {pixels_per_mm:.3f} px/mm using {known_mm:.2f} mm"
                )

                save_config()

    elif key == ord('['):
        known_mm = max(0.01, known_mm - 0.01)

    elif key == ord(']'):
        known_mm += 0.01

    # ========================================================
    # HELP
    # ========================================================

    elif key == ord('h'):
        show_help = not show_help

# ============================================================
# CLEANUP
# ============================================================

cap.release()

if writer is not None:
    writer.release()

cv2.destroyAllWindows()
'''```

## Run

```bash
python3 microcam.py
```

## Recommended Hardware Improvements

* Diffused LED ring light
* Better microscope optics
* Stable stand
* Macro lens cleaning
* USB3 capture hardware

## Future Upgrade Ideas

* OCR chip reading
* AI solder defect detection
* Autofocus
* MP4/H264 encoding
* Focus stacking
* OpenGL acceleration
* Qt GUI
* Multi-camera support
* Image stitching
* HDR capture
'''