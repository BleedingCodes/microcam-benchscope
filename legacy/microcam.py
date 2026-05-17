import cv2, time, os


DEVICE = "/dev/video0"

WIDTH = 1440
HEIGHT = 1180
FPS = 35
zoom = 1.000
last_record_toggle = 0
pan_x = 0
pan_y = 0
pixel_scale = 1.0
pixels_per_mm = 15.08
known_mm = 2.54
known_mils = known_mm * 39.3701
threshold_value = 120
show_help = False
paused = False
last_frame = None
recording = False
writer = None
crosshair = True
edge_mode = False
grid = False
magnifier = False
measure_mode = False
point1 = None
point2 = None
dragging = False
invert_mode = False
threshold_mode = False


CAPTURE_DIR = "captures"

os.makedirs(CAPTURE_DIR, exist_ok=True)

cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)

if not cap.isOpened():
    print("Failed to open camera")
    exit()

print("  q = quit")
print("  p = save snapshot")
print("  WASD = pan zoom")
print("  r = start/stop recording")
print("  f = fullscreen")
print("  space = freeze frame")
print("  + / - = zoom")
print("  c = toggle crosshair on/off")
print("  e = edge detection mode")
print("  g = grid overlay")
print("  m = magnifier")
print("  t = measure mode")
print("  x = clear measurement")
print("  i = invert mode")

snapshot_count = 0

fullscreen = False
prev_time = time.time()

def focus_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def mouse_callback(event, x, y, flags, param):
    global zoom

    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            zoom = min(zoom + 0.20, 5.0)
        else:
            zoom = max(1.0, zoom - 0.20)

        print(f"Zoom: {zoom:.2f}x")
    
    global point1, point2
    global measure_mode
    global dragging

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

while True:
    if not paused:
        ok, frame = cap.read()

        if not ok:
            print("Frame read failed")
            break

        last_frame = frame.copy()
    else:
        frame = last_frame.copy()

    # Software image adjustment
    alpha = 1
    beta = 20
    frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    # FPS calculation
    current_time = time.time()
    fps_actual = 1 / (current_time - prev_time)
    prev_time = current_time

    focus = focus_score(frame)

    if invert_mode:
        frame = cv2.bitwise_not(frame)

    cv2.putText(
        frame,
        f"{WIDTH}x{HEIGHT} FPS:{fps_actual:.1f} Zoom:{zoom:.2f}x Focus:{focus:.0f} Cal:{known_mm:.2f}mm",
        (20, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        .75,
        (100, 255, 10),
        2
    )
    if recording:
        cv2.putText(
            frame,
            "REC",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

# Digital zoom + pan
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

    if recording and writer is not None:
        writer.write(frame)
    
    if crosshair:
        h, w = frame.shape[:2]

        center_x = w // 2
        center_y = h // 2

        # Horizontal reticle
        cv2.line(
            frame,
            (center_x - 40, center_y),
            (center_x + 40, center_y),
            (0, 255, 255),
            1
        )

        # Vertical reticle
        cv2.line(
            frame,
            (center_x, center_y - 40),
            (center_x, center_y + 40),
            (0, 255, 255),
            1
        )

        # Center dot
        cv2.circle(
            frame,
            (center_x, center_y),
            2,
            (0, 0, 255),
            -1
        )

    if edge_mode:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 5), 5)
        edges = cv2.Canny(blur, 20, 25)
        frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    if grid:
        h, w = frame.shape[:2]
    
        spacing = int(100 * pixel_scale)
    
        for x in range(0, w, spacing):
            cv2.line(frame, (x, 0), (x, h), (80, 80, 80), 1)
    
            cv2.putText(
                frame,
                f"{x}px",
                (x + 2, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (100, 200, 255),
                1
            )
    
        for y in range(0, h, spacing):
            cv2.line(frame, (0, y), (w, y), (80, 80, 80), 1)
    
            cv2.putText(
                frame,
                f"{y}px",
                (2, y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (120, 120, 120),
                1
            )

    if magnifier:
        h, w = frame.shape[:2]

        mag_size = 120
        zoom_factor = 3

        cx = w // 2
        cy = h // 2

        x1 = max(0, cx - mag_size // 2)
        y1 = max(0, cy - mag_size // 2)

        x2 = min(w, x1 + mag_size)
        y2 = min(h, y1 + mag_size)

        roi = frame[y1:y2, x1:x2]

        magnified = cv2.resize(
            roi,
            (mag_size * zoom_factor, mag_size * zoom_factor)
        )

        mh, mw = magnified.shape[:2]

        frame[10:10+mh, w-mw-10:w-10] = magnified

    if point1 is not None:
        cv2.circle(frame, point1, 5, (0, 255, 0), -1)

    if point1 is not None and point2 is not None:
        cv2.circle(frame, point2, 5, (0, 255, 0), -1)
    
        cv2.line(frame, point1, point2, (255, 0, 0), 2)
    
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
    
        pixel_distance = (dx**2 + dy**2) ** 0.5
    
        text = f"{pixel_distance:.1f}px"
    
        if pixels_per_mm is not None:
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
            "e: edge mode",
            "m: magnifier",
            "t: measure mode",
            "x: clear measurement",
            "k: calibrate from current measurement",
            "[: smaller known distance",
            "]: larger known distance",
            "h: hide help",
        ]

        y = 100

        for line in help_lines:
            cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            y += 24

    cv2.imshow("MicroCam", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('p'):
        filename = os.path.join(
            CAPTURE_DIR,
            time.strftime("snapshot_%Y%m%d_%H%M%S.png")
        )
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")

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

    elif key == ord('+') or key == ord('='):
        zoom = min(zoom + 0.20, 5.0)
        print(f"Zoom: {zoom:.2f}x")

    elif key == ord('-'):
        zoom = max(0.10, zoom - 0.20)
        print(f"Zoom: {zoom:.2f}x")

    elif key == ord(' '):
        paused = not paused
        print("Paused" if paused else "Live")

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

                if not writer.isOpened():
                    print("Failed to start recording")
                    recording = False
                    writer = None
                else:
                    print(f"Recording started: {filename}")

            else:
                if writer is not None:
                    writer.release()
                    writer = None
                print("Recording stopped")

    elif key == ord('c'):
        crosshair = not crosshair
        print("Crosshair ON" if crosshair else "Crosshair OFF")

    elif key == ord('e'):
        edge_mode = not edge_mode
        print("Edge mode ON" if edge_mode else "Edge mode OFF")

    elif key == ord('g'):
        grid = not grid
        print("Grid ON" if grid else "Grid OFF")
    elif key == ord('w'):
        pan_y -= int(40 / zoom)

    elif key == ord('s'):
        pan_y += int(40 / zoom)

    elif key == ord('a'):
        pan_x -= int(40 / zoom)

    elif key == ord('d'):
        pan_x += int(40 / zoom)

    elif key == ord('m'):
        magnifier = not magnifier
        print("Magnifier ON" if magnifier else "Magnifier OFF")

    elif key == ord('t'):
        measure_mode = not measure_mode

        point1 = None
        point2 = None

        print("Measure mode ON" if measure_mode else "Measure mode OFF")

    elif key == ord('x'):
        point1 = None
        point2 = None
        print("Measurement cleared")

    elif key == ord('k'):
        if point1 is not None and point2 is not None:
            dx = point2[0] - point1[0]
            dy = point2[1] - point1[1]
            pixel_distance = (dx**2 + dy**2) ** 0.5

            if pixel_distance > 0:
                pixels_per_mm = pixel_distance / known_mm
                print(f"Calibrated: {pixels_per_mm:.3f} px/mm using {known_mm} mm")
        else:
            print("Draw a measurement first, then press k")

    elif key == ord('['):
        known_mm = max(0.01, known_mm - 0.01)
        known_mils = known_mm * 39.3701
        print(f"Known distance: {known_mm:.3f} mm ({known_mils:.1f} mil)")

    elif key == ord(']'):
        known_mm += 0.01
        known_mils = known_mm * 39.3701
        print(f"Known distance: {known_mm:.3f} mm ({known_mils:.1f} mil)")

    elif key == ord('h'):
        show_help = not show_help

    elif key == ord('i'):
        invert_mode = not invert_mode
        print("Invert mode ON" if invert_mode else "Invert mode OFF")

cap.release()
if writer is not None:
    writer.release()
cv2.destroyAllWindows()