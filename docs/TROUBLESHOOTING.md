========================================================
MICROCAM BENCHSCOPE TROUBLESHOOTING
========================================================

NO VIDEO DEVICE FOUND
--------------------------------------------------------

Check:
ls /dev/video*

Check V4L2 devices:
v4l2-ctl --list-devices

Verify webcam permissions:
groups

You may need:
sudo usermod -aG video $USER

Then reboot or relogin.

--------------------------------------------------------
BLACK SCREEN
--------------------------------------------------------

Possible causes:
- Capture card already in use
- Unsupported resolution
- Unsupported FOURCC
- HDMI source not active

Recommended settings:
1920x1080
MJPG
30 FPS

--------------------------------------------------------
HIGH LATENCY
--------------------------------------------------------

Recommended:
- Use MJPG
- Lower resolution
- Lower FPS
- Close browser tabs
- Disable histogram overlay

--------------------------------------------------------
RECORDING FAILS
--------------------------------------------------------

FFMPEG VERSION:
Check:
ffmpeg -version

PYAV VERSION:
Check:
python3 -c "import av; print(av.__version__)"

--------------------------------------------------------
OCR NOT WORKING
--------------------------------------------------------

Install:
sudo apt install tesseract-ocr

Test:
tesseract --version

Improve OCR:
- Increase lighting
- Enable CLAHE
- Sharpen image
- Improve focus

--------------------------------------------------------
POOR FOCUS SCORE
--------------------------------------------------------

Improve:
- Lighting
- Microscope stability
- Focus position
- Disable motion blur

--------------------------------------------------------
CAPTURE CARD NOT DETECTED
--------------------------------------------------------

Try:
lsusb

MacroSilicon devices often appear as:
MacroSilicon USB Video

Try unplugging/replugging.

--------------------------------------------------------
LOW FPS
--------------------------------------------------------

Try:
- MJPG format
- Lower overlays
- Disable denoise
- Disable histogram
- Lower resolution

--------------------------------------------------------
PYAV IMPORT ERROR
--------------------------------------------------------

Reinstall:
pip install --upgrade av

Sometimes FFmpeg/PyAV ABI mismatches occur after distro upgrades.

Use FFmpeg edition as fallback.

--------------------------------------------------------
QT PLATFORM ERRORS
--------------------------------------------------------

Try:
export QT_QPA_PLATFORM=xcb

Then rerun application.

--------------------------------------------------------
WAYLAND ISSUES
--------------------------------------------------------

Some Linux systems behave better using X11/XCB instead of Wayland.

Try:
QT_QPA_PLATFORM=xcb python3 app.py
