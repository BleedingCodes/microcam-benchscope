# MicroCam BenchScope

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PySide6](https://img.shields.io/badge/GUI-PySide6-green)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red)
![FFmpeg](https://img.shields.io/badge/Recording-FFmpeg-purple)
![Platform](https://img.shields.io/badge/Platform-Linux-lightgrey)
![License](https://img.shields.io/badge/License-MIT-yellow)

Linux HDMI microscope and USB capture-card inspection workstation for electronics repair, PCB analysis, solder inspection, OCR, measurements, focus stacking, HDR fusion, and recording.

Designed primarily for:
- MacroSilicon HDMI USB capture cards
- HDMI microscopes
- USB microscopes
- Linux V4L2 devices

Built with:
- Python
- PySide6
- OpenCV
- NumPy
- PyAV
- FFmpeg
- Tesseract OCR

--------------------------------------------------------
SCREENSHOTS
--------------------------------------------------------

MAIN UI

![microcam_hyperlab_ffmpeg UI](screenshots/ui/microcam_hyperlab_ffmpeg_image_menu_tab)

![microcam_hyperlab_pyav UI](screenshots/ui/microcam_hyperlab_pyav_image_menu_tab)

--------------------------------------------------------
FEATURES
--------------------------------------------------------

LIVE MICROSCOPE VIEWER
- Realtime HDMI microscope display
- Multi-device V4L2 support
- MJPG/YUYV support
- Zoom and pan
- Crosshair and ruler overlays

IMAGE PROCESSING
- Brightness
- Contrast
- Gamma
- Saturation
- CLAHE local contrast
- Sharpening
- Denoising
- Edge detection
- Threshold modes
- False-color visualization

MEASUREMENT TOOLS
- Pixel/mm calibration
- Locked measurements
- mm and mil conversion
- Trace-width estimation

ELECTRONICS INSPECTION
- OCR chip-marking reader
- Solder bridge candidate detection
- Pad/component segmentation
- Focus scoring

CAPTURE TOOLS
- PNG snapshots
- MP4 recording
- Focus stacking
- HDR fusion
- Panorama stitching
- HTML report generation

--------------------------------------------------------
REPOSITORY STRUCTURE
--------------------------------------------------------

modern/
    Current recommended versions

legacy/
    Historical development versions

docs/
    Reference documentation

screenshots/
    UI and example images

--------------------------------------------------------
MODERN VERSIONS
--------------------------------------------------------

FFMPEG EDITION
Uses external ffmpeg subprocess recording.

Best for:
- Maximum reliability
- Simpler dependencies
- Stable Linux systems

Requirements:
sudo apt install ffmpeg v4l-utils tesseract-ocr
pip install -r requirements-ffmpeg.txt

Run:
python3 modern/microcam_benchscope_ffmpeg.py

--------------------------------------------------------

PYAV EDITION
Uses PyAV internal FFmpeg bindings.

Best for:
- Cleaner Python-native recording
- Better future extensibility
- Advanced workflows

Requirements:
sudo apt install v4l-utils tesseract-ocr
pip install -r requirements-pyav.txt

Run:
python3 modern/microcam_benchscope_pyav.py

--------------------------------------------------------
LEGACY VERSIONS
--------------------------------------------------------

The legacy folder contains the original development history of the project.

These versions are:
- simpler
- easier to hack
- useful for experimentation
- educational references

Included:
- microcam_original.py
- microcam_v2.py
- microcam_v3.py
- microcam_hyperlab.py

--------------------------------------------------------
RECOMMENDED HDMI MICROSCOPE SETTINGS
--------------------------------------------------------

Resolution:
1920x1080

FPS:
30

FOURCC:
MJPG

Recommended filters:
CLAHE ON
Sharpen ON

--------------------------------------------------------
KEYBOARD SHORTCUTS
--------------------------------------------------------

W A S D
Pan

MouseWheel
Zoom

P
Snapshot

R
Record

SPACE
Pause

L
Lock measurement

C
Clear measurements

--------------------------------------------------------
RECOMMENDED HARDWARE
--------------------------------------------------------

- MacroSilicon USB HDMI capture cards
- HDMI digital microscopes
- USB microscopes
- Linux systems with V4L2 support

--------------------------------------------------------
NOTES
--------------------------------------------------------

This project is intentionally practical and repair-bench focused.

It prioritizes:
- realtime usability
- low-latency inspection
- easy modification
- Linux compatibility
- single-file experimentation
- practical electronics workflows
