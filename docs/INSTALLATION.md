========================================================
MICROCAM BENCHSCOPE INSTALLATION
========================================================

SUPPORTED SYSTEMS
--------------------------------------------------------

Primary target:
- Linux
- Ubuntu
- Debian
- Linux Mint
- Fedora
- Arch Linux

Recommended:
Ubuntu 24.04 LTS or newer

--------------------------------------------------------
SYSTEM PACKAGES
--------------------------------------------------------

Ubuntu/Debian:
sudo apt install \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    v4l-utils \
    tesseract-ocr

--------------------------------------------------------
CREATE VIRTUAL ENVIRONMENT
--------------------------------------------------------

python3 -m venv venv

source venv/bin/activate

--------------------------------------------------------
INSTALL FFMPEG EDITION
--------------------------------------------------------

pip install -r requirements-ffmpeg.txt

Run:
python3 modern/microcam_benchscope_ffmpeg.py

--------------------------------------------------------
INSTALL PYAV EDITION
--------------------------------------------------------

pip install -r requirements-pyav.txt

Run:
python3 modern/microcam_benchscope_pyav.py

--------------------------------------------------------
VERIFY VIDEO DEVICES
--------------------------------------------------------

Check:
ls /dev/video*

List devices:
v4l2-ctl --list-devices

--------------------------------------------------------
VERIFY HDMI CAPTURE
--------------------------------------------------------

Test with ffplay:
ffplay /dev/video0

Or:
ffplay -f v4l2 /dev/video0

--------------------------------------------------------
RECOMMENDED CAPTURE SETTINGS
--------------------------------------------------------

Resolution:
1920x1080

FPS:
30

FOURCC:
MJPG

--------------------------------------------------------
OPTIONAL OCR SUPPORT
--------------------------------------------------------

Install:
sudo apt install tesseract-ocr

Verify:
tesseract --version

--------------------------------------------------------
OPTIONAL DEVELOPMENT TOOLS
--------------------------------------------------------

Useful packages:
sudo apt install git htop neovim

--------------------------------------------------------
GPU ACCELERATION NOTES
--------------------------------------------------------

Current versions are primarily CPU-based.

Future versions may include:
- CUDA
- OpenGL
- Vulkan
- VAAPI
- NVENC

--------------------------------------------------------
KNOWN GOOD HARDWARE
--------------------------------------------------------

- MacroSilicon HDMI USB capture cards
- Generic HDMI microscope cameras
- USB microscopes with V4L2 support

--------------------------------------------------------
NOTES
--------------------------------------------------------

PyAV version:
- cleaner architecture
- more advanced recording path

FFmpeg version:
- simpler dependency model
- easier troubleshooting
- maximum compatibility
