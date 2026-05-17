========================================================
MICROCAM BENCHSCOPE HARDWARE GUIDE
========================================================

RECOMMENDED HDMI CAPTURE CARDS
--------------------------------------------------------

BEST LOW-COST OPTION
- MacroSilicon USB HDMI capture cards

Common chipsets:
- MS2109
- MS2130

Typical Linux support:
- Excellent
- UVC compatible
- V4L2 compatible

Usually appears as:
- USB Video
- MacroSilicon USB Video

--------------------------------------------------------
RECOMMENDED MICROSCOPE TYPES
--------------------------------------------------------

HDMI DIGITAL MICROSCOPES
Best option for:
- low latency
- high resolution
- PCB repair
- soldering

USB MICROSCOPES
Useful for:
- portability
- lower cost

Less ideal:
- usually higher latency
- lower image quality

--------------------------------------------------------
RECOMMENDED RESOLUTION
--------------------------------------------------------

1920x1080

Best balance:
- clarity
- performance
- latency

--------------------------------------------------------
RECOMMENDED FPS
--------------------------------------------------------

30 FPS

Higher FPS:
- increases CPU usage
- may increase USB bandwidth usage

--------------------------------------------------------
RECOMMENDED FOURCC
--------------------------------------------------------

MJPG

Usually lowest latency on cheap HDMI capture cards.

Fallback:
YUYV

--------------------------------------------------------
LIGHTING RECOMMENDATIONS
--------------------------------------------------------

Best results:
- diffuse LED lighting
- adjustable brightness
- ring lights
- side lighting

Avoid:
- harsh reflections
- direct glare
- uneven illumination

--------------------------------------------------------
PCB INSPECTION TIPS
--------------------------------------------------------

Enable:
- CLAHE
- Sharpen

Useful for:
- trace visibility
- solder joints
- chip markings

--------------------------------------------------------
OCR BEST PRACTICES
--------------------------------------------------------

For chip marking OCR:
- stable focus
- strong contrast
- bright lighting
- CLAHE enabled

--------------------------------------------------------
LATENCY REDUCTION
--------------------------------------------------------

Recommended:
- MJPG
- 1080p
- 30 FPS
- disable histogram
- disable unnecessary overlays

--------------------------------------------------------
RECOMMENDED LINUX ENVIRONMENT
--------------------------------------------------------

Best experience:
- X11/XCB
- modern Mesa drivers
- hardware acceleration enabled

Wayland may behave differently depending on distro.

--------------------------------------------------------
KNOWN GOOD USE CASES
--------------------------------------------------------

- PCB repair
- Solder inspection
- BGA inspection
- Trace analysis
- Chip identification
- Connector inspection
- Flux cleanup verification
- Fine-pitch solder work

