<div align="center">

<img src="assets/vanish_icon.ico" alt="VanishCam icon" width="96">

# VanishCam

### Become invisible on webcam with anime-inspired effects.

![Platform](https://img.shields.io/badge/platform-Windows-0078D4?style=flat-square&logo=windows)
![Python](https://img.shields.io/badge/python-3.x-3776AB?style=flat-square&logo=python)
![Processing](https://img.shields.io/badge/camera%20processing-local-success?style=flat-square)

Use hand gestures to disappear from your live video feed with smoke, teleportation, and substitution effects.

[Quick Start](#quick-start) · [Effects](#effects) · [OBS Setup](#obs-and-virtual-camera) · [Troubleshooting](#troubleshooting)

</div>

## Why VanishCam?

- 🍃 **Naruto Vanish** — joined hands trigger smoke and a falling substitution log.
- ⚡ **Goku Vanish** — two fingers touching your forehead trigger a teleport effect.
- 🎥 **Streaming ready** — works with OBS Window Capture and Unity Capture.
- 🔒 **Private by design** — video is processed locally and never uploaded.

<p align="center">
  <img src="assets/substitution_log.png" alt="Naruto substitution log effect" width="420">
</p>

<details>
<summary>Contents</summary>

- [Quick Start](#quick-start)
- [Launch Modes](#launch-modes)
- [Effects](#effects)
- [Controls](#controls)
- [OBS and Virtual Camera](#obs-and-virtual-camera)
- [Troubleshooting](#troubleshooting)
- [Build the Windows Executable](#build-the-windows-executable)
- [Development](#development)

</details>

## Quick Start

This repository contains the source code, not a prebuilt `VanishCam.exe`.

### Requirements

- Windows 10 or later
- Python 3
- A working webcam
- Internet access on the first launch to install dependencies and download MediaPipe models

### Run from source

```powershell
git clone https://github.com/kunokun12/vanish_webcam_improved.git
cd vanish_webcam_improved
AVVIA_LOCALE.bat
```

Choose a camera, select **Naruto Vanish** or **Goku Vanish**, then follow the preview prompt: step out during the three-second countdown and keep the room empty for the four-second background capture.

> The first launch creates `.venv-local`, installs the packages from `requirements.txt`, and downloads the required models. Later launches reuse them.

If you already have a trusted prebuilt `VanishCam.exe`, Python is not required. Install [Unity Capture](https://github.com/schellingb/UnityCapture) first if you want virtual-camera output.

## Launch Modes

- `AVVIA_LOCALE.bat` — normal local preview window
- `AVVIA_OBS.bat` — clean preview for OBS Window Capture
- `AVVIA_VIRTUAL.bat` — output through Unity Capture
- `BUILD_WINDOWS.bat` — build `dist/VanishCam.exe`

You can skip the selectors by passing options:

```powershell
AVVIA_LOCALE.bat --mode naruto
AVVIA_LOCALE.bat --mode goku
AVVIA_LOCALE.bat --camera 1 --width 1280 --height 720 --fps 30
```

`--camera N` uses a specific webcam index. Omit it to show the camera selector.

## Effects

### 🍃 Naruto Vanish

1. Keep both hands visible.
2. Bring your palms together with your fingers side by side; do not interlace them.
3. The app replaces you with smoke and a falling wooden log.
4. Press `R` to reappear.

Effect sound: `assets/vanish.MP3`

### ⚡ Goku Vanish

1. Hold your index and middle fingers against your forehead for about 0.25 seconds.
2. You disappear with a teleport effect.
3. Lower your hand, then repeat the gesture to reappear.

The hand can face sideways. Good lighting and a clear view of the hand improve recognition.

Effect sound: `assets/goku-teleport-sound.mp3`

## Controls

**Camera**

- `SPACE` — capture a new background
- `F` — toggle fullscreen

**Effects**

- `V` — trigger the selected effect manually
- `R` — reappear
- `M` — return to the effect selector

**Debugging and exit**

- `G` — show or hide gesture diagnostics
- `Q` or `ESC` — quit

Gesture diagnostics appear in a separate debug window or in the terminal. They are never sent to the clean preview or virtual-camera output.

## OBS and Virtual Camera

### OBS Window Capture

1. Run `AVVIA_OBS.bat`.
2. In OBS, add a **Window Capture** source.
3. Select **Webcam - Preview**.
4. Start the OBS Virtual Camera if you need to send the scene to another application.

The preview contains the webcam and effects but no menus or debug markers. Press `G` to show diagnostics separately as **Webcam - Debug Preview**.

### Unity Capture

1. Install [Unity Capture](https://github.com/schellingb/UnityCapture).
2. Run `AVVIA_VIRTUAL.bat`.
3. Select **Unity Video Capture** in the receiving application.

Do not use the same virtual camera as both an OBS input and an OBS output. If a virtual camera is already running, stop it before launching another backend. See the [pyvirtualcam supported backends](https://github.com/letmaik/pyvirtualcam#supported-virtual-cameras) for more information.

## Troubleshooting

### No camera appears

Close applications that may already be using the webcam, check Windows camera permissions, and run the launcher again. You can also try a specific index, such as `--camera 1`.

### The virtual camera cannot start

Install Unity Capture or OBS Studio, make sure the corresponding virtual camera is available, and stop any other application currently using it. Use `AVVIA_OBS.bat` if you only need OBS Window Capture.

### Gestures are not recognized

Improve the lighting, keep the hands inside the frame, and use `G` to inspect recognition diagnostics. For Goku mode, hold the two fingers against the forehead briefly instead of moving through the pose quickly.

### The first launch is slow

This is expected. The launcher may create the virtual environment, install dependencies, and download the required MediaPipe models.

## Build the Windows Executable

Install Python 3, then run:

```powershell
BUILD_WINDOWS.bat
```

The script installs PyInstaller when needed, downloads missing models, builds the application, and runs its self-test. The result is written to `dist/VanishCam.exe`. The `build/` and `dist/` directories are generated artifacts and are not committed to the repository.

## Development

Install the dependencies in the local environment, then run the test suite with:

```powershell
.venv-local\Scripts\python.exe -m unittest discover -p "test_*.py"
```

The main application is in `vanish_cam.py`. Naruto effect rendering is in `naruto_effect.py`, and audio playback is in `vanish_sound.py`.

## Project Status

VanishCam is an experimental Windows webcam project. It works best with good lighting, a stable camera, and clearly visible hand gestures.
