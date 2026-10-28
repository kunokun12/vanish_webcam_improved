# VanishCam

VanishCam is a Windows webcam app that makes you disappear from a live video feed. It supports two effects:

- **Naruto Vanish** — joined hands trigger smoke and a substitution log.
- **Goku Vanish** — two fingers touching your forehead trigger a teleport effect.

Video is processed locally. VanishCam does not record or upload your camera feed. The first source-code launch may download Python packages and MediaPipe models.

## Quick start: Windows

This repository currently contains the source code, not a prebuilt `VanishCam.exe`. The recommended way to run the latest version is to [run it from source](#run-from-source):

```powershell
git clone https://github.com/kunokun12/vanish_webcam_improved.git
cd vanish_webcam_improved
AVVIA_LOCALE.bat
```

If you already have a trusted prebuilt `VanishCam.exe`, Python is not required. The packaged app is configured to use [Unity Capture](https://github.com/schellingb/UnityCapture) for virtual-camera output by default, so install Unity Capture before launching it. To build your own executable, use `BUILD_WINDOWS.bat` as described [below](#build-the-windows-executable).

## Run from source

### Requirements

- Windows 10 or later
- Python 3
- A working webcam
- Internet access on the first launch so the launcher can install dependencies and download models

Clone the repository and start the local preview:

```powershell
git clone https://github.com/kunokun12/vanish_webcam_improved.git
cd vanish_webcam_improved
AVVIA_LOCALE.bat
```

`AVVIA_LOCALE.bat` creates `.venv-local` automatically and installs the packages from `requirements.txt`. The app then opens a camera selector followed by an effect selector. Choose a camera, choose **Naruto Vanish** or **Goku Vanish**, and step out of frame for about four seconds while the background is captured.

The required MediaPipe models are downloaded into `models/` the first time each effect is used. The models and generated build files are intentionally excluded from Git.

## Choose a launch mode

- `AVVIA_LOCALE.bat` — normal local preview window.
- `AVVIA_OBS.bat` — clean preview for OBS Window Capture; virtual-camera output is disabled.
- `AVVIA_VIRTUAL.bat` — send the result through Unity Capture.
- `BUILD_WINDOWS.bat` — build a new `dist/VanishCam.exe`.

You can skip the selectors by passing options:

```powershell
AVVIA_LOCALE.bat --mode naruto
AVVIA_LOCALE.bat --mode goku
AVVIA_LOCALE.bat --camera 1 --width 1280 --height 720 --fps 30
```

`--camera N` uses a specific webcam index. Omit it to show the camera selector.

## Effects and gestures

### Naruto Vanish

1. Keep both hands visible.
2. Bring your palms together with your fingers side by side; do not interlace them.
3. The app replaces you with smoke and a falling wooden log.
4. Press `R` to reappear.

The effect sound is `assets/vanish.MP3`.

### Goku Vanish

1. Hold your index and middle fingers against your forehead for about 0.25 seconds.
2. You disappear with a teleport effect.
3. Lower your hand, then repeat the gesture to reappear.

The effect sound is `assets/goku-teleport-sound.mp3`. The hand can face sideways; good lighting and a clear view of the hand improve recognition.

## Controls

These keys work while the webcam window is active:

- `SPACE` — capture a new background
- `R` — reappear
- `V` — trigger the selected effect manually
- `G` — show or hide gesture diagnostics
- `M` — return to the effect selector
- `F` — toggle fullscreen
- `Q` or `ESC` — quit

Gesture diagnostics are shown in a separate debug window or in the terminal. They are not sent to the clean preview or virtual-camera output.

## OBS and virtual-camera setup

### OBS Window Capture

1. Run `AVVIA_OBS.bat`.
2. In OBS, add a **Window Capture** source.
3. Select **Webcam - Preview**.
4. Start the OBS Virtual Camera if you need to send the scene to another application.

The preview contains the webcam and effects but no menus or debug markers. If you press `G`, diagnostics appear separately as **Webcam - Debug Preview**.

### Unity Capture

1. Install Unity Capture.
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

This is expected. The launcher may create the virtual environment, install dependencies, and download the required MediaPipe models. Later launches reuse those files.

## Build the Windows executable

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
