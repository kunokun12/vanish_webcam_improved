# VanishCam — webcam disappearance effects

Video is processed on your computer. It is neither recorded nor sent to a server. The preview window and virtual camera show only the video and effects, without overlaid menus, instructions, or text.

For step-by-step instructions in English and Italian, see the [user guide](USER_GUIDE_EN_IT.md).

## ☕ Buy me a coffee

If you like this project and want to support it, you can buy me a coffee through [PayPal](https://paypal.me/nicovic17) or [Revolut](https://revolut.me/nicoauriemma).

## Download for Windows

[Download VanishCam.exe]( https://nicoaauriemma.vercel.app/shop ) and double-click it to run the app. You do not need to install Python. The packaged EXE starts in **Unity Capture virtual-camera mode** by default; use `AVVIA_LOCALE.bat` for the source/local workflow.

The published EXE may predate the latest source changes. To use the current modes, run the local scripts or rebuild the EXE from these sources.

## Getting started

1. Double-click `AVVIA_LOCALE.bat`.
2. In the webcam selector, click the correct camera or press its number. This is useful when Windows exposes a built-in camera, OBS Virtual Camera, and other devices at different indices.
3. In the start menu, choose **Naruto Vanish** or **Goku Vanish** with the mouse or the `1` or `2` key.
4. Step out of the camera frame for 4 seconds while the background is captured. The terminal tells you when you can step back in.

**Goku:** Hold your index and middle fingers against your forehead for about 0.25 seconds. The hand may face sideways, as in the anime pose. The detector uses the last visible finger segments and a forehead contact zone, so exact fingertip separation is not required when the fingertips are occluded. Animated vertical comic-style lines run across your silhouette during the teleport effect. The supplied `assets/goku-teleport-sound.mp3` plays when you vanish or reappear. To reappear, lower your hand, then repeat the gesture. You can also use `V` to disappear and `R` to reappear manually. Press `G` to see the hand landmarks, forehead anchor, pose conditions, and hold timer.

The former Normal Vanish mode is currently hidden from the mode selector.

**Naruto Vanish:** Bring your palms together in front of the webcam, with your fingers side by side rather than interlaced. Keep both hands visible. As soon as the gesture is recognized, you vanish in a cloud of smoke and a wooden log appears, falls, and remains visible until you press `R`. No microphone is needed. Naruto mode uses `hand_landmarker.task` to track both hands. `assets/vanish.MP3` plays through your computer speakers when the disappearance starts, including when triggered with `V`. You can trigger it again after pressing `R`.

To start a mode without the menu:

```powershell
AVVIA_LOCALE.bat --mode naruto
AVVIA_LOCALE.bat --mode goku
```

## Virtual camera and OBS

If [OBS Studio](https://obsproject.com/) is installed, the app automatically tries to open **OBS Virtual Camera**. The terminal shows the device name. Use `AVVIA_VIRTUAL.bat` to start **Unity Capture** automatically. It shows the camera selector first, then the effect-mode selector, and uses the camera you choose. Select **Unity Video Capture** in the app that will receive the video.

To use the effect in an OBS scene, run `AVVIA_OBS.bat`, add a **Window Capture** source, and select **Webcam - Preview**. This is the clean OBS-facing preview: it contains the webcam and effects, but never diagnostic markers. The optional `G` gesture-debug view opens separately as **Webcam - Debug Preview**, so it cannot contaminate the OBS source. Then start the virtual camera from OBS. Alternatively, use [Unity Capture](https://github.com/schellingb/UnityCapture) as a separate virtual video source:

```powershell
AVVIA_VIRTUAL.bat
```

You can skip either selector when needed, for example `AVVIA_VIRTUAL.bat --mode goku` or `AVVIA_VIRTUAL.bat --camera 1 --mode goku`.

OBS has only one virtual camera. Do not use it as both an OBS input and output at the same time. See the [pyvirtualcam documentation](https://github.com/letmaik/pyvirtualcam#supported-virtual-cameras).

## Controls

- `SPACE`: capture a new background.
- `R`: make the person reappear.
- `V`: manually start the selected mode's effect.
- `G`: toggle the hand-gesture debug overlay (landmarks and detection status).
- `M`: return to the mode-selection menu.
- `F`: toggle fullscreen.
- `Q` or `ESC`: quit and release the cameras.

In Goku mode, `V` starts the disappearance manually and `R` starts reappearance when hidden.

You can change the camera, resolution, and maximum virtual-output FPS:

```powershell
AVVIA_LOCALE.bat --camera 1 --width 1280 --height 720 --fps 30
```

Passing `--camera N` skips the webcam selector. If the selector is closed, press `Q` or `Esc` to quit.

To troubleshoot gesture recognition without adding text to the video, pass `--gesture-debug`. Diagnostic information appears only in the terminal.

## Visual assets

The [wooden log](assets/substitution_log.png) is an original illustration generated with ImageGen using the supplied image as a visual reference. Smoke is drawn in real time by `naruto_effect.py` so it follows the person's position and scales with the video. We also considered a [CC0 smoke sprite sheet](https://opengameart.org/content/smoke-sprite-sheet), but the procedural animation adapts better to dynamic positioning without an additional asset. No anime frames or assets are included in the project.

Goku mode uses `face_landmarker.task` to estimate the forehead position and `hand_landmarker.task` to track the fingers. The teleport's visual trace is drawn in code, without anime frames.
