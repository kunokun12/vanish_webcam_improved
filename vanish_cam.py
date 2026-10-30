"""Local webcam: disappearance effect triggered by an index-and-middle-finger V gesture."""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from contextlib import ExitStack
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    from .naruto_effect import (
        EFFECT_SECONDS, joined_hands_from_landmarks, landed_log_anchor,
        load_log, overlay_log, render_substitution, substitution_anchor,
    )
    from .vanish_sound import GOKU_SOUND_PATH, SOUND_PATH, VanishSound
except ImportError:
    from naruto_effect import (
        EFFECT_SECONDS, joined_hands_from_landmarks, landed_log_anchor,
        load_log, overlay_log, render_substitution, substitution_anchor,
    )
    from vanish_sound import GOKU_SOUND_PATH, SOUND_PATH, VanishSound


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
WINDOW_NAME = "Webcam - Preview"
DEBUG_WINDOW_NAME = "Webcam - Debug Preview"

SEGMENTER_MODEL = MODEL_DIR / "selfie_segmenter_landscape.tflite"
GESTURE_MODEL = MODEL_DIR / "gesture_recognizer.task"
HAND_MODEL = APP_DIR / "hand_landmarker.task"
FACE_MODEL = APP_DIR / "face_landmarker.task"

MODEL_URLS = {
    SEGMENTER_MODEL: (
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
        "selfie_segmenter_landscape/float16/latest/"
        "selfie_segmenter_landscape.tflite"
    ),
    GESTURE_MODEL: (
        "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
        "gesture_recognizer/float16/1/gesture_recognizer.task"
    ),
}

CALIBRATION_SECONDS = 4.0
SETUP_COUNTDOWN_SECONDS = 3.0
BACKGROUND_READY_NOTICE_SECONDS = 2.0
BACKGROUND_SAMPLE_SECONDS = 0.55
VANISH_SECONDS = 3.6
GOKU_EFFECT_SECONDS = 1.15
GOKU_HOLD_SECONDS = 0.25
GESTURE_INTERVAL_SECONDS = 0.14


@dataclass
class AppState:
    phase: str = "setup_countdown"
    calibration_started: float = 0.0
    vanish_started: float | None = None
    last_gesture_check: float = 0.0
    fullscreen: bool = False


def draw_background_setup_overlay(
    frame: np.ndarray,
    phase: str,
    remaining: int = 0,
    progress: float = 0.0,
) -> np.ndarray:
    """Draw setup guidance on the local preview without changing its source frame."""
    preview = frame.copy()
    height, width = preview.shape[:2]
    panel_width = min(620, width - 32)
    panel_height = 172 if phase != "ready" else 118
    left = max(16, (width - panel_width) // 2)
    top = max(16, (height - panel_height) // 2)
    right, bottom = left + panel_width, top + panel_height

    panel = preview.copy()
    cv2.rectangle(panel, (left, top), (right, bottom), (20, 25, 32), -1)
    cv2.addWeighted(panel, 0.84, preview, 0.16, 0, preview)
    cv2.rectangle(preview, (left, top), (right, bottom), (72, 91, 101), 1)

    if phase == "setup_countdown":
        heading, detail = "STEP OUT OF VIEW", "Background capture starts in"
        accent = (112, 224, 191)
        cv2.putText(preview, str(max(1, remaining)), (right - 72, top + 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.7, accent, 3, cv2.LINE_AA)
    elif phase == "calibrating":
        heading, detail = "CAPTURING BACKGROUND", "Keep the room empty and camera still"
        accent = (112, 224, 191)
    else:
        heading, detail = "BACKGROUND READY", "Step back in  |  Space to recapture"
        accent = (112, 224, 191)

    cv2.putText(preview, heading, (left + 24, top + 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (242, 246, 248), 2, cv2.LINE_AA)
    cv2.putText(preview, detail, (left + 24, top + 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (190, 203, 210), 1, cv2.LINE_AA)

    if phase == "calibrating":
        bar_left, bar_top, bar_right = left + 24, top + 112, right - 24
        cv2.rectangle(preview, (bar_left, bar_top), (bar_right, bar_top + 8), (58, 68, 76), -1)
        fill_right = bar_left + int((bar_right - bar_left) * min(1.0, max(0.0, progress)))
        if fill_right > bar_left:
            cv2.rectangle(preview, (bar_left, bar_top), (fill_right, bar_top + 8), accent, -1)
        cv2.putText(preview, f"{max(0, remaining)} sec", (bar_left, top + 148),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (190, 203, 210), 1, cv2.LINE_AA)
    return preview


@dataclass
class GokuPoseResult:
    valid: bool = False
    score: float = 0.0
    hold_seconds: float = 0.0
    trigger: bool = False
    forehead: tuple[float, float] | None = None
    hand_landmarks: list = field(default_factory=list)
    face_landmarks: list = field(default_factory=list)
    details: str = "waiting for hand and face"


def positive_fps(value: str) -> float:
    fps = float(value)
    if not np.isfinite(fps) or not 1 <= fps <= 120:
        raise argparse.ArgumentTypeError("FPS must be between 1 and 120.")
    return fps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Choose Normal Vanish or Naruto Vanish for the webcam."
    )
    parser.add_argument(
        "--camera", type=int, default=None,
        help="Webcam index (skip camera selection; for example --camera 1)",
    )
    parser.add_argument("--width", type=int, default=960, help="Video width")
    parser.add_argument("--height", type=int, default=540, help="Video height")
    parser.add_argument(
        "--fullscreen", action="store_true", help="Start in fullscreen"
    )
    parser.add_argument(
        "--no-mirror", action="store_true", help="Do not mirror the webcam"
    )
    parser.add_argument("--fps", type=positive_fps, default=30.0, help="Maximum virtual-output FPS (default: 30)")
    parser.add_argument(
        "--virtual-camera", action=argparse.BooleanOptionalAction, default=None,
        help="Require the virtual webcam; enabled by default when available. --no-virtual-camera disables it",
    )
    parser.add_argument(
        "--virtual-backend", choices=("obs", "unitycapture", "v4l2loopback"),
        help="Virtual webcam driver to use",
    )
    parser.add_argument("--virtual-device", help="Name of the virtual device to use")
    parser.add_argument(
        "--mode", choices=("normal", "naruto", "goku"),
        help="Skip the start menu and launch the selected mode directly",
    )
    parser.add_argument(
        "--gesture-debug", action="store_true",
        help="Print recognition state in the terminal (without text in the video)",
    )
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    # The packaged Windows app is intended to feed OBS/Unity Capture when it
    # is double-clicked.  Source launches keep the flexible automatic default,
    # while explicit command-line options always take precedence.
    if (
        getattr(sys, "frozen", False)
        and args.virtual_camera is not False
        and args.virtual_backend is None
        and args.virtual_device is None
    ):
        args.virtual_camera = True
        args.virtual_backend = "unitycapture"
    return args


def available_cameras(max_index: int = 8) -> list[tuple[int, np.ndarray]]:
    """Return camera indices that open and provide at least one frame."""
    cameras = []
    for index in range(max_index):
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW) if platform.system() == "Windows" else cv2.VideoCapture(index)
        if not capture.isOpened():
            capture.release()
            continue
        ok, frame = capture.read()
        capture.release()
        if ok and frame is not None and frame.size:
            cameras.append((index, frame))
    return cameras


def choose_camera() -> int | None:
    """Show detected cameras and let the user choose one with mouse or keyboard."""
    cameras = available_cameras()
    if not cameras:
        raise RuntimeError("No webcam was detected. Check your camera connection and permissions.")

    title = "Choose webcam"
    selected = [None]
    card_width, card_height = 380, 230
    columns = 2
    rows = (len(cameras) + columns - 1) // columns
    window = np.zeros((110 + rows * card_height, columns * card_width, 3), dtype=np.uint8)
    window[:] = (26, 29, 34)
    cv2.putText(window, "CHOOSE YOUR WEBCAM", (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (236, 239, 242), 2, cv2.LINE_AA)
    cv2.putText(window, "Click a camera or press its number", (32, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (179, 195, 201), 1, cv2.LINE_AA)

    cards = []
    for position, (index, frame) in enumerate(cameras):
        column, row = position % columns, position // columns
        left, top = column * card_width + 20, 100 + row * card_height
        right, bottom = left + card_width - 40, top + card_height - 30
        cards.append((index, left, top, right, bottom, position + 1))
        cv2.rectangle(window, (left, top), (right, bottom), (48, 54, 62), -1)
        preview = cv2.resize(frame, (right - left - 20, bottom - top - 55), interpolation=cv2.INTER_AREA)
        window[top + 10:bottom - 45, left + 10:right - 10] = preview
        cv2.putText(window, f"{position + 1}: Camera {index}", (left + 12, bottom - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 245, 247), 1, cv2.LINE_AA)

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONUP:
            for index, left, top, right, bottom, _ in cards:
                if left <= x <= right and top <= y <= bottom:
                    selected[0] = index

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, on_mouse)
    try:
        while selected[0] is None:
            cv2.imshow(title, window)
            key = cv2.waitKey(30) & 0xFF
            if ord("1") <= key <= ord("9"):
                position = key - ord("1")
                if position < len(cameras):
                    selected[0] = cameras[position][0]
            elif key in (ord("q"), 27):
                return None
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                return None
        return selected[0]
    finally:
        cv2.destroyWindow(title)


def choose_mode() -> str | None:
    """Simple keyboard/mouse menu shown before the clean camera feed."""
    title = "Choose effect"
    selected = [None]
    window = np.zeros((540, 700, 3), dtype=np.uint8)
    window[:] = (26, 29, 34)
    cv2.putText(window, "CHOOSE YOUR VANISH", (68, 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (236, 239, 242), 2, cv2.LINE_AA)
    buttons = (
        ("naruto", "1  NARUTO VANISH", "Joined hands  /  log and smoke", 140),
        ("goku", "2  GOKU VANISH", "Two fingers to forehead  /  teleport", 260),
    )
    for mode, heading, detail, top in buttons:
        cv2.rectangle(window, (65, top), (635, top + 94), (48, 54, 62), -1, cv2.LINE_AA)
        cv2.rectangle(window, (65, top), (635, top + 94), (111, 142, 153), 2, cv2.LINE_AA)
        cv2.putText(window, heading, (90, top + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.77, (240, 245, 247), 2, cv2.LINE_AA)
        cv2.putText(window, detail, (90, top + 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.53, (179, 195, 201), 1, cv2.LINE_AA)

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONUP and 65 <= x <= 635:
            for mode, _, _, top in buttons:
                if top <= y <= top + 94:
                    selected[0] = mode

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, on_mouse)
    try:
        while selected[0] is None:
            cv2.imshow(title, window)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("1"):
                selected[0] = "naruto"
            elif key == ord("2"):
                selected[0] = "goku"
            elif key in (ord("q"), 27):
                return None
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                return None
        return selected[0]
    finally:
        cv2.destroyWindow(title)


def open_virtual_camera(args: argparse.Namespace, width: int, height: int):
    if args.virtual_camera is False:
        return None
    try:
        import pyvirtualcam

        camera = pyvirtualcam.Camera(
            width=width, height=height, fps=args.fps,
            fmt=pyvirtualcam.PixelFormat.BGR,
            backend=args.virtual_backend, device=args.virtual_device,
        )
    except (ImportError, RuntimeError, OSError, ValueError) as error:
        message = (
            "Virtual webcam unavailable. Install OBS Studio (with the OBS Virtual "
            "Camera driver) or Unity Capture. If you use OBS Virtual Camera, stop "
            "'Start Virtual Camera' in OBS. Launch AVVIA_LOCALE.bat to install "
            f"pyvirtualcam. Details: {error}"
        )
        if args.virtual_camera is True or args.virtual_backend or args.virtual_device:
            raise RuntimeError(message) from error
        print(f"[Webcam] {message}\n[Webcam] Continuing with the local preview.")
        return None
    print(f"[Webcam] Virtual output active: {camera.device} ({width}x{height}, {args.fps:g} maximum FPS)")
    return camera


def send_virtual_frame(camera, frame: np.ndarray) -> None:
    if frame.shape[:2] != (camera.height, camera.width):
        frame = cv2.resize(frame, (camera.width, camera.height), interpolation=cv2.INTER_LINEAR)
    camera.send(np.ascontiguousarray(frame))


def download_models(mode: str = "normal") -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for destination, url in MODEL_URLS.items():
        if mode in ("naruto", "goku") and destination == GESTURE_MODEL:
            continue
        if destination.exists() and destination.stat().st_size > 100_000:
            continue

        temporary = destination.with_suffix(destination.suffix + ".part")
        print(f"[Vanish Lab] Downloading {destination.name}...")
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "Vanish-Lab/1.0"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                total = int(response.headers.get("Content-Length", 0))
                received = 0
                with temporary.open("wb") as output:
                    while chunk := response.read(1024 * 256):
                        output.write(chunk)
                        received += len(chunk)
                        if total:
                            percent = int(received * 100 / total)
                            print(f"\r  {percent:3d}%", end="", flush=True)
            print()
            os.replace(temporary, destination)
        except (OSError, urllib.error.URLError) as error:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                "Could not download the MediaPipe models. "
                "Check your internet connection and try again."
            ) from error


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    if platform.system() == "Windows":
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(index)
    else:
        capture = cv2.VideoCapture(index)

    if not capture.isOpened():
        raise RuntimeError(
            f"Webcam {index} is unavailable. Close other apps or try --camera 1."
        )

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def _finger_is_extended(points: np.ndarray, base: int) -> tuple[bool, float]:
    wrist = points[0]
    finger = points[base : base + 4]
    path = float(np.linalg.norm(np.diff(finger, axis=0), axis=1).sum())
    if path < 1e-6:
        return False, 0.0
    straightness = float(np.linalg.norm(finger[3] - finger[0]) / path)
    reach = float(np.linalg.norm(finger[3] - wrist) /
                  max(np.linalg.norm(finger[0] - wrist), 1e-6))
    score = min(straightness / 0.80, reach / 1.12, 1.0)
    return straightness >= 0.72 and reach >= 1.05, float(np.clip(score, 0.0, 1.0))


def _finger_is_folded(points: np.ndarray, base: int) -> bool:
    wrist = points[0]
    finger = points[base : base + 4]
    path = float(np.linalg.norm(np.diff(finger, axis=0), axis=1).sum())
    straightness = float(np.linalg.norm(finger[3] - finger[0]) / max(path, 1e-6))
    reach = float(np.linalg.norm(finger[3] - wrist) /
                  max(np.linalg.norm(finger[0] - wrist), 1e-6))
    return straightness < 0.82 or reach < 1.10


def _point_to_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    length_squared = float(np.dot(segment, segment))
    if length_squared < 1e-8:
        return float(np.linalg.norm(point - start))
    amount = float(np.clip(np.dot(point - start, segment) / length_squared, 0.0, 1.0))
    projection = start + amount * segment
    return float(np.linalg.norm(point - projection))


def forehead_anchor(face) -> tuple[float, float] | None:
    """Return a stable forehead-center anchor from face geometry."""
    if len(face) < 11:
        return None
    face_points = np.array([(landmark.x, landmark.y) for landmark in face], dtype=np.float32)
    if not np.isfinite(face_points).all():
        return None

    face_left, face_top = face_points.min(axis=0)
    face_right, face_bottom = face_points.max(axis=0)
    face_height = max(float(face_bottom - face_top), 1e-3)

    # Use eye landmarks when available. A profile face has an asymmetric
    # bounding box, but the eye line remains a much better facial centerline.
    if len(face) > 362:
        eye_indices = (33, 133, 263, 362)
        eye_points = face_points[list(eye_indices)]
        eye_center = eye_points.mean(axis=0)
        return (
            float(eye_center[0]),
            float(eye_center[1] - face_height * 0.22),
        )

    # Fallback for reduced test models or unusual face-landmark outputs.
    return (
        float((face_left + face_right) * 0.5),
        float(face_top + face_height * 0.22),
    )


def evaluate_goku_pose(
    hand,
    face,
    forehead_override: tuple[float, float] | None = None,
) -> tuple[bool, float, tuple[float, float] | None, str]:
    """Evaluate the pose without relying on accurate fingertip landmarks.

    MediaPipe can lose the fingertips when they touch or occlude the forehead,
    so contact is measured against the last visible finger segments instead.
    """
    if len(face) < 11:
        return False, 0.0, None, "face not fully detected"

    face_points = np.array([(landmark.x, landmark.y) for landmark in face], dtype=np.float32)
    if not np.isfinite(face_points).all():
        return False, 0.0, None, "invalid face landmark coordinates"

    face_left, face_top = face_points.min(axis=0)
    face_right, face_bottom = face_points.max(axis=0)
    face_width = max(float(face_right - face_left), 1e-3)
    face_height = max(float(face_bottom - face_top), 1e-3)
    forehead = forehead_override or forehead_anchor(face)
    if forehead is None:
        return False, 0.0, None, "face anchor unavailable"

    if len(hand) != 21:
        return False, 0.0, forehead, "hand not fully detected"

    points = np.array([(landmark.x, landmark.y, landmark.z) for landmark in hand], dtype=np.float32)
    if not np.isfinite(points).all():
        return False, 0.0, forehead, "invalid hand landmark coordinates"

    index_ok, index_score = _finger_is_extended(points, 5)
    middle_ok, middle_score = _finger_is_extended(points, 9)
    ring_folded = _finger_is_folded(points, 13)
    pinky_folded = _finger_is_folded(points, 17)

    # The DIP-to-tip and PIP-to-tip segments remain useful when the actual
    # fingertips are hidden by contact with the forehead.
    forehead_point = np.array(forehead, dtype=np.float32)
    contact_distance = min(
        _point_to_segment_distance(forehead_point, points[start, :2], points[end, :2])
        for start, end in ((5, 7), (6, 8), (9, 11), (10, 12))
    )
    forehead_limit = max(face_width * 0.24, face_height * 0.09)
    contact_score = float(np.clip(1.0 - contact_distance / forehead_limit, 0.0, 1.0))
    near_forehead = contact_distance <= forehead_limit

    # One extension test may fail when the contacting finger is occluded. The
    # other finger plus the folded fingers still identify the pose.
    finger_evidence = index_ok or middle_ok
    finger_score = max(index_score, middle_score)
    fingertip_center = (points[7] + points[11]) * 0.5
    direction = fingertip_center - points[0]
    vertical_score = float(np.clip(abs(float(direction[1])) /
                                   max(abs(float(direction[0])) * 0.8, 1e-3), 0.0, 1.0))
    fingers_vertical = abs(float(direction[1])) >= abs(float(direction[0])) * 0.8

    valid = (
        finger_evidence and ring_folded and pinky_folded
        and near_forehead and fingers_vertical
    )
    score = float(min(finger_score, contact_score, vertical_score))
    details = (
        f"index={'OK' if index_ok else 'NO'} middle={'OK' if middle_ok else 'NO'} "
        f"contact={'NEAR' if near_forehead else 'FAR'} tips=not required "
        f"vertical={'OK' if fingers_vertical else 'NO'}"
    )
    return valid, score, forehead, details


class GokuPoseDetector:
    """Detect and time the side-facing two-fingers-to-forehead pose."""

    def __init__(self, hand_detector, face_detector):
        self.hand_detector = hand_detector
        self.face_detector = face_detector
        self.pose_started_ms: int | None = None
        self.trigger_consumed = False
        self.smoothed_forehead: tuple[float, float] | None = None
        self.face_missing_frames = 0
        self.last_result = GokuPoseResult()

    def detect(self, image: mp.Image, timestamp_ms: int) -> GokuPoseResult:
        hand_result = self.hand_detector.detect_for_video(image, timestamp_ms)
        face_result = self.face_detector.detect_for_video(image, timestamp_ms)
        hands = hand_result.hand_landmarks[0] if hand_result.hand_landmarks else []
        faces = face_result.face_landmarks[0] if face_result.face_landmarks else []
        raw_forehead = forehead_anchor(faces)
        if raw_forehead is not None:
            self.face_missing_frames = 0
            if self.smoothed_forehead is None:
                self.smoothed_forehead = raw_forehead
            else:
                smoothing = 0.25
                self.smoothed_forehead = tuple(
                    (1.0 - smoothing) * old + smoothing * new
                    for old, new in zip(self.smoothed_forehead, raw_forehead)
                )
        else:
            self.face_missing_frames += 1
            if self.face_missing_frames > 8:
                self.smoothed_forehead = None

        valid, score, _, details = evaluate_goku_pose(
            hands, faces, forehead_override=self.smoothed_forehead
        )

        if valid:
            if self.pose_started_ms is None:
                self.pose_started_ms = timestamp_ms
            hold_seconds = max(0.0, (timestamp_ms - self.pose_started_ms) / 1000.0)
            trigger = hold_seconds >= GOKU_HOLD_SECONDS and not self.trigger_consumed
            if trigger:
                self.trigger_consumed = True
        else:
            self.pose_started_ms = None
            self.trigger_consumed = False
            hold_seconds = 0.0
            trigger = False

        self.last_result = GokuPoseResult(
            valid=valid,
            score=score,
            hold_seconds=hold_seconds,
            trigger=trigger,
            forehead=self.smoothed_forehead,
            hand_landmarks=hand_result.hand_landmarks,
            face_landmarks=face_result.face_landmarks,
            details=details,
        )
        return self.last_result

    def close(self) -> None:
        try:
            self.hand_detector.close()
        finally:
            self.face_detector.close()


def create_tasks(mode: str = "normal"):
    segmenter_options = vision.ImageSegmenterOptions(
        base_options=python.BaseOptions(model_asset_path=str(SEGMENTER_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        output_confidence_masks=True,
        output_category_mask=False,
    )
    if mode == "goku":
        if not HAND_MODEL.is_file():
            raise RuntimeError(f"Hand model not found: {HAND_MODEL}")
        if not FACE_MODEL.is_file():
            raise RuntimeError(f"Face model not found: {FACE_MODEL}")
        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.45,
        )
        face_options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(FACE_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.35,
            min_face_presence_confidence=0.35,
            min_tracking_confidence=0.45,
        )
        segmenter = vision.ImageSegmenter.create_from_options(segmenter_options)
        try:
            hand_detector = vision.HandLandmarker.create_from_options(hand_options)
            try:
                face_detector = vision.FaceLandmarker.create_from_options(face_options)
            except Exception:
                hand_detector.close()
                raise
        except Exception:
            segmenter.close()
            raise
        return segmenter, GokuPoseDetector(hand_detector, face_detector)

    if mode == "naruto":
        if not HAND_MODEL.is_file():
            raise RuntimeError(f"Hand model not found: {HAND_MODEL}")
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.45,
        )
        task_class = vision.HandLandmarker
        wrapper_class = MultiViewHandLandmarker
    else:
        classifier_options = mp.tasks.components.processors.ClassifierOptions(
            score_threshold=0.55,
            category_allowlist=["Victory"],
        )
        options = vision.GestureRecognizerOptions(
            base_options=python.BaseOptions(model_asset_path=str(GESTURE_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.5,
            canned_gesture_classifier_options=classifier_options,
        )
        task_class = vision.GestureRecognizer
        wrapper_class = MultiViewGestureRecognizer

    segmenter = vision.ImageSegmenter.create_from_options(segmenter_options)
    try:
        primary = task_class.create_from_options(options)
        try:
            alternate = task_class.create_from_options(
                replace(options, running_mode=vision.RunningMode.IMAGE)
            )
        except Exception:
            primary.close()
            raise
    except Exception:
        segmenter.close()
        raise
    return segmenter, wrapper_class(primary, alternate)


def smoothstep(edge0: float, edge1: float, values: np.ndarray) -> np.ndarray:
    values = np.clip((values - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def ease_in_out_cubic(value: float) -> float:
    if value < 0.5:
        return 4.0 * value * value * value
    return 1.0 - ((-2.0 * value + 2.0) ** 3) / 2.0


def make_dissolve_fields(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7391)
    y, x = np.mgrid[0:height, 0:width]
    y_normal = y.astype(np.float32) / max(height - 1, 1)
    x_normal = x.astype(np.float32) / max(width - 1, 1)
    low_noise = rng.random((max(2, height // 10), max(2, width // 10)), dtype=np.float32)
    low_noise = cv2.resize(low_noise, (width, height), interpolation=cv2.INTER_CUBIC)
    fine_noise = rng.random((height, width), dtype=np.float32)
    field = (
        (1.0 - y_normal) * 0.72
        + low_noise * 0.17
        + fine_noise * 0.08
        + np.sin(x_normal * 18.0) * 0.025
    )
    return np.clip(field, 0.0, 1.0), fine_noise


def person_mask(
    segmenter: vision.ImageSegmenter,
    image: mp.Image,
    timestamp_ms: int,
    width: int,
    height: int,
) -> np.ndarray:
    result = segmenter.segment_for_video(image, timestamp_ms)
    if not result.confidence_masks:
        return np.zeros((height, width), dtype=np.float32)
    raw = np.squeeze(result.confidence_masks[0].numpy_view()).astype(np.float32)
    resized = cv2.resize(raw, (width, height), interpolation=cv2.INTER_LINEAR)
    resized = cv2.GaussianBlur(resized, (0, 0), sigmaX=2.2, sigmaY=2.2)
    return np.clip((resized - 0.12) / 0.68, 0.0, 1.0)


def alpha_composite(person: np.ndarray, background: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha_3 = alpha[:, :, None]
    return (
        person.astype(np.float32) * alpha_3
        + background.astype(np.float32) * (1.0 - alpha_3)
    ).astype(np.uint8)


def apply_dissolve(
    frame: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
    progress: float,
    field: np.ndarray,
    sparkle_noise: np.ndarray,
) -> np.ndarray:
    if progress <= 0.002:
        return alpha_composite(frame, background, mask)
    if progress >= 0.999:
        return background.copy()

    points = cv2.findNonZero((mask > 0.01).astype(np.uint8))
    if points is None:
        return background.copy()
    x, y, width, height = cv2.boundingRect(points)
    left, top = max(0, x - 18), max(0, y - 18)
    right = min(mask.shape[1], x + width + 18)
    bottom = min(mask.shape[0], y + height + 18)
    area = np.s_[top:bottom, left:right]
    mask = mask[area]
    field = field[area]
    sparkle_noise = sparkle_noise[area]
    eased = ease_in_out_cubic(progress)
    survival = smoothstep(eased - 0.055, eased + 0.055, field)
    alpha = mask * survival
    output = alpha_composite(frame[area], background[area], alpha)
    edge = (
        (np.abs(field - eased) < 0.018)
        & (mask > 0.48)
        & (sparkle_noise > 0.54)
    ).astype(np.float32)
    glow = cv2.GaussianBlur(edge, (0, 0), sigmaX=5.0, sigmaY=5.0)
    glow = np.clip(glow * 5.0 + edge, 0.0, 1.0)[:, :, None]
    cyan = np.full_like(output, (255, 228, 92), dtype=np.uint8)
    result = np.clip(
        output.astype(np.float32) * (1.0 - glow * 0.8)
        + cyan.astype(np.float32) * glow * 0.8,
        0,
        255,
    ).astype(np.uint8)
    canvas = background.copy()
    canvas[area] = result
    return canvas


def apply_goku_effect(
    frame: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
    progress: float,
    field: np.ndarray,
    sparkle_noise: np.ndarray,
    disappearing: bool,
) -> np.ndarray:
    """Apply a reversible teleport dissolve with vertical comic-style traces."""
    dissolve_progress = progress if disappearing else 1.0 - progress
    output = apply_dissolve(
        frame, background, mask, dissolve_progress, field, sparkle_noise
    )
    height, width = output.shape[:2]
    spacing = max(22, width // 28)
    offset = int(progress * spacing * 2.0) % spacing
    x_grid = np.arange(width)[None, :]
    line_pattern = ((x_grid + offset) % spacing < 2).repeat(height, axis=0)
    active = np.clip(np.sin(progress * np.pi), 0.0, 1.0)
    line_alpha = (line_pattern & (mask > 0.18)).astype(np.float32) * (0.65 * active)
    line_alpha = cv2.GaussianBlur(line_alpha, (0, 0), sigmaX=1.2, sigmaY=1.2)[:, :, None]
    line_color = np.empty_like(output, dtype=np.float32)
    line_color[:] = (255, 245, 170)
    return np.clip(
        output.astype(np.float32) * (1.0 - line_alpha)
        + line_color * line_alpha,
        0,
        255,
    ).astype(np.uint8)


def victory_landmark_score(points: np.ndarray) -> float:
    """Riconosci indice e medio distesi usando distanze 3D, non l'orientamento."""
    if points.shape != (21, 3) or not np.isfinite(points).all():
        return 0.0

    straightness = []
    extended = []
    folded = []
    lengths = []
    # MCP, PIP, DIP e punta: indice, medio, anulare, mignolo.
    for index, base in enumerate((5, 9, 13, 17)):
        finger = points[base : base + 4]
        length = float(np.linalg.norm(np.diff(finger, axis=0), axis=1).sum())
        if length < 1e-6:
            if index < 2:
                return 0.0
            straightness.append(0.0)
            lengths.append(0.0)
            extended.append(False)
            folded.append(True)
            continue
        ratio = float(np.linalg.norm(finger[3] - finger[0])) / length
        tip_distance = float(np.linalg.norm(finger[3] - points[0]))
        joint_distance = float(np.linalg.norm(finger[1] - points[0]))
        straightness.append(ratio)
        lengths.append(length)
        extended.append(ratio >= 0.82 and tip_distance > joint_distance * 1.10)
        folded.append(ratio < 0.78 or tip_distance <= joint_distance * 1.05)

    # Di lato le dita ripiegate possono apparire dritte ma molto corte:
    # non pretendere articolazioni ben visibili per anulare e mignolo.
    reference_length = min(lengths[:2])
    for index, base in ((2, 13), (3, 17)):
        reach = float(np.linalg.norm(points[base + 3] - points[base]))
        folded[index] = folded[index] or reach < reference_length * 0.65
    if not (extended[0] and extended[1] and folded[2] and folded[3]):
        return 0.0
    # Le due punte devono essere separate; il pollice puo' restare nascosto.
    separation = float(np.linalg.norm(points[8] - points[12]))
    if separation < min(lengths[:2]) * 0.20:
        return 0.0
    # Qualita' geometrica delle dita distese, non probabilita' del classificatore.
    return float(np.clip(min(straightness[:2]), 0.0, 1.0))


def classify_gesture_result(
    result: vision.GestureRecognizerResult,
    image: mp.Image,
) -> tuple[str, float]:
    # I landmark vengono restituiti anche quando il classificatore non riconosce
    # "Victory" (ad esempio con il dorso della mano rivolto alla webcam).
    for landmarks in result.hand_world_landmarks:
        points = np.array([(point.x, point.y, point.z) for point in landmarks])
        score = victory_landmark_score(points)
        if score:
            return "Victory", score
    # Controlla SEMPRE anche la geometria visibile: la profondita' stimata
    # puo' essere imprecisa quando palmo e dita piegate sono nascosti.
    for landmarks in result.hand_landmarks:
        points = np.array([(point.x * image.width, point.y * image.height, 0.0)
                           for point in landmarks])
        score = victory_landmark_score(points)
        if score:
            return "Victory", score
    for categories in result.gestures:
        for category in categories:
            if category.category_name == "Victory" and category.score >= 0.62:
                return "Victory", float(category.score)
    if result.hand_landmarks or result.hand_world_landmarks:
        return "Hand_Detected", 0.0
    return "No_Hand", 0.0


class MultiViewGestureRecognizer:
    """Riprova la rilevazione su fotogrammi ruotati senza alterare il tracking."""

    def __init__(self, primary, alternate):
        self.primary = primary
        self.alternate = alternate
        self.preferred_rotation: int | None = None
        self.preferred_crop: str | None = None
        self.next_retry_ms = 0
        self.retry_cursor = 0
        self.last_hand_landmarks = []
        self.last_name = "No_Hand"
        self.last_score = 0.0

    def recognize_gesture(self, image: mp.Image, timestamp_ms: int) -> tuple[str, float]:
        result = self.primary.recognize_for_video(image, timestamp_ms)
        self.last_hand_landmarks = result.hand_landmarks
        best = classify_gesture_result(result, image)
        self.last_name, self.last_score = best
        if best[0] == "Victory":
            self.preferred_rotation = None
            self.preferred_crop = None
            return best
        if timestamp_ms < self.next_retry_ms:
            return best

        # IMAGE forza una nuova rilevazione; il modello VIDEO resta sul frame
        # originale. Evita timestamp/tracking incoerenti tra rotazioni diverse.
        rgb = image.numpy_view()
        height, width = rgb.shape[:2]
        # Il palmo nella posa laterale e' spesso troppo piccolo nel frame
        # completo. I due ritagli inferiori si sovrappongono e coprono entrambe
        # le mani, anche quando l'immagine e' specchiata.
        crops = {
            "full": rgb,
            "lower_left": rgb[int(height * 0.28):, :int(width * 0.78)],
            "lower_right": rgb[int(height * 0.28):, int(width * 0.22):],
        }
        views = [("full", rotation) for rotation in (0, 1, 3, 2)]
        views += [(crop, rotation) for crop in ("lower_left", "lower_right")
                  for rotation in (0, 1, 3)]
        preferred = (self.preferred_crop, self.preferred_rotation)
        selected = []
        if preferred in views:
            selected.append(preferred)
        # Spread the expensive IMAGE attempts over successive video frames.
        while len(selected) < 3:
            view = views[self.retry_cursor % len(views)]
            self.retry_cursor += 1
            if view not in selected:
                selected.append(view)
        for crop, rotation in selected:
            rotated = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(np.rot90(crops[crop], rotation)),
            )
            result = self.alternate.recognize(rotated)
            candidate = classify_gesture_result(result, rotated)
            if candidate[0] == "Victory":
                self.preferred_rotation = rotation
                self.preferred_crop = crop
                self.next_retry_ms = timestamp_ms + int(GESTURE_INTERVAL_SECONDS * 1000)
                self.last_name, self.last_score = candidate
                return candidate
            if candidate[0] == "Hand_Detected":
                best = candidate
        self.preferred_rotation = None
        self.preferred_crop = None
        self.next_retry_ms = timestamp_ms + int(GESTURE_INTERVAL_SECONDS * 1000)
        self.last_name, self.last_score = best
        return best

    def close(self) -> None:
        try:
            self.primary.close()
        finally:
            self.alternate.close()


class MultiViewHandLandmarker:
    """Track two hands in video and retry difficult views with the hand model."""

    def __init__(self, primary, alternate):
        self.primary = primary
        self.alternate = alternate
        self.preferred_view: tuple[str, int] | None = None
        self.next_retry_ms = 0
        self.last_hand_count = 0
        self.retry_cursor = 0
        self.last_hand_landmarks = []
        self.last_joined = False

    def recognize_joined_hands(self, image: mp.Image, timestamp_ms: int) -> bool:
        result = self.primary.detect_for_video(image, timestamp_ms)
        self.last_hand_landmarks = result.hand_landmarks
        self.last_hand_count = len(result.hand_landmarks)
        if joined_hands_from_landmarks(result.hand_landmarks, image.width, image.height):
            self.preferred_view = None
            self.last_joined = True
            return True
        if timestamp_ms < self.next_retry_ms:
            return False

        rgb = image.numpy_view()
        height, width = rgb.shape[:2]
        views = {
            "full": rgb,
            "center": rgb[int(height * 0.18):, int(width * 0.12):int(width * 0.88)],
            "lower_left": rgb[int(height * 0.22):, :int(width * 0.78)],
            "lower_right": rgb[int(height * 0.22):, int(width * 0.22):],
        }
        attempts = [("full", rotation) for rotation in (1, 3, 2)]
        attempts += [("center", rotation) for rotation in (0, 1, 3)]
        attempts += [("lower_left", 0), ("lower_right", 0)]
        selected = []
        if self.preferred_view in attempts:
            selected.append(self.preferred_view)
        while len(selected) < 2:
            view = attempts[self.retry_cursor % len(attempts)]
            self.retry_cursor += 1
            if view not in selected:
                selected.append(view)
        for crop, rotation in selected:
            rotated = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(np.rot90(views[crop], rotation)),
            )
            result = self.alternate.detect(rotated)
            self.last_hand_count = max(self.last_hand_count, len(result.hand_landmarks))
            if joined_hands_from_landmarks(result.hand_landmarks, rotated.width, rotated.height):
                self.preferred_view = (crop, rotation)
                self.next_retry_ms = timestamp_ms + 90
                return True
        self.preferred_view = None
        self.next_retry_ms = timestamp_ms + 80
        self.last_joined = False
        return False

    def close(self) -> None:
        try:
            self.primary.close()
        finally:
            self.alternate.close()


def recognize_gesture(
    recognizer: MultiViewGestureRecognizer,
    image: mp.Image,
    timestamp_ms: int,
) -> tuple[str, float]:
    if isinstance(recognizer, MultiViewGestureRecognizer):
        return recognizer.recognize_gesture(image, timestamp_ms)
    return classify_gesture_result(recognizer.recognize_for_video(image, timestamp_ms), image)


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)


def draw_gesture_debug(
    frame: np.ndarray,
    recognizer,
    mode: str,
    name: str,
    score: float,
    joined: bool,
) -> np.ndarray:
    """Draw hand landmarks and recognition state on the local preview only."""
    output = frame.copy()
    height, width = output.shape[:2]
    hands = getattr(recognizer, "last_hand_landmarks", [])
    for hand in hands:
        points = [
            (int(np.clip(point.x, 0.0, 1.0) * (width - 1)),
             int(np.clip(point.y, 0.0, 1.0) * (height - 1)))
            for point in hand
        ]
        for start, end in HAND_CONNECTIONS:
            cv2.line(output, points[start], points[end], (0, 220, 255), 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(output, point, 4, (0, 255, 80), -1, cv2.LINE_AA)

    if mode == "normal":
        status = f"Gesture: {name} ({score:.2f}) | Hands: {len(hands)}"
    else:
        status = f"Joined hands: {'YES' if joined else 'NO'} | Hands: {len(hands)}"
    cv2.rectangle(output, (0, 0), (width, 42), (20, 30, 35), -1)
    cv2.putText(output, f"DEBUG ON  |  {status}  |  G: hide", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 245, 247), 1, cv2.LINE_AA)
    return output


def draw_goku_debug(frame: np.ndarray, result: GokuPoseResult) -> np.ndarray:
    """Draw Goku hand/face landmarks and each pose condition on the local preview."""
    output = frame.copy()
    height, width = output.shape[:2]
    for hand in result.hand_landmarks:
        points = [
            (int(np.clip(point.x, 0.0, 1.0) * (width - 1)),
             int(np.clip(point.y, 0.0, 1.0) * (height - 1)))
            for point in hand
        ]
        for start, end in HAND_CONNECTIONS:
            cv2.line(output, points[start], points[end], (0, 220, 255), 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(output, point, 4, (0, 255, 80), -1, cv2.LINE_AA)

    if result.forehead is not None:
        forehead_point = (
            int(np.clip(result.forehead[0], 0.0, 1.0) * (width - 1)),
            int(np.clip(result.forehead[1], 0.0, 1.0) * (height - 1)),
        )
        cv2.circle(output, forehead_point, 11, (255, 80, 180), 2, cv2.LINE_AA)
        cv2.putText(output, "forehead", (forehead_point[0] + 14, forehead_point[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 80, 180), 1, cv2.LINE_AA)

    status = "VALID" if result.valid else "NOT READY"
    text = (
        f"DEBUG ON | Goku pose: {status} | score {result.score:.2f} | "
        f"hold {result.hold_seconds:.1f}/{GOKU_HOLD_SECONDS:.1f}s | G: hide"
    )
    cv2.rectangle(output, (0, 0), (width, 68), (20, 30, 35), -1)
    cv2.putText(output, text, (12, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.53, (240, 245, 247), 1, cv2.LINE_AA)
    cv2.putText(output, result.details, (12, 51), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (179, 220, 230), 1, cv2.LINE_AA)
    return output


def start_vanish_on_gesture(state: AppState, name: str, score: float, now: float) -> bool:
    """Avvia la dissolvenza al primo riconoscimento affidabile della V."""
    if state.phase == "ready" and name == "Victory" and score >= 0.62:
        state.phase = "vanishing"
        state.vanish_started = now
        return True
    return False


def start_naruto_on_join(state: AppState, joined: bool, now: float) -> bool:
    if state.phase == "ready" and joined:
        state.phase = "naruto_vanishing"
        state.vanish_started = now
        return True
    return False


def set_fullscreen(enabled: bool) -> None:
    mode = cv2.WINDOW_FULLSCREEN if enabled else cv2.WINDOW_NORMAL
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, mode)


def open_debug_window(width: int, height: int) -> None:
    """Open the optional diagnostic window separately from the clean output."""
    cv2.namedWindow(DEBUG_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(DEBUG_WINDOW_NAME, width, height)


def close_debug_window() -> None:
    """Close the diagnostic window without affecting the OBS-facing preview."""
    try:
        cv2.destroyWindow(DEBUG_WINDOW_NAME)
    except cv2.error:
        # OpenCV raises when the optional window was already closed manually.
        pass


def self_test() -> int:
    """Check bundled resources and model initialization without opening a webcam."""
    required = (
        SEGMENTER_MODEL, GESTURE_MODEL, HAND_MODEL, FACE_MODEL,
        SOUND_PATH, GOKU_SOUND_PATH,
    )
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing resources: {', '.join(missing)}")
    load_log()
    import pygame.mixer  # The distributed build must contain its MP3 decoder.
    for mode in ("naruto", "goku"):
        segmenter, recognizer = create_tasks(mode)
        try:
            recognizer.close()
        finally:
            segmenter.close()
    print("[Webcam] Self-test completed: models, assets, and audio are present.")
    return 0


def run() -> int:
    args = parse_args()
    if getattr(args, "self_test", False):
        return self_test()
    if args.camera is None:
        args.camera = choose_camera()
        if args.camera is None:
            return 0
    while True:
        mode = args.mode or choose_mode()
        if mode is None:
            return 0
        with ExitStack() as resources:
            result = run_camera(args, resources, mode)
        if result != "menu":
            return result
        args.mode = None


def run_camera(args: argparse.Namespace, resources: ExitStack, mode: str) -> int | str:
    download_models(mode)
    capture = open_camera(args.camera, args.width, args.height)
    resources.callback(capture.release)
    capture.set(cv2.CAP_PROP_FPS, args.fps)
    segmenter, recognizer = create_tasks(mode)
    resources.callback(segmenter.close)
    resources.callback(recognizer.close)
    log_sprite = load_log() if mode == "naruto" else None
    effect_sound = None
    if mode == "naruto":
        effect_sound = VanishSound(SOUND_PATH)
    elif mode == "goku":
        effect_sound = VanishSound(GOKU_SOUND_PATH)
    if effect_sound is not None:
        resources.callback(effect_sound.close)
    resources.callback(cv2.destroyAllWindows)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, args.width, args.height)

    state = AppState(fullscreen=args.fullscreen)
    if state.fullscreen:
        set_fullscreen(True)

    background: np.ndarray | None = None
    background_samples: list[np.ndarray] = []
    last_mask: np.ndarray | None = None
    last_mask_at = float("-inf")
    dissolve_field: np.ndarray | None = None
    sparkle_noise: np.ndarray | None = None
    log_anchor: tuple[int, int, int] | None = None
    log_cache: dict[int, np.ndarray] = {}
    vanished_frame: np.ndarray | None = None
    virtual_camera = None
    virtual_camera_initialized = False
    calibration_remaining = None
    background_ready_notice_until = 0.0
    last_gesture_report = 0.0
    gesture_overlay = False
    debug_window_open = False
    last_gesture_name = "No_Hand"
    last_gesture_score = 0.0
    last_joined_hands = False
    last_goku_result = GokuPoseResult()
    started = time.perf_counter()
    print("[Webcam] Step out of frame. A 3-second countdown starts when the preview opens, then the background is captured for 4 seconds.")
    print("[Webcam] Window controls: SPACE new background, R reappear, V disappear, G gesture debug, M mode menu, F fullscreen, Q quit.")

    try:
        while True:
            ok, camera_frame = capture.read()
            if not ok:
                raise RuntimeError("The webcam stopped returning frames.")

            if not args.no_mirror:
                camera_frame = cv2.flip(camera_frame, 1)

            height, width = camera_frame.shape[:2]
            if not virtual_camera_initialized:
                virtual_camera = open_virtual_camera(args, width, height)
                virtual_camera_initialized = True
                if virtual_camera is not None:
                    resources.callback(virtual_camera.close)
            if mode in ("normal", "goku") and (dissolve_field is None or dissolve_field.shape != (height, width)):
                dissolve_field, sparkle_noise = make_dissolve_fields(height, width)
            now = time.perf_counter()

            elapsed_ms = int((now - started) * 1000)
            mp_image = None

            if (
                mode == "goku"
                and state.phase in ("ready", "vanished")
                and now - state.last_gesture_check >= 0.08
            ):
                state.last_gesture_check = now
                rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                last_goku_result = recognizer.detect(mp_image, elapsed_ms)
                if last_goku_result.trigger:
                    state.phase = "goku_vanishing" if state.phase == "ready" else "goku_reappearing"
                    state.vanish_started = time.perf_counter()
                    vanished_frame = None
                    effect_sound.play()

            if state.phase == "setup_countdown":
                if state.calibration_started <= 0:
                    state.calibration_started = now
                elapsed = now - state.calibration_started
                if elapsed >= SETUP_COUNTDOWN_SECONDS:
                    state.phase = "calibrating"
                    state.calibration_started = now
                    calibration_remaining = None

            if state.phase == "setup_countdown":
                display = camera_frame
            elif state.phase == "calibrating":
                elapsed = now - state.calibration_started
                remaining = max(0, int(np.ceil(CALIBRATION_SECONDS - elapsed)))
                if remaining != calibration_remaining:
                    print(f"[Webcam] Capturing background: {remaining} seconds.")
                    calibration_remaining = remaining
                if elapsed >= CALIBRATION_SECONDS - BACKGROUND_SAMPLE_SECONDS:
                    background_samples.append(camera_frame.copy())
                if elapsed >= CALIBRATION_SECONDS:
                    if background_samples:
                        stack = np.stack(background_samples[-18:], axis=0)
                        background = np.median(stack, axis=0).astype(np.uint8)
                    else:
                        background = camera_frame.copy()
                    background_samples.clear()
                    state.phase = "ready"
                    background_ready_notice_until = now + BACKGROUND_READY_NOTICE_SECONDS
                    print("[Webcam] Background ready. You can step back into frame.")
                display = camera_frame

            elif background is None and state.phase != "setup_countdown":
                state.phase = "calibrating"
                state.calibration_started = now
                display = camera_frame

            elif state.phase == "vanished":
                if vanished_frame is None:
                    vanished_frame = background.copy()
                    if mode == "naruto" and log_anchor is not None:
                        overlay_log(vanished_frame, log_sprite, landed_log_anchor(log_anchor, height),
                                    cache=log_cache)
                display = vanished_frame

            else:
                mask = last_mask
                can_check_gesture = state.phase == "ready"
                if (
                    mode != "goku"
                    and
                    can_check_gesture
                    and now - state.last_gesture_check >=
                    (0.08 if mode == "naruto" else GESTURE_INTERVAL_SECONDS)
                ):
                    state.last_gesture_check = now
                    rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    if mode == "normal":
                        name, score = recognize_gesture(recognizer, mp_image, elapsed_ms)
                        last_gesture_name, last_gesture_score = name, score
                        if args.gesture_debug and now - last_gesture_report >= 1.0:
                            print(
                                f"[Gesto] {name}, valore={score:.2f}, "
                                f"ritaglio={recognizer.preferred_crop or 'full'}, "
                                f"rotazione={recognizer.preferred_rotation or 0}"
                            )
                            last_gesture_report = now
                        start_vanish_on_gesture(state, name, score, now)
                    else:
                        joined = recognizer.recognize_joined_hands(mp_image, elapsed_ms)
                        last_joined_hands = joined
                        if args.gesture_debug and now - last_gesture_report >= 1.0:
                            print(f"[Mani] rilevate={recognizer.last_hand_count}, "
                                  f"unite={joined}, vista={recognizer.preferred_view or 'full'}")
                            last_gesture_report = now
                        if start_naruto_on_join(state, joined, time.perf_counter()):
                            effect_sound.play()

                if state.phase in ("vanishing", "goku_vanishing", "goku_reappearing", "naruto_vanishing"):
                    # Normal dissolve tracks the silhouette at 20 Hz; Naruto
                    # needs it only while the person is still visible.
                    needs_mask = (
                        (mode in ("normal", "goku") and (mask is None or now - last_mask_at >= 0.05))
                        or (mode == "naruto" and (log_anchor is None or now - state.vanish_started < 0.13))
                    )
                    if needs_mask:
                        if mp_image is None:
                            rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        mask = person_mask(segmenter, mp_image, elapsed_ms, width, height)
                        last_mask = mask
                        last_mask_at = now
                    if mode == "naruto" and log_anchor is None:
                        log_anchor = substitution_anchor(mask)

                if mode == "goku" and state.phase in ("goku_vanishing", "goku_reappearing"):
                    age = max(0.0, time.perf_counter() - state.vanish_started)
                    progress = min(1.0, age / GOKU_EFFECT_SECONDS)
                    display = apply_goku_effect(
                        camera_frame,
                        background,
                        mask,
                        progress,
                        dissolve_field,
                        sparkle_noise,
                        disappearing=state.phase == "goku_vanishing",
                    )
                    if progress >= 1.0:
                        if state.phase == "goku_vanishing":
                            state.phase = "vanished"
                            vanished_frame = display
                        else:
                            state.phase = "ready"
                            vanished_frame = None
                elif mode == "naruto" and state.phase == "naruto_vanishing":
                    age = max(0.04, time.perf_counter() - state.vanish_started)
                    display = render_substitution(
                        camera_frame, background, mask, log_sprite, log_anchor, age, log_cache
                    )
                    if age >= EFFECT_SECONDS:
                        state.phase = "vanished"
                        vanished_frame = display
                elif state.phase == "ready":
                    display = camera_frame
                else:
                    progress = 0.0
                    if state.phase == "vanishing" and state.vanish_started is not None:
                        progress = min(
                            1.0, (time.perf_counter() - state.vanish_started) / VANISH_SECONDS
                        )
                        if progress >= 1.0:
                            state.phase = "vanished"
                    display = apply_dissolve(
                        camera_frame, background, mask, progress, dissolve_field, sparkle_noise
                    )
                    if state.phase == "vanished":
                        vanished_frame = display

            if virtual_camera is not None:
                send_virtual_frame(virtual_camera, display)
            # Keep setup guidance in the local window only; the virtual camera
            # continues to receive the clean webcam/effect frame.
            preview_display = display
            if state.phase in ("setup_countdown", "calibrating"):
                phase_elapsed = now - state.calibration_started
                if state.phase == "setup_countdown":
                    countdown = max(1, int(np.ceil(SETUP_COUNTDOWN_SECONDS - phase_elapsed)))
                    preview_display = draw_background_setup_overlay(
                        display, "setup_countdown", remaining=countdown
                    )
                else:
                    capture_elapsed = now - state.calibration_started
                    capture_remaining = max(0, int(np.ceil(CALIBRATION_SECONDS - capture_elapsed)))
                    preview_display = draw_background_setup_overlay(
                        display,
                        "calibrating",
                        remaining=capture_remaining,
                        progress=capture_elapsed / CALIBRATION_SECONDS,
                    )
            elif now < background_ready_notice_until:
                preview_display = draw_background_setup_overlay(display, "ready")
            cv2.imshow(WINDOW_NAME, preview_display)
            if gesture_overlay:
                if not debug_window_open:
                    open_debug_window(width, height)
                    debug_window_open = True
                if mode == "goku":
                    debug_preview = draw_goku_debug(display, last_goku_result)
                else:
                    debug_preview = draw_gesture_debug(
                        display, recognizer, mode, last_gesture_name,
                        last_gesture_score, last_joined_hands,
                    )
                cv2.imshow(DEBUG_WINDOW_NAME, debug_preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                state.phase = "setup_countdown"
                state.calibration_started = now
                state.vanish_started = None
                background_ready_notice_until = 0.0
                log_anchor = None
                last_mask = None
                last_mask_at = float("-inf")
                vanished_frame = None
                background_samples.clear()
                calibration_remaining = None
                print("[Webcam] New background capture starts in 3 seconds, then runs for 4 seconds.")
            elif key == ord("r") and background is not None:
                if background is not None:
                    if mode == "goku":
                        if state.phase == "vanished":
                            state.phase = "goku_reappearing"
                            state.vanish_started = time.perf_counter()
                            vanished_frame = None
                            effect_sound.play()
                    else:
                        state.phase = "ready"
                        state.vanish_started = None
                        vanished_frame = None
                        log_anchor = None
                        last_mask = None
                        last_mask_at = float("-inf")
            elif key == ord("v") and background is not None:
                if mode == "goku":
                    if state.phase == "ready":
                        state.phase = "goku_vanishing"
                        state.vanish_started = time.perf_counter()
                        vanished_frame = None
                        effect_sound.play()
                else:
                    state.phase = "naruto_vanishing" if mode == "naruto" else "vanishing"
                    state.vanish_started = time.perf_counter()
                    if mode == "naruto":
                        effect_sound.play()
                        log_anchor = None
                    vanished_frame = None
            elif key == ord("m"):
                return "menu"
            elif key == ord("f"):
                state.fullscreen = not state.fullscreen
                set_fullscreen(state.fullscreen)
            elif key == ord("g"):
                gesture_overlay = not gesture_overlay
                if not gesture_overlay and debug_window_open:
                    close_debug_window()
                    debug_window_open = False
                print(f"[Webcam] Gesture debug overlay {'enabled' if gesture_overlay else 'disabled'}.")
            if virtual_camera is not None:
                virtual_camera.sleep_until_next_frame()
    finally:
        resources.close()

    return 0


def main() -> int:
    try:
        return run()
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        print(f"\n[Vanish Lab] ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
